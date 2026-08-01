#!/usr/bin/env python3
import rospy
import numpy as np
import os
import sys
import json
import time
import traceback
from geometry_msgs.msg import Twist

# Depending on Unitree ROS installation, the package name could be unitree_legged_msgs or similar
try:
    from unitree_legged_msgs.msg import LowCmd, LowState, MotorCmd, MotorState
except ImportError:
    rospy.logerr("Could not import unitree_legged_msgs! Please ensure your ROS workspace with Unitree messages is sourced.")
    sys.exit(1)

class Go1PolicyDeployNode:
    def __init__(self):
        rospy.init_node('go1_policy_deploy_node')

        # ==========================================
        # 1. Load ROS Parameters & Configs
        # ==========================================
        self.model_path = rospy.get_param('~model_path', '/home/shw/go1_real/model/policy.pt')
        self.is_onnx = self.model_path.endswith('.onnx')
        self.is_numpy = self.model_path.endswith('.npz')
        self.policy_backend = None

        # Control frequencies (env: sim dt 0.005 * decimation 4 = 0.02s -> 50Hz)
        self.loop_rate = rospy.Rate(50)  # 50Hz (dt = 0.02s)

        # Injury setting (-1 for healthy, 0=FL, 1=FR, 2=RL, 3=RR)
        # NOTE: This package ships the Phase-1 *healthy* baseline policy
        #       (experiment_name: unitree_go1_rough_healthy). It was trained with the
        #       peg-leg privileged terms fixed at their healthy defaults, so it does NOT
        #       adapt its gait to an injured leg. Setting injured_leg_idx >= 0 here only
        #       physically frees the corresponding calf motor (so a splint can hold it);
        #       use the Phase-2 / student peg-leg policy for true injury adaptation.
        self.injured_leg_idx = rospy.get_param('~injured_leg_idx', -1)
        rospy.loginfo(f"Injured Leg Index set to: {self.injured_leg_idx} (-1 = Healthy)")

        # Gains settings.
        # The distilled students were TRAINED with a PD actuator at Kp=20 / Kd=0.5
        # (GO1_PD_KP/KD). Running the policy at a much stiffer gain (e.g. 60) was
        # verified in Isaac Sim (sim_test/) to ride high and barely track the
        # command; Kp=20 / Kd=0.5 reproduces the trained gait. The STAND phase can
        # use a stiffer gain to get on its feet; keep the POLICY phase near training.
        self.Kp = rospy.get_param('~Kp', 20.0)
        self.Kd = rospy.get_param('~Kd', 0.5)
        # Stand-up phase gains (stiffer is fine here, only holds the default pose).
        self.stand_Kp = rospy.get_param('~stand_Kp', 60.0)
        self.stand_Kd = rospy.get_param('~stand_Kd', 1.0)
        self.enable_policy = rospy.get_param('~enable_policy', False)
        self.stand_up_time = rospy.get_param('~stand_up_time', 6.0)
        self.action_scale_multiplier = rospy.get_param('~action_scale_multiplier', 1.0)
        self.policy_ramp_time = rospy.get_param('~policy_ramp_time', 5.0)
        self.shutdown_damp_repeats = rospy.get_param('~shutdown_damp_repeats', 20)
        self.shutdown_damp_dt = rospy.get_param('~shutdown_damp_dt', 0.02)
        rospy.loginfo(f"Policy execution enabled: {self.enable_policy}")

        # Remapping Index: Unitree Hardware order -> Isaac Lab articulation order.
        #
        # The Isaac Lab Go1 articulation groups joints BY TYPE (all hips, then all
        # thighs, then all calfs), NOT by leg. This was verified against the sim
        # articulation joint_names via sim_test/ (deployment-parity test). The
        # previous per-leg mapping scrambled every joint on hardware.
        #   Isaac (articulation): FL_hip, FR_hip, RL_hip, RR_hip,
        #                         FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #                         FL_calf, FR_calf, RL_calf, RR_calf
        #   Unitree (SDK):        FR(0,1,2), FL(3,4,5), RR(6,7,8), RL(9,10,11)
        # joint_pos_isaac = joint_pos_unitree[U2I];  target_unitree = target_isaac[I2U].
        # (U2I and I2U are inverse permutations, no longer involutive.)
        self.U2I = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.I2U = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

        # Default Joint Positions in Isaac articulation (type-grouped) order.
        # Order: [FL_hip, FR_hip, RL_hip, RR_hip,
        #         FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #         FL_calf, FR_calf, RL_calf, RR_calf]
        self.default_joint_pos = np.array([
            0.1, -0.1, 0.1, -0.1,     # hips  (L=+0.1, R=-0.1)
            0.8, 0.8, 1.0, 1.0,       # thighs (front 0.8, rear 1.0)
            -1.5, -1.5, -1.5, -1.5    # calfs
        ])

        # Default joint limits in Isaac articulation (type-grouped) order.
        self.joint_pos_min = np.array([
            -1.0, -1.0, -1.0, -1.0,   # hips
            -1.0, -1.0, -1.0, -1.0,   # thighs
            -2.7, -2.7, -2.7, -2.7    # calfs
        ])
        self.joint_pos_max = np.array([
            1.0, 1.0, 1.0, 1.0,       # hips
            3.0, 3.0, 3.0, 3.0,       # thighs
            -0.8, -0.8, -0.8, -0.8    # calfs
        ])

        # ==========================================
        # 2. Load deployment config (obs layout, action scale, privileged defaults)
        # ==========================================
        # Defaults faithful to model/deployment_config.json for the Phase-1 healthy policy.
        self.obs_dim = 51
        self.action_scale = 0.25
        # Privileged peg-leg terms appended after the 48 proprio dims, in layout order.
        # Empty for the proprioception-only (R48) student.
        self.privileged_obs = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # [index, splint_len, friction]
        # Recurrence metadata (set from deployment_config.json; auto-detected at load time as a fallback).
        self.is_recurrent = False
        self.rnn_num_layers = 1
        self.rnn_hidden_size = 256
        self.numpy_lstm = None
        self.load_deployment_config()

        # ==========================================
        # 3. Internal State Buffers
        # ==========================================
        self.current_joint_pos_unitree = np.zeros(12)
        self.current_joint_vel_unitree = np.zeros(12)

        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0]) # [w, x, y, z]
        self.imu_gyro = np.zeros(3)

        self.cmd_vel = np.zeros(3) # [vx, vy, wz]

        self.last_action = np.zeros(12) # Last action output from policy

        # Recurrent LSTM state (used only when the loaded policy is recurrent).
        # ONNX carries these as external inputs; the NumPy path advances them in place;
        # the TorchScript path keeps them internally.
        self.h_state = None
        self.c_state = None

        self.has_state = False
        self.is_running = False
        self._sending_damp = False

        # ==========================================
        # 4. Load Machine Learning Policy
        # ==========================================
        self.load_policy()

        # ==========================================
        # 5. Setup Subscribers & Publishers
        # ==========================================
        self.pub_low_cmd = rospy.Publisher('/low_cmd', LowCmd, queue_size=1)
        rospy.on_shutdown(self.on_ros_shutdown)

        self.sub_low_state = rospy.Subscriber('/low_state', LowState, self.low_state_callback)
        self.sub_cmd_vel = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo("Go1 Policy Deployment Node successfully initialized!")

    def load_deployment_config(self):
        # Look for deployment_config.json next to the model file (falls back to hardcoded defaults).
        cfg_path = os.path.join(os.path.dirname(self.model_path), 'deployment_config.json')
        if not os.path.exists(cfg_path):
            rospy.logwarn(f"deployment_config.json not found at {cfg_path}; using built-in defaults "
                          f"(obs_dim={self.obs_dim}, action_scale={self.action_scale}).")
            return
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)

            self.obs_dim = int(cfg['input']['obs_dim'])
            self.action_scale = float(cfg['action']['action_scale'])

            # Build the privileged term vector from the layout order + healthy defaults so the
            # values always line up with whatever model is dropped in.
            defaults = cfg['input'].get('healthy_privileged_defaults', {})
            priv = [entry for entry in cfg['input']['layout'] if entry.get('group') == 'privileged_obs']
            priv.sort(key=lambda e: e['slice'][0])
            if priv:
                self.privileged_obs = np.array(
                    [float(defaults.get(e['name'], 0.0)) for e in priv], dtype=np.float32)
            else:
                self.privileged_obs = np.zeros(0, dtype=np.float32)  # student: proprioception-only, no privileged

            # Optional recurrence block (LSTM student). Auto-detected again at model load as a fallback.
            rec = cfg.get('recurrent')
            if isinstance(rec, dict) and rec.get('is_recurrent'):
                self.is_recurrent = True
                self.rnn_num_layers = int(rec.get('rnn_num_layers', self.rnn_num_layers))
                self.rnn_hidden_size = int(rec.get('rnn_hidden_size', self.rnn_hidden_size))

            rospy.loginfo(f"Loaded deployment_config.json: obs_dim={self.obs_dim}, "
                          f"action_scale={self.action_scale}, privileged_defaults={self.privileged_obs.tolist()}, "
                          f"recurrent={self.is_recurrent}")
        except Exception as e:
            rospy.logwarn(f"Failed to parse {cfg_path}: {e}. Using built-in defaults.")

    def load_policy(self):
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file not found at path: {self.model_path}")
            sys.exit(1)

        rospy.loginfo(f"Loading model from {self.model_path}...")

        if self.is_numpy:
            self.load_numpy_policy(self.model_path)
        elif self.is_onnx:
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(self.model_path)
                self.onnx_in_names = [i.name for i in self.ort_session.get_inputs()]
                self.onnx_out_names = [o.name for o in self.ort_session.get_outputs()]
                self.onnx_input_name = self.onnx_in_names[0]
                # A recurrent export exposes the LSTM state as extra I/O:
                # (obs, h_in, c_in) -> (actions, h_out, c_out).
                if len(self.onnx_in_names) >= 3:
                    self.is_recurrent = True
                self.policy_backend = 'onnxruntime'
                rospy.loginfo(f"Successfully loaded ONNX model (inputs={self.onnx_in_names}, "
                              f"recurrent={self.is_recurrent}).")
            except Exception as e:
                fallback_path = os.path.join(os.path.dirname(self.model_path), 'policy_numpy.npz')
                if os.path.exists(fallback_path):
                    rospy.logwarn(f"Failed to load ONNX Runtime: {e}. Falling back to NumPy policy at {fallback_path}.")
                    self.load_numpy_policy(fallback_path)
                else:
                    rospy.logerr(f"Failed to load ONNX Runtime: {e}. Try 'pip3 install onnxruntime' or use policy_numpy.npz.")
                    sys.exit(1)
        else:
            try:
                import torch
                self.torch = torch
                self.device = torch.device("cpu")
                self.policy = torch.jit.load(self.model_path, map_location=self.device)
                self.policy.eval()
                # A recurrent JIT export carries its LSTM state internally as buffers
                # named hidden_state / cell_state; forward(obs) updates them in place.
                self.jit_state_buffers = [b for n, b in self.policy.named_buffers()
                                          if n in ('hidden_state', 'cell_state')]
                if self.jit_state_buffers:
                    self.is_recurrent = True
                self.policy_backend = 'torch'
                rospy.loginfo(f"Successfully loaded PyTorch JIT policy (recurrent={self.is_recurrent}).")
            except Exception as e:
                fallback_path = os.path.join(os.path.dirname(self.model_path), 'policy_numpy.npz')
                if os.path.exists(fallback_path):
                    rospy.logwarn(f"Failed to load PyTorch JIT policy: {e}. Falling back to NumPy policy at {fallback_path}.")
                    self.load_numpy_policy(fallback_path)
                else:
                    rospy.logerr(f"Failed to load PyTorch JIT policy: {e}")
                    sys.exit(1)

        self.verify_policy()

    def load_numpy_policy(self, path):
        try:
            data = np.load(path)
            self.numpy_weights = [
                (data['0_weight'].astype(np.float32), data['0_bias'].astype(np.float32)),
                (data['2_weight'].astype(np.float32), data['2_bias'].astype(np.float32)),
                (data['4_weight'].astype(np.float32), data['4_bias'].astype(np.float32)),
                (data['6_weight'].astype(np.float32), data['6_bias'].astype(np.float32)),
            ]
            # A recurrent student npz also carries the single-layer LSTM that runs before the MLP.
            if 'lstm_weight_ih' in data:
                self.numpy_lstm = {
                    'weight_ih': data['lstm_weight_ih'].astype(np.float32),
                    'weight_hh': data['lstm_weight_hh'].astype(np.float32),
                    'bias_ih': data['lstm_bias_ih'].astype(np.float32),
                    'bias_hh': data['lstm_bias_hh'].astype(np.float32),
                }
                self.is_recurrent = True
                self.rnn_hidden_size = int(self.numpy_lstm['weight_hh'].shape[1])
                self.rnn_num_layers = 1
            else:
                self.numpy_lstm = None
            self.policy_backend = 'numpy'
            self.model_path = path
            rospy.loginfo(f"Successfully loaded NumPy policy from {path} (recurrent={self.is_recurrent}).")
        except Exception as e:
            rospy.logerr(f"Failed to load NumPy policy from {path}: {e}")
            sys.exit(1)

    def reset_hidden(self):
        # Zero the recurrent state so an episode starts from the same initial state the policy
        # was evaluated with. No-op for feed-forward policies.
        if not self.is_recurrent:
            return
        if self.policy_backend == 'onnxruntime':
            shape = (self.rnn_num_layers, 1, self.rnn_hidden_size)
            self.h_state = np.zeros(shape, dtype=np.float32)
            self.c_state = np.zeros(shape, dtype=np.float32)
        elif self.policy_backend == 'numpy':
            self.h_state = np.zeros(self.rnn_hidden_size, dtype=np.float32)
            self.c_state = np.zeros(self.rnn_hidden_size, dtype=np.float32)
        elif self.policy_backend == 'torch':
            for b in getattr(self, 'jit_state_buffers', []):
                b.zero_()

    def verify_policy(self):
        # Optional self-test: if policy_metadata.json carries reference outputs, confirm the
        # loaded model reproduces them. Catches a wrong / corrupted / mismatched export early,
        # before any motor command is ever sent.
        meta_path = os.path.join(os.path.dirname(self.model_path), 'policy_metadata.json')
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            # For a recurrent policy the reference is taken from a freshly zeroed hidden state.
            ref = meta.get('reference_zero_obs_action') if self.is_recurrent else None
            if ref is None:
                ref = meta.get('reference_healthy_obs_action')
            if ref is None:
                return
            probe = np.zeros(self.obs_dim, dtype=np.float32)
            if self.privileged_obs.size > 0:
                probe[-self.privileged_obs.size:] = self.privileged_obs  # healthy probe = zeros + privileged defaults
            self.reset_hidden()
            out = self.run_inference(probe)
            if np.allclose(out, np.array(ref, dtype=np.float32), atol=1e-3):
                rospy.loginfo("Policy self-test PASSED (matches reference action).")
            else:
                rospy.logwarn("Policy self-test MISMATCH vs reference action! "
                              "Check that the model and obs layout are consistent before running on hardware.")
        except Exception as e:
            rospy.logwarn(f"Policy self-test skipped ({e}).")
        finally:
            # Leave the hidden state clean for the control loop regardless of the test result.
            self.reset_hidden()

    def low_state_callback(self, msg):
        # 1. Extract Joint Positions and Velocities
        for i in range(12):
            self.current_joint_pos_unitree[i] = msg.motorState[i].q
            self.current_joint_vel_unitree[i] = msg.motorState[i].dq

        # 2. Extract IMU data
        # Unitree LowState IMU format is typically float[4] for quaternion: [w, x, y, z]
        self.imu_quat[0] = msg.imu.quaternion[0] # w
        self.imu_quat[1] = msg.imu.quaternion[1] # x
        self.imu_quat[2] = msg.imu.quaternion[2] # y
        self.imu_quat[3] = msg.imu.quaternion[3] # z

        # Gyroscope: [wx, wy, wz]
        self.imu_gyro[0] = msg.imu.gyroscope[0]
        self.imu_gyro[1] = msg.imu.gyroscope[1]
        self.imu_gyro[2] = msg.imu.gyroscope[2]

        self.has_state = True

    def cmd_vel_callback(self, msg):
        # Update target velocities [vx, vy, wz] from standard geometry_msgs/Twist
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z

    def compute_projected_gravity(self, q):
        # q = [w, x, y, z]. Gravity unit vector (0,0,-1) expressed in the base frame
        # (== Isaac Lab quat_rotate_inverse(q, [0,0,-1])).
        w, x, y, z = q
        gx = 2 * (w * y - x * z)
        gy = -2 * (w * x + y * z)
        gz = 2 * (x * x + y * y) - 1
        return np.array([gx, gy, gz], dtype=np.float32)

    def get_observations(self):
        # Remap current states from Unitree Hardware to Isaac Simulator order
        joint_pos_isaac = self.current_joint_pos_unitree[self.U2I]
        joint_vel_isaac = self.current_joint_vel_unitree[self.U2I]

        # Compute relative positions (joint_pos_rel); joint_vel default offset is 0 so joint_vel_rel == joint_vel
        joint_pos_rel = joint_pos_isaac - self.default_joint_pos

        # Compute projected gravity
        projected_gravity = self.compute_projected_gravity(self.imu_quat)

        # Base linear velocity: not directly measurable on the Go1, but the student
        # obs was trained WITH it. Feeding zeros makes the policy think it is
        # stationary, so it over-accelerates (verified in Isaac Sim: ~0.9 m/s for a
        # 0.5 m/s command). Using the velocity COMMAND as a proxy needs no extra
        # sensing and reproduced near-ideal tracking in sim (~0.46 vs 0.5). If a
        # real body-velocity estimate is available, feed it here instead.
        # [vx_cmd, vy_cmd, 0] in the base frame.
        base_lin_vel = np.array([self.cmd_vel[0], self.cmd_vel[1], 0.0], dtype=np.float32)

        # Assemble the 51-dim observation in the exact deployment_config.json layout order:
        # [0:3]   base_lin_vel
        # [3:6]   base_ang_vel        (IMU gyroscope)
        # [6:9]   projected_gravity
        # [9:12]  velocity_commands   (vx, vy, wz)
        # [12:24] joint_pos_rel
        # [24:36] joint_vel
        # [36:48] actions             (last policy action)
        # [48:51] privileged peg-leg  [peg_leg_index, peg_leg_splint_length, peg_leg_foot_friction]
        parts = [
            base_lin_vel,                                  # 3
            self.imu_gyro.astype(np.float32),              # 3
            projected_gravity,                             # 3
            self.cmd_vel.astype(np.float32),               # 3
            joint_pos_rel.astype(np.float32),              # 12
            joint_vel_isaac.astype(np.float32),            # 12
            self.last_action.astype(np.float32),           # 12
        ]
        if self.privileged_obs.size > 0:
            parts.append(self.privileged_obs)              # 3 (Phase-1 healthy only)

        return np.concatenate(parts)

    def run_inference(self, obs):
        if self.policy_backend == 'numpy':
            if self.is_recurrent and self.numpy_lstm is not None:
                return self._numpy_recurrent_forward(obs)
            x = obs.astype(np.float32)
            for i, (weight, bias) in enumerate(self.numpy_weights):
                x = np.matmul(x, weight.T) + bias
                if i < len(self.numpy_weights) - 1:
                    x = np.where(x > 0.0, x, np.exp(x) - 1.0)
            return x.astype(np.float32)
        elif self.policy_backend == 'onnxruntime':
            obs_in = obs.reshape(1, -1).astype(np.float32)  # (1, obs_dim)
            if self.is_recurrent:
                # Feed and carry the LSTM state: (obs, h_in, c_in) -> (actions, h_out, c_out).
                feeds = {
                    self.onnx_in_names[0]: obs_in,
                    self.onnx_in_names[1]: self.h_state,
                    self.onnx_in_names[2]: self.c_state,
                }
                actions, self.h_state, self.c_state = self.ort_session.run(self.onnx_out_names, feeds)
                return actions.flatten()
            actions = self.ort_session.run(None, {self.onnx_input_name: obs_in})[0]
            return actions.flatten()
        elif self.policy_backend == 'torch':
            with self.torch.inference_mode():
                obs_t = self.torch.from_numpy(obs).float().to(self.device).unsqueeze(0)  # (1, obs_dim)
                # A recurrent JIT export advances its hidden/cell buffers internally.
                actions_t = self.policy(obs_t)
                return actions_t.cpu().numpy().flatten()
        else:
            rospy.logerr("Policy backend is not initialized.")
            sys.exit(1)

    def _numpy_recurrent_forward(self, obs):
        # Single-step LSTM (PyTorch gate order i, f, g, o) then the ELU MLP; advances h/c in place.
        w = self.numpy_lstm
        x = obs.astype(np.float32)
        gates = w['weight_ih'] @ x + w['bias_ih'] + w['weight_hh'] @ self.h_state + w['bias_hh']
        H = self.rnn_hidden_size
        i = 1.0 / (1.0 + np.exp(-gates[:H]))
        f = 1.0 / (1.0 + np.exp(-gates[H:2 * H]))
        g = np.tanh(gates[2 * H:3 * H])
        o = 1.0 / (1.0 + np.exp(-gates[3 * H:]))
        self.c_state = (f * self.c_state + i * g).astype(np.float32)
        self.h_state = (o * np.tanh(self.c_state)).astype(np.float32)
        x = self.h_state
        for j, (weight, bias) in enumerate(self.numpy_weights):
            x = np.matmul(x, weight.T) + bias
            if j < len(self.numpy_weights) - 1:
                x = np.where(x > 0.0, x, np.exp(x) - 1.0)
        return x.astype(np.float32)

    def make_low_cmd(self):
        cmd = LowCmd()
        cmd.head = b'\xfe\xef'
        cmd.levelFlag = 0xFF
        return cmd

    def stand_up(self):
        # Smooth stand up routine
        rospy.loginfo("Initiating Stand Up Phase... Moving slowly to default posture.")
        initial_q = self.current_joint_pos_unitree.copy()
        target_q_isaac = self.default_joint_pos.copy()
        target_q_unitree = target_q_isaac[self.I2U]

        interpolation_time = max(1.0, float(self.stand_up_time))
        steps = int(interpolation_time * 50) # 50Hz * 4s = 200 steps

        for s in range(steps):
            if rospy.is_shutdown():
                break

            alpha = float(s) / steps
            # Linear interpolation between current pose and stand pose
            current_target = (1 - alpha) * initial_q + alpha * target_q_unitree

            # Send commands
            cmd = self.make_low_cmd()
            for idx in range(12):
                cmd.motorCmd[idx].mode = 0x0A # Low-level joint mode
                cmd.motorCmd[idx].q = current_target[idx]
                cmd.motorCmd[idx].dq = 0.0
                cmd.motorCmd[idx].Kp = 10.0 + alpha * (self.stand_Kp - 10.0) # ramp to stand stiffness
                cmd.motorCmd[idx].Kd = self.stand_Kd
                cmd.motorCmd[idx].tau = 0.0

            self.pub_low_cmd.publish(cmd)
            self.loop_rate.sleep()

        rospy.loginfo("Stand Up Complete. Starting Policy Execution!")

    def safety_check(self):
        # Perform orientation limits check. Prevent violent action if tipped over.
        # Projected gravity z value. Straight up gz = -1. Safe limit: gz < -0.5 (about 60 degrees tilt limit)
        projected_gravity = self.compute_projected_gravity(self.imu_quat)
        if projected_gravity[2] > -0.5: # Tilted more than ~60 deg
            rospy.logwarn("SAFETY STOP ACTIVATED: Robot Tilt Threshold Exceeded!")
            return False
        return True

    def send_damp_command(self):
        # Switch robot motors to pure damping mode safely
        cmd = self.make_low_cmd()
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = 0.0
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = 0.0
            cmd.motorCmd[i].Kd = 3.5 # Moderate damping to safe collapse
            cmd.motorCmd[i].tau = 0.0
        self.pub_low_cmd.publish(cmd)

    def send_state_poll_command(self):
        # Some Unitree ROS bridges publish /low_state only from the /low_cmd callback.
        # This zero-torque packet wakes that callback without standing the robot up.
        cmd = self.make_low_cmd()
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = 0.0
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = 0.0
            cmd.motorCmd[i].Kd = 0.0
            cmd.motorCmd[i].tau = 0.0
        self.pub_low_cmd.publish(cmd)

    def send_repeated_damp_commands(self, reason, repeats=None):
        # Repeat damping commands so the final packets are likely to reach the low-level bridge.
        if self._sending_damp:
            return

        self._sending_damp = True
        try:
            repeats = self.shutdown_damp_repeats if repeats is None else repeats
            rospy.logwarn(f"Sending damping commands for safe stop: {reason}")
            for _ in range(max(1, int(repeats))):
                self.send_damp_command()
                time.sleep(max(0.0, float(self.shutdown_damp_dt)))
        except Exception:
            rospy.logerr("Failed while sending damping commands:\n" + traceback.format_exc())
        finally:
            self._sending_damp = False

    def on_ros_shutdown(self):
        self.send_repeated_damp_commands("ROS shutdown")

    def main_loop(self):
        try:
            # Wait for initial state callback to populate buffers
            rospy.loginfo("Waiting for /low_state topics to initialize...")
            while not rospy.is_shutdown() and not self.has_state:
                self.send_state_poll_command()
                rospy.sleep(0.1)

            if rospy.is_shutdown():
                return

            # Execute safety stand up
            self.stand_up()

            # Start the policy episode from a clean recurrent state and zero last action.
            self.last_action = np.zeros(12)
            self.reset_hidden()

            # Initialize main policy loop
            self.is_running = True
            policy_start_time = rospy.Time.now().to_sec()
            if self.enable_policy:
                rospy.loginfo("Starting Policy Control Loop at 50Hz...")
            else:
                rospy.logwarn("Policy inference is DISABLED. Holding default standing posture with zero actions.")

            while not rospy.is_shutdown():
                # 1. Perform orientation safety check
                if not self.safety_check():
                    self.send_repeated_damp_commands("tilt threshold exceeded")
                    break

                # 2. Gather and construct observations
                obs = self.get_observations()

                # 3. Run inference to get raw actions [-1, 1], or hold the default posture.
                if self.enable_policy:
                    raw_actions = self.run_inference(obs)
                    ramp = min(1.0, max(0.0, (rospy.Time.now().to_sec() - policy_start_time)
                                    / max(1e-3, float(self.policy_ramp_time))))
                    raw_actions = raw_actions * float(self.action_scale_multiplier) * ramp
                else:
                    raw_actions = np.zeros(12, dtype=np.float32)

                # 4. Apply Peg Leg Masking for action storing & publishing
                # If a leg is physically pegged/locked, we don't allow the policy to compute commands for it
                if self.injured_leg_idx >= 0:
                    # Calf joint is locked at default splint lock position.
                    # In the type-grouped Isaac order, calf of leg L is at index 8 + L.
                    calf_idx = 8 + self.injured_leg_idx
                    raw_actions[calf_idx] = 0.0 # Action mask = 0 ensures target = default joint position

                # Store last action for next policy step
                self.last_action = raw_actions.copy()

                # 5. Post-process and scale actions
                # action -> target joint pos = default_pos + raw_action * scale
                target_q_isaac = self.default_joint_pos + raw_actions * self.action_scale

                # Clip targets to absolute safety joint limits
                target_q_isaac = np.clip(target_q_isaac, self.joint_pos_min, self.joint_pos_max)

                # 6. Remap target joint positions from Isaac order back to Unitree order
                target_q_unitree = target_q_isaac[self.I2U]

                # 7. Construct LowCmd message
                cmd = self.make_low_cmd()

                for idx in range(12):
                    cmd.motorCmd[idx].mode = 0x0A # low level joint control
                    cmd.motorCmd[idx].q = target_q_unitree[idx]
                    cmd.motorCmd[idx].dq = 0.0
                    cmd.motorCmd[idx].tau = 0.0

                    # Set gains
                    cmd.motorCmd[idx].Kp = self.Kp
                    cmd.motorCmd[idx].Kd = self.Kd

                # 8. Special overrides for the injured leg physically
                if self.injured_leg_idx >= 0:
                    # Get index of injured calf in the Unitree motor array.
                    # calf of leg L is at isaac index 8+L; the isaac->unitree slot
                    # lookup is the inverse of I2U, which is U2I.
                    isaac_calf_idx = 8 + self.injured_leg_idx
                    unitree_calf_idx = self.U2I[isaac_calf_idx]

                    # Options for injured leg motor command:
                    # Setting Kp=0, Kd=0 effectively powers OFF the joint so the user can lock it physically!
                    # Alternatively, keep high stiffness if the motor is electronically locked.
                    # By default we set it to damping mode to prevent fighting a physical lock splint!
                    cmd.motorCmd[unitree_calf_idx].Kp = 0.0
                    cmd.motorCmd[unitree_calf_idx].Kd = 0.0
                    cmd.motorCmd[unitree_calf_idx].tau = 0.0

                # 9. Publish to hardware
                self.pub_low_cmd.publish(cmd)

                # Maintain 50Hz frequency
                self.loop_rate.sleep()
        except rospy.ROSInterruptException:
            self.send_repeated_damp_commands("ROSInterruptException in control loop")
            raise
        except Exception:
            rospy.logerr("Unhandled exception in policy control loop:\n" + traceback.format_exc())
            self.send_repeated_damp_commands("unhandled exception")
            raise
        finally:
            self.is_running = False
            rospy.loginfo("Shutting down deployment node safely...")
            self.send_repeated_damp_commands("main loop exit")

if __name__ == '__main__':
    node = None
    try:
        node = Go1PolicyDeployNode()
        node.main_loop()
    except KeyboardInterrupt:
        if node is not None:
            node.send_repeated_damp_commands("KeyboardInterrupt")
    except rospy.ROSInterruptException:
        if node is not None:
            node.send_repeated_damp_commands("ROSInterruptException")
    except Exception:
        if node is not None:
            node.send_repeated_damp_commands("top-level exception")
        raise
