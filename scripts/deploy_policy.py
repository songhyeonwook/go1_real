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
        # The sim used a learned ActuatorNetMLP (no explicit stiffness/damping), so there is
        # no exact PD equivalent. Kp=25 / Kd=0.5 is the conventional Go1 position-control gain
        # used for sim-to-real of this policy. Tune on hardware if the gait feels stiff/soft.
        self.Kp = rospy.get_param('~Kp', 25.0)
        self.Kd = rospy.get_param('~Kd', 0.5)
        self.shutdown_damp_repeats = rospy.get_param('~shutdown_damp_repeats', 20)
        self.shutdown_damp_dt = rospy.get_param('~shutdown_damp_dt', 0.02)

        # Remapping Index: Unitree Hardware Order [FR, FL, RR, RL] -> Isaac Order [FL, FR, RL, RR]
        # Isaac indices: FL(0,1,2), FR(3,4,5), RL(6,7,8), RR(9,10,11)
        # Unitree indices: FR(0,1,2), FL(3,4,5), RR(6,7,8), RL(9,10,11)
        self.U2I = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.I2U = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # Involutive mapping

        # Default Joint Positions in Isaac Order
        # Order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        self.default_joint_pos = np.array([
            0.1, 0.8, -1.5,   # FL
            -0.1, 0.8, -1.5,  # FR
            0.1, 1.0, -1.5,   # RL
            -0.1, 1.0, -1.5   # RR
        ])

        # Default joint limits in Isaac Order
        self.joint_pos_min = np.array([
            -1.0, -1.0, -2.7,  # FL
            -1.0, -1.0, -2.7,  # FR
            -1.0, -1.0, -2.7,  # RL
            -1.0, -1.0, -2.7   # RR
        ])
        self.joint_pos_max = np.array([
            1.0, 3.0, -0.8,  # FL
            1.0, 3.0, -0.8,  # FR
            1.0, 3.0, -0.8,  # RL
            1.0, 3.0, -0.8   # RR
        ])

        # ==========================================
        # 2. Load deployment config (obs layout, action scale, privileged defaults)
        # ==========================================
        # Defaults faithful to model/deployment_config.json for the Phase-1 healthy policy.
        self.obs_dim = 51
        self.action_scale = 0.25
        # Privileged peg-leg terms appended after the 48 proprio dims, in layout order.
        self.privileged_obs = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # [index, splint_len, friction]
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

            rospy.loginfo(f"Loaded deployment_config.json: obs_dim={self.obs_dim}, "
                          f"action_scale={self.action_scale}, privileged_defaults={self.privileged_obs.tolist()}")
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
                # This is a feed-forward ActorCritic (no recurrent state): single input -> single output.
                self.onnx_input_name = self.ort_session.get_inputs()[0].name
                self.policy_backend = 'onnxruntime'
                rospy.loginfo(f"Successfully loaded ONNX model (input='{self.onnx_input_name}').")
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
                self.policy_backend = 'torch'
                rospy.loginfo("Successfully loaded PyTorch JIT policy.")
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
            self.policy_backend = 'numpy'
            self.model_path = path
            rospy.loginfo(f"Successfully loaded NumPy policy from {path}.")
        except Exception as e:
            rospy.logerr(f"Failed to load NumPy policy from {path}: {e}")
            sys.exit(1)

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
            ref = meta.get('reference_healthy_obs_action')
            if ref is None:
                return
            probe = np.zeros(self.obs_dim, dtype=np.float32)
            probe[-len(self.privileged_obs):] = self.privileged_obs  # healthy probe = zeros + privileged defaults
            out = self.run_inference(probe)
            if np.allclose(out, np.array(ref, dtype=np.float32), atol=1e-3):
                rospy.loginfo("Policy self-test PASSED (matches reference healthy action).")
            else:
                rospy.logwarn("Policy self-test MISMATCH vs reference action! "
                              "Check that the model and obs layout are consistent before running on hardware.")
        except Exception as e:
            rospy.logwarn(f"Policy self-test skipped ({e}).")

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

        # Base linear velocity: not directly measurable on the Go1. The policy was trained with it,
        # so we feed zeros (best available estimate). This is a known sim-to-real approximation.
        base_lin_vel = np.zeros(3, dtype=np.float32)

        # Assemble the 51-dim observation in the exact deployment_config.json layout order:
        # [0:3]   base_lin_vel
        # [3:6]   base_ang_vel        (IMU gyroscope)
        # [6:9]   projected_gravity
        # [9:12]  velocity_commands   (vx, vy, wz)
        # [12:24] joint_pos_rel
        # [24:36] joint_vel
        # [36:48] actions             (last policy action)
        # [48:51] privileged peg-leg  [peg_leg_index, peg_leg_splint_length, peg_leg_foot_friction]
        obs = np.concatenate([
            base_lin_vel,                                  # 3
            self.imu_gyro.astype(np.float32),              # 3
            projected_gravity,                             # 3
            self.cmd_vel.astype(np.float32),               # 3
            joint_pos_rel.astype(np.float32),              # 12
            joint_vel_isaac.astype(np.float32),            # 12
            self.last_action.astype(np.float32),           # 12
            self.privileged_obs                            # 3
        ])

        return obs

    def run_inference(self, obs):
        # Feed-forward ActorCritic inference (no recurrent hidden state).
        if self.policy_backend == 'numpy':
            x = obs.astype(np.float32)
            for i, (weight, bias) in enumerate(self.numpy_weights):
                x = np.matmul(x, weight.T) + bias
                if i < len(self.numpy_weights) - 1:
                    x = np.where(x > 0.0, x, np.exp(x) - 1.0)
            return x.astype(np.float32)
        elif self.policy_backend == 'onnxruntime':
            obs_in = obs.reshape(1, -1).astype(np.float32)  # (1, obs_dim)
            actions = self.ort_session.run(None, {self.onnx_input_name: obs_in})[0]
            return actions.flatten()
        elif self.policy_backend == 'torch':
            with self.torch.inference_mode():
                obs_t = self.torch.from_numpy(obs).float().to(self.device).unsqueeze(0)  # (1, obs_dim)
                actions_t = self.policy(obs_t)
                return actions_t.cpu().numpy().flatten()
        else:
            rospy.logerr("Policy backend is not initialized.")
            sys.exit(1)

    def stand_up(self):
        # Smooth stand up routine
        rospy.loginfo("Initiating Stand Up Phase... Moving slowly to default posture.")
        initial_q = self.current_joint_pos_unitree.copy()
        target_q_isaac = self.default_joint_pos.copy()
        target_q_unitree = target_q_isaac[self.I2U]

        interpolation_time = 4.0 # 4 seconds to stand up
        steps = int(interpolation_time * 50) # 50Hz * 4s = 200 steps

        for s in range(steps):
            if rospy.is_shutdown():
                break

            alpha = float(s) / steps
            # Linear interpolation between current pose and stand pose
            current_target = (1 - alpha) * initial_q + alpha * target_q_unitree

            # Send commands
            cmd = LowCmd()
            cmd.levelFlag = 0xFF
            for idx in range(12):
                cmd.motorCmd[idx].mode = 0x0A # Low-level joint mode
                cmd.motorCmd[idx].q = current_target[idx]
                cmd.motorCmd[idx].dq = 0.0
                cmd.motorCmd[idx].Kp = 10.0 + alpha * (self.Kp - 10.0) # Gradually ramp stiffness
                cmd.motorCmd[idx].Kd = 1.0
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
        cmd = LowCmd()
        cmd.levelFlag = 0xFF
        for i in range(12):
            cmd.motorCmd[i].mode = 0x0A
            cmd.motorCmd[i].q = 0.0
            cmd.motorCmd[i].dq = 0.0
            cmd.motorCmd[i].Kp = 0.0
            cmd.motorCmd[i].Kd = 3.5 # Moderate damping to safe collapse
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
                rospy.sleep(0.1)

            if rospy.is_shutdown():
                return

            # Execute safety stand up
            self.stand_up()

            # Initialize main policy loop
            self.is_running = True
            rospy.loginfo("Starting Control Loop at 50Hz...")

            while not rospy.is_shutdown():
                # 1. Perform orientation safety check
                if not self.safety_check():
                    self.send_repeated_damp_commands("tilt threshold exceeded")
                    break

                # 2. Gather and construct observations
                obs = self.get_observations()

                # 3. Run inference to get raw actions [-1, 1]
                raw_actions = self.run_inference(obs)

                # 4. Apply Peg Leg Masking for action storing & publishing
                # If a leg is physically pegged/locked, we don't allow the policy to compute commands for it
                if self.injured_leg_idx >= 0:
                    # Calf joint is locked at default splint lock position
                    # In Isaac Order, calf joint is leg_idx * 3 + 2
                    calf_idx = self.injured_leg_idx * 3 + 2
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
                cmd = LowCmd()
                cmd.levelFlag = 0xFF

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
                    # Get index of injured calf in Unitree array
                    isaac_calf_idx = self.injured_leg_idx * 3 + 2
                    unitree_calf_idx = self.I2U[isaac_calf_idx]

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
