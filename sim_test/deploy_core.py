#!/usr/bin/env python3
"""ROS-free port of the deployment pipeline in scripts/deploy_policy.py.

This module reproduces *exactly* the observation assembly, policy inference and
action post-processing that the real Go1 node performs, but without any rospy
dependency so it can be exercised inside Isaac Sim (or a plain unit test).

Keep this in sync with scripts/deploy_policy.py. The constants and math below are
copied verbatim from that node; the class-level self-test (`selftest`) checks the
loaded model against the reference action in policy_metadata.json, the same guard
deploy_policy.py runs before it ever commands a motor.
"""

from __future__ import annotations

import json
import os

import numpy as np


class DeployPolicyCore:
    """The hardware deployment pipeline, minus ROS.

    Feed it hardware-equivalent raw signals (joint pos/vel in *Unitree* order,
    IMU quaternion [w,x,y,z], IMU gyro, velocity command) via get_observations();
    it returns the exact obs vector the network is given on the robot. run_inference
    then produces the raw policy action, and postprocess_action turns that into the
    joint-position targets that would be published to /low_cmd (Unitree order).
    """

    # Unitree Hardware order -> Isaac Lab articulation order (type-grouped:
    # all hips, all thighs, all calfs). Kept identical to the patched
    # scripts/deploy_policy.py. U2I and I2U are inverse permutations.
    #   joint_pos_isaac = joint_pos_unitree[U2I];  target_unitree = target_isaac[I2U]
    U2I = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    I2U = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

    # Default joint positions in Isaac articulation (type-grouped) order:
    # [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh..RR_thigh, FL_calf..RR_calf]
    DEFAULT_JOINT_POS = np.array([
        0.1, -0.1, 0.1, -0.1,    # hips
        0.8, 0.8, 1.0, 1.0,      # thighs
        -1.5, -1.5, -1.5, -1.5,  # calfs
    ], dtype=np.float32)

    JOINT_POS_MIN = np.array([
        -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0,
        -2.7, -2.7, -2.7, -2.7,
    ], dtype=np.float32)
    JOINT_POS_MAX = np.array([
        1.0, 1.0, 1.0, 1.0,
        3.0, 3.0, 3.0, 3.0,
        -0.8, -0.8, -0.8, -0.8,
    ], dtype=np.float32)

    def __init__(self, model_dir, backend="numpy"):
        """backend: 'numpy' (policy_numpy.npz) or 'torch' (policy.pt JIT)."""
        self.model_dir = model_dir
        self.backend = backend

        # config-driven fields (populated by load_deployment_config)
        self.obs_dim = 48
        self.action_scale = 0.25
        self.privileged_obs = np.zeros(0, dtype=np.float32)
        self.is_recurrent = False
        self.rnn_num_layers = 1
        self.rnn_hidden_size = 256

        # recurrent state
        self.h_state = None
        self.c_state = None

        self.load_deployment_config()
        self.load_policy()

    # ------------------------------------------------------------------ config
    def load_deployment_config(self):
        cfg_path = os.path.join(self.model_dir, "deployment_config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        self.obs_dim = int(cfg["input"]["obs_dim"])
        self.action_scale = float(cfg["action"]["action_scale"])

        defaults = cfg["input"].get("healthy_privileged_defaults", {})
        priv = [e for e in cfg["input"]["layout"] if e.get("group") == "privileged_obs"]
        priv.sort(key=lambda e: e["slice"][0])
        if priv:
            self.privileged_obs = np.array(
                [float(defaults.get(e["name"], 0.0)) for e in priv], dtype=np.float32)
        else:
            self.privileged_obs = np.zeros(0, dtype=np.float32)

        rec = cfg.get("recurrent")
        if isinstance(rec, dict) and rec.get("is_recurrent"):
            self.is_recurrent = True
            self.rnn_num_layers = int(rec.get("rnn_num_layers", self.rnn_num_layers))
            self.rnn_hidden_size = int(rec.get("rnn_hidden_size", self.rnn_hidden_size))

    # ------------------------------------------------------------------- model
    def load_policy(self):
        if self.backend == "numpy":
            self._load_numpy_policy(os.path.join(self.model_dir, "policy_numpy.npz"))
        elif self.backend == "torch":
            import torch
            self.torch = torch
            self.device = torch.device("cpu")
            self.policy = torch.jit.load(os.path.join(self.model_dir, "policy.pt"),
                                         map_location=self.device)
            self.policy.eval()
            self.jit_state_buffers = [b for n, b in self.policy.named_buffers()
                                      if n in ("hidden_state", "cell_state")]
            if self.jit_state_buffers:
                self.is_recurrent = True
        else:
            raise ValueError(f"unknown backend {self.backend}")
        self.reset_hidden()

    def _load_numpy_policy(self, path):
        data = np.load(path)
        self.numpy_weights = [
            (data["0_weight"].astype(np.float32), data["0_bias"].astype(np.float32)),
            (data["2_weight"].astype(np.float32), data["2_bias"].astype(np.float32)),
            (data["4_weight"].astype(np.float32), data["4_bias"].astype(np.float32)),
            (data["6_weight"].astype(np.float32), data["6_bias"].astype(np.float32)),
        ]
        if "lstm_weight_ih" in data:
            self.numpy_lstm = {
                "weight_ih": data["lstm_weight_ih"].astype(np.float32),
                "weight_hh": data["lstm_weight_hh"].astype(np.float32),
                "bias_ih": data["lstm_bias_ih"].astype(np.float32),
                "bias_hh": data["lstm_bias_hh"].astype(np.float32),
            }
            self.is_recurrent = True
            self.rnn_hidden_size = int(self.numpy_lstm["weight_hh"].shape[1])
            self.rnn_num_layers = 1
        else:
            self.numpy_lstm = None

    def reset_hidden(self):
        if not self.is_recurrent:
            return
        if self.backend == "numpy":
            self.h_state = np.zeros(self.rnn_hidden_size, dtype=np.float32)
            self.c_state = np.zeros(self.rnn_hidden_size, dtype=np.float32)
        elif self.backend == "torch":
            for b in getattr(self, "jit_state_buffers", []):
                b.zero_()

    # ------------------------------------------------------------ observations
    @staticmethod
    def compute_projected_gravity(q):
        # q = [w, x, y, z] == Isaac Lab quat_rotate_inverse(q, [0,0,-1]).
        w, x, y, z = q
        gx = 2 * (w * y - x * z)
        gy = -2 * (w * x + y * z)
        gz = 2 * (x * x + y * y) - 1
        return np.array([gx, gy, gz], dtype=np.float32)

    def get_observations(self, joint_pos_unitree, joint_vel_unitree, imu_quat,
                         imu_gyro, cmd_vel, last_action):
        """Assemble the deployment obs vector exactly as deploy_policy.py does.

        joint_pos_unitree / joint_vel_unitree : 12-vec in Unitree hardware order
        imu_quat : [w,x,y,z]; imu_gyro : [wx,wy,wz]; cmd_vel : [vx,vy,wz]
        last_action : previous raw policy action (12)
        """
        joint_pos_unitree = np.asarray(joint_pos_unitree, dtype=np.float32)
        joint_vel_unitree = np.asarray(joint_vel_unitree, dtype=np.float32)

        joint_pos_isaac = joint_pos_unitree[self.U2I]
        joint_vel_isaac = joint_vel_unitree[self.U2I]
        joint_pos_rel = joint_pos_isaac - self.DEFAULT_JOINT_POS

        projected_gravity = self.compute_projected_gravity(imu_quat)

        # Base linear velocity not measurable on hardware; use the velocity command
        # as a proxy (mirrors patched deploy_policy.py; see sim_test/README.md).
        cmd_vel = np.asarray(cmd_vel, dtype=np.float32)
        base_lin_vel = np.array([cmd_vel[0], cmd_vel[1], 0.0], dtype=np.float32)

        parts = [
            base_lin_vel,
            np.asarray(imu_gyro, dtype=np.float32),
            projected_gravity,
            np.asarray(cmd_vel, dtype=np.float32),
            joint_pos_rel.astype(np.float32),
            joint_vel_isaac.astype(np.float32),
            np.asarray(last_action, dtype=np.float32),
        ]
        if self.privileged_obs.size > 0:
            parts.append(self.privileged_obs)
        return np.concatenate(parts)

    # --------------------------------------------------------------- inference
    def run_inference(self, obs):
        if self.backend == "numpy":
            if self.is_recurrent and self.numpy_lstm is not None:
                return self._numpy_recurrent_forward(obs)
            x = obs.astype(np.float32)
            for i, (weight, bias) in enumerate(self.numpy_weights):
                x = np.matmul(x, weight.T) + bias
                if i < len(self.numpy_weights) - 1:
                    x = np.where(x > 0.0, x, np.exp(x) - 1.0)
            return x.astype(np.float32)
        elif self.backend == "torch":
            with self.torch.inference_mode():
                obs_t = self.torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                actions_t = self.policy(obs_t)
                return actions_t.cpu().numpy().flatten()

    def _numpy_recurrent_forward(self, obs):
        w = self.numpy_lstm
        x = obs.astype(np.float32)
        gates = w["weight_ih"] @ x + w["bias_ih"] + w["weight_hh"] @ self.h_state + w["bias_hh"]
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

    # ------------------------------------------------------------- action post
    def postprocess_action(self, raw_actions, injured_leg_idx=-1):
        """raw policy action -> (target_q_isaac, target_q_unitree), exactly as node."""
        raw_actions = np.asarray(raw_actions, dtype=np.float32).copy()
        if injured_leg_idx >= 0:
            raw_actions[injured_leg_idx * 3 + 2] = 0.0
        target_q_isaac = self.DEFAULT_JOINT_POS + raw_actions * self.action_scale
        target_q_isaac = np.clip(target_q_isaac, self.JOINT_POS_MIN, self.JOINT_POS_MAX)
        target_q_unitree = target_q_isaac[self.I2U]
        return target_q_isaac, target_q_unitree

    # --------------------------------------------------------------- self-test
    def selftest(self, atol=1e-3):
        """Reproduce deploy_policy.py's verify_policy() against policy_metadata.json.

        Returns (passed: bool|None, max_abs_err: float|None). None if no reference.
        """
        meta_path = os.path.join(self.model_dir, "policy_metadata.json")
        if not os.path.exists(meta_path):
            return None, None
        with open(meta_path, "r") as f:
            meta = json.load(f)
        ref = meta.get("reference_zero_obs_action") if self.is_recurrent else None
        if ref is None:
            ref = meta.get("reference_healthy_obs_action")
        if ref is None:
            return None, None
        probe = np.zeros(self.obs_dim, dtype=np.float32)
        if self.privileged_obs.size > 0:
            probe[-self.privileged_obs.size:] = self.privileged_obs
        self.reset_hidden()
        out = self.run_inference(probe)
        self.reset_hidden()
        err = float(np.max(np.abs(out - np.array(ref, dtype=np.float32))))
        return bool(err <= atol), err


if __name__ == "__main__":
    # Standalone self-test across the three deployable students (no ROS/sim needed).
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = {
        "antalgic": os.path.join(root, "antalgic", "exported"),
        "fault_tolerant": os.path.join(root, "fault_tolerant", "exported"),
        "symmetry": os.path.join(root, "symmetry", "exported"),
    }
    backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    ok_all = True
    for name, d in dirs.items():
        core = DeployPolicyCore(d, backend=backend)
        passed, err = core.selftest()
        tag = "n/a" if passed is None else ("PASS" if passed else "FAIL")
        ok_all = ok_all and (passed is not False)
        print(f"[{tag}] {name:15s} backend={backend} obs_dim={core.obs_dim} "
              f"recurrent={core.is_recurrent} max_err={err}")
    sys.exit(0 if ok_all else 1)
