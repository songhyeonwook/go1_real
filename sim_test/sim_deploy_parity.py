#!/usr/bin/env python3
"""Deployment-parity test (Test B) for the go1_real students, inside Isaac Sim.

Goal: before touching hardware, prove that the *exact artifact + pipeline that will
run on the robot* -- the exported policy (policy_numpy.npz / policy.pt) plus the
observation assembly, joint remapping and action post-processing in
scripts/deploy_policy.py (ported verbatim to sim_test/deploy_core.py) -- reproduces
the trained policy and actually walks the robot in the training environment.

It does three things each control step:

  1. Net-export fidelity   : feed the SAME env observation to the RSL-RL reference
                             policy and to the deployed artifact -> action MSE.
                             Isolates the ONNX/npz/JIT export from everything else.
  2. Obs-assembly fidelity : rebuild the observation the way deploy_policy.py does,
                             from hardware-emulated raw sensor signals, and diff it
                             block-by-block against the env's own observation.
                             Catches obs-layout, joint-order and projected-gravity bugs.
  3. Closed-loop behavior  : DRIVE the sim with the full hardware pipeline (sensor ->
                             deploy_core -> motor targets, remapped by joint name the
                             same way the Unitree bridge would) and log whether the
                             robot stays up and tracks the velocity command.

It also runs a JOINT-ORDER AUDIT: the sim articulation joint order vs the per-leg
order deploy_policy.py hardcodes. A mismatch there silently scrambles every obs and
action on hardware, so it is reported loudly.

Run via IsaacLab's python, e.g.
  ~/IsaacLab/isaaclab.sh -p sim_test/sim_deploy_parity.py \
      --model_dir antalgic/exported --backend numpy --num_steps 600 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# go1_peg local imports live next to play.py
GO1_PEG_RSL = os.path.expanduser("~/go1_peg/scripts/rsl_rl")
if GO1_PEG_RSL not in sys.path:
    sys.path.insert(0, GO1_PEG_RSL)
# our ROS-free deployment core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Go1 deployment-parity test in Isaac Sim.")
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distill_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=600, help="control steps to run")
parser.add_argument("--model_dir", type=str, required=True,
                    help="dir with policy_numpy.npz/policy.pt + deployment_config.json "
                         "(relative to repo root ok), e.g. antalgic/exported")
parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"],
                    help="deployment inference backend to exercise")
parser.add_argument("--cmd_vx", type=float, default=0.5, help="forced forward velocity command")
parser.add_argument("--linvel_source", type=str, default="command",
                    choices=["zero", "command", "true"],
                    help="base_lin_vel fed to the deploy policy: zero (current), "
                         "command proxy, or true sim velocity (perfect estimator)")
parser.add_argument("--drive_mode", type=str, default="fixed",
                    choices=["asis", "fixed", "reference"],
                    help="asis = current deploy_policy.py path (bug included); "
                         "fixed = corrected name-based joint mapping; "
                         "reference = the raw RSL-RL policy")
# Reproduce the real bring-up: stand at (stand_kp/stand_kd), then run the policy at
# (policy_kp/policy_kd). The hardware low-level PD gains differ from the training
# actuator (Kp=20/Kd=0.5), so this is where a stiff-gain instability shows up.
parser.add_argument("--stand_kp", type=float, default=60.0)
parser.add_argument("--stand_kd", type=float, default=1.0)
parser.add_argument("--policy_kp", type=float, default=60.0)
parser.add_argument("--policy_kd", type=float, default=1.0)
parser.add_argument("--stand_steps", type=int, default=100,
                    help="control steps holding the default pose before the policy")
parser.add_argument("--flat_terrain", action="store_true", default=False,
                    help="force plane terrain (fair gait comparison across policies)")
parser.add_argument("--report", type=str, default=None, help="output JSON path")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import json  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import go1_lab.tasks  # noqa: F401,E402

from peg_leg_action_wrapper import PegLegActionMaskWrapper  # noqa: E402
from deploy_core import DeployPolicyCore  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Physical Unitree hardware joint order and the per-leg order deploy_policy assumes.
UNITREE_ORDER = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
                 "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
ASSUMED_ISAAC_ORDER = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
                       "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]

OBS_BLOCKS = [("base_lin_vel", 0, 3), ("base_ang_vel", 3, 6), ("projected_gravity", 6, 9),
              ("velocity_commands", 9, 12), ("joint_pos", 12, 24), ("joint_vel", 24, 36),
              ("last_action", 36, 48)]


def _leg_key(joint_name):
    """'FL_hip_joint' -> 'FL_hip'. Robust to the Isaac Lab naming convention."""
    return joint_name.replace("_joint", "")


def build_name_maps(artic_names):
    """Return index maps between articulation order and the canonical orders, by name."""
    keys = [_leg_key(n) for n in artic_names]
    missing = [k for k in UNITREE_ORDER if k not in keys]
    if missing:
        raise RuntimeError(f"sim joints {keys} do not contain expected Go1 joints {missing}")
    # unitree slot i  <- articulation index
    u2a = np.array([keys.index(k) for k in UNITREE_ORDER], dtype=np.int64)
    a2u = np.array([UNITREE_ORDER.index(k) for k in keys], dtype=np.int64)
    return keys, u2a, a2u


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    model_dir = args_cli.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(REPO_ROOT, model_dir)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # --- Clean the env for a parity test: no obs noise, healthy (no peg leg) ------
    try:
        env_cfg.observations.policy.enable_corruption = False
    except Exception:
        pass
    # disable peg-leg injection so we test healthy locomotion first
    for ev in ("randomize_peg_leg_actuation", "enforce_peg_leg"):
        if hasattr(env_cfg.events, ev):
            setattr(env_cfg.events, ev, None)
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "peg_leg_difficulty"):
        env_cfg.curriculum.peg_leg_difficulty = None

    # Fixed straight-forward command + face +x on reset, so actual_vx is directly
    # comparable to the commanded vx (clean walking test, no random heading/yaw).
    try:
        cv = args_cli.cmd_vx
        rng = env_cfg.commands.base_velocity.ranges
        rng.lin_vel_x = (cv, cv)
        rng.lin_vel_y = (0.0, 0.0)
        rng.ang_vel_z = (0.0, 0.0)
        rng.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    except Exception as e:
        print(f"[WARN] could not force velocity command: {e}")
    try:
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    except Exception:
        pass
    if args_cli.flat_terrain:
        try:
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None
            if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
            print("[INFO] forced plane terrain")
        except Exception as e:
            print(f"[WARN] could not flatten terrain: {e}")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        vsub = f"{args_cli.drive_mode}_pkp{int(args_cli.policy_kp)}"
        env = gym.wrappers.RecordVideo(
            env, video_folder=os.path.join(model_dir, "sim_test_videos", vsub),
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length, disable_logger=True)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- reference RSL-RL policy (ground truth) ---------------------------------
    # Mirror train.py: strip cfg keys the installed rsl_rl PPO doesn't accept
    # (newer schema fields). Without this, loading an OnPolicyRunner checkpoint
    # fails with "PPO.__init__() got an unexpected keyword argument 'optimizer'".
    agent_cfg_dict = agent_cfg.to_dict()
    alg_cfg = agent_cfg_dict.get("algorithm", {})
    for taboo in ("optimizer", "config_class", "share_cnn_encoders"):
        alg_cfg.pop(taboo, None)
    if agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if resume_path is None:
        raise ValueError("pass --checkpoint <student.pt> (the RSL-RL checkpoint)")
    print(f"[INFO] loading reference checkpoint: {resume_path}")
    runner.load(resume_path)
    reference_policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- deployed artifact (what runs on the robot) -----------------------------
    # Optional: a checkpoint without exported artifacts (e.g. a fresh phase-1 run)
    # can still be characterized in --drive_mode reference (gait only, no parity).
    deploy = deploy_shadow = None
    st_ok = st_err = None
    try:
        deploy = DeployPolicyCore(model_dir, backend=args_cli.backend)
        deploy_shadow = DeployPolicyCore(model_dir, backend=args_cli.backend)
        st_ok, st_err = deploy.selftest()
        print(f"[INFO] deploy backend={args_cli.backend} obs_dim={deploy.obs_dim} "
              f"recurrent={deploy.is_recurrent} selftest={'PASS' if st_ok else st_ok} err={st_err}")
    except Exception as e:
        if args_cli.drive_mode == "reference":
            print(f"[WARN] no deploy artifact in {model_dir} ({e}); "
                  f"reference mode = gait/behavior only, parity metrics skipped.")
        else:
            raise

    # --- JOINT-ORDER AUDIT ------------------------------------------------------
    robot = env.unwrapped.scene["robot"]
    artic_names = list(robot.joint_names)
    keys, u2a, a2u = build_name_maps(artic_names)
    order_match = (keys == ASSUMED_ISAAC_ORDER)

    # Foot contact sensor indices (for the gait diagram / duty factor).
    FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    contact_sensor = env.unwrapped.scene.sensors.get("contact_forces")
    foot_idx = None
    if contact_sensor is not None:
        foot_idx = []
        for f in FEET:
            hit = [i for i, b in enumerate(contact_sensor.body_names) if f in b]
            foot_idx.append(hit[0] if hit else -1)
        print(f"[INFO] foot contact indices {dict(zip(FEET, foot_idx))}")
    default_artic = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    print("\n================= JOINT-ORDER AUDIT =================")
    print(f"  sim articulation order : {keys}")
    print(f"  deploy assumed  order  : {ASSUMED_ISAAC_ORDER}")
    print(f"  MATCH                  : {order_match}")
    if not order_match:
        perm = [keys.index(k) for k in ASSUMED_ISAAC_ORDER]
        print("  !! MISMATCH: deploy_policy.py's hardcoded Isaac order differs from the sim")
        print("     articulation order. On hardware this permutes every joint obs & action.")
        print(f"     articulation index for each assumed slot: {perm}")
    print("=====================================================\n")

    action_scale = float(deploy.action_scale) if deploy is not None else 0.25

    def cmd_of_env():
        # Feed the deploy pipeline the env's own velocity command so the
        # velocity_commands obs block matches the reference exactly.
        return env.unwrapped.command_manager.get_command(
            "base_velocity")[0].detach().cpu().numpy().astype(np.float32)

    def read_sim():
        jp = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        jv = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
        quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
        gyro = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        return jp, jv, quat, gyro

    def deploy_step(mode, last_raw_action):
        """Compute (deploy_obs, raw_action, raw_action_in_artic_order) for env.step.

        mode='asis'  : the CURRENT deploy_policy.py path -- reconstruct Unitree-order
                       sensor readings, run deploy_core with its hardcoded U2I remap
                       and per-leg default, remap motor targets back. Faithfully
                       reproduces whatever the real robot would do (bug included).
        mode='fixed' : the CORRECT path -- joints stay in articulation order (name
                       based), so obs and action line up with how the net was trained.
        Both use the exact same exported network (self.deploy).
        """
        jp_artic, jv_artic, quat, gyro = read_sim()
        cmd = cmd_of_env()
        # base_lin_vel source (hardware can't measure it directly):
        #   zero    = current deploy_policy.py (feeds 0 -> policy overshoots)
        #   command = use the velocity command as a proxy (no extra sensing)
        #   true    = the sim's true body velocity (a perfect estimator, upper bound)
        src = args_cli.linvel_source
        if src == "true":
            blv = robot.data.root_lin_vel_b[0].detach().cpu().numpy().astype(np.float32)
        elif src == "command":
            blv = np.array([cmd[0], cmd[1], 0.0], dtype=np.float32)
        else:
            blv = np.zeros(3, dtype=np.float32)
        if mode == "asis":
            jp_unitree = jp_artic[u2a]          # physical Unitree-order sensor read
            jv_unitree = jv_artic[u2a]
            obs = deploy.get_observations(jp_unitree, jv_unitree, quat, gyro,
                                          cmd, last_raw_action)
            obs[0:3] = blv                       # inject the chosen base_lin_vel
            raw = deploy.run_inference(obs)
            _, target_unitree = deploy.postprocess_action(raw)
            target_artic = np.empty(12, dtype=np.float32)
            target_artic[u2a] = target_unitree
            raw_action_artic = (target_artic - default_artic) / action_scale
        else:  # fixed
            joint_pos_rel = jp_artic - default_artic
            proj = DeployPolicyCore.compute_projected_gravity(quat)
            obs = np.concatenate([blv, gyro, proj, cmd,
                                  joint_pos_rel, jv_artic, last_raw_action])
            raw = deploy.run_inference(obs)     # net output already in artic order
            raw_action_artic = raw
        return obs, raw, raw_action_artic

    # This rsl_rl version returns observations as a group dict/TensorDict; the
    # student policy group (48-dim) is the "policy" key.
    def student_obs_np(o):
        t = o["policy"] if (isinstance(o, dict) or hasattr(o, "keys")) else o
        return t[0].detach().cpu().numpy().astype(np.float32)

    # Override the low-level PD gains to the *hardware* values. The env's DCMotor
    # actuator computes torque = stiffness*(target-q) - damping*qd (clamped to the
    # 23.7 Nm motor curve), so setting these tensors reproduces the real bring-up.
    def set_gains(kp, kd):
        applied = False
        for act in robot.actuators.values():
            try:
                act.stiffness[:] = kp
                act.damping[:] = kd
                applied = True
            except Exception:
                pass  # e.g. ActuatorNetMLP has no settable PD gains
        print(f"[INFO] set actuator gains Kp={kp} Kd={kd}"
              if applied else "[INFO] actuator has no PD gains to override (native model kept)")

    zeros_action = torch.zeros((args_cli.num_envs, 12), device=env.unwrapped.device)

    # --- STAND phase: hold the default pose at the stand gains -------------------
    set_gains(args_cli.stand_kp, args_cli.stand_kd)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    stand_h = []
    for _ in range(max(0, args_cli.stand_steps)):
        obs, _, _, _ = env.step(zeros_action)   # action 0 -> target = default pose
        if isinstance(obs, tuple):
            obs = obs[0]
        stand_h.append(float(robot.data.root_pos_w[0, 2].item()))
        if not simulation_app.is_running():
            break
    if stand_h:
        print(f"[INFO] stand phase done ({len(stand_h)} steps), "
              f"base height {stand_h[-1]:.3f}")

    # --- POLICY phase: switch to policy gains and run the deployed policy --------
    set_gains(args_cli.policy_kp, args_cli.policy_kd)

    # --- rollout ----------------------------------------------------------------
    if deploy is not None:
        deploy.reset_hidden()
        deploy_shadow.reset_hidden()
    last_raw = np.zeros(12, dtype=np.float32)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    logs = {"net_export_err": [], "obs_block_err": {b[0]: [] for b in OBS_BLOCKS},
            "base_height": [], "cmd_vx": [], "actual_vx": [], "proj_grav_z": [],
            "fell": [], "foot_force": [], "joint_pos": []}

    for step in range(args_cli.num_steps):
        env_obs_np = student_obs_np(obs)

        with torch.inference_mode():
            ref_action_t = reference_policy(obs)
        ref_action = ref_action_t[0].detach().cpu().numpy().astype(np.float32)

        # (1) net-export fidelity: identical input (env_obs) to shadow deploy net
        if deploy_shadow is not None:
            deploy_net_action = deploy_shadow.run_inference(env_obs_np)
            logs["net_export_err"].append(
                float(np.max(np.abs(deploy_net_action - ref_action))))

        # (2/3) deploy obs + action from current sim state (asis or fixed path)
        if deploy is not None:
            mode = "fixed" if args_cli.drive_mode == "reference" else args_cli.drive_mode
            deploy_obs, deploy_raw, raw_action_artic = deploy_step(mode, last_raw)
            for name, lo, hi in OBS_BLOCKS:
                logs["obs_block_err"][name].append(
                    float(np.max(np.abs(deploy_obs[lo:hi] - env_obs_np[lo:hi]))))

        # choose the controller that actually drives the sim
        if args_cli.drive_mode == "reference" or deploy is None:
            step_action = ref_action_t
            last_raw = ref_action.copy()
        else:
            step_action = torch.from_numpy(raw_action_artic).float().unsqueeze(0).to(
                env.unwrapped.device)
            last_raw = deploy_raw.copy()

        obs, _, _, _ = env.step(step_action)
        if isinstance(obs, tuple):
            obs = obs[0]

        height = float(robot.data.root_pos_w[0, 2].item())
        vx = float(robot.data.root_lin_vel_b[0, 0].item())
        pg = DeployPolicyCore.compute_projected_gravity(
            robot.data.root_quat_w[0].detach().cpu().numpy())
        if foot_idx is not None:
            nf = contact_sensor.data.net_forces_w[0]  # (num_bodies, 3)
            fmag = [float(nf[i].norm().item()) if i >= 0 else 0.0 for i in foot_idx]
            logs["foot_force"].append(fmag)
        logs["joint_pos"].append(
            robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).tolist())
        logs["base_height"].append(height)
        logs["cmd_vx"].append(float(cmd_of_env()[0]))
        logs["actual_vx"].append(vx)
        logs["proj_grav_z"].append(float(pg[2]))
        logs["fell"].append(bool(pg[2] > -0.5 or height < 0.15))

        if not simulation_app.is_running():
            break
        if args_cli.video and step + 1 >= args_cli.video_length:
            break

    # --- summarize --------------------------------------------------------------
    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {"mean": float(a.mean()), "max": float(a.max()), "min": float(a.min())} if a.size else {}

    n = len(logs["base_height"])
    fell_any = bool(np.any(logs["fell"])) if n else None
    fell_step = int(np.argmax(logs["fell"])) if (n and fell_any) else -1

    # --- gait analysis (foot contact pattern) -----------------------------------
    gait = {}
    ff = np.asarray(logs["foot_force"], dtype=np.float64)
    if ff.ndim == 2 and ff.shape[0] > 40:
        ff = ff[20:]                                   # drop startup transient
        contact = ff > 1.0                             # (T, 4) FL FR RL RR
        duty = contact.mean(axis=0)
        # pair-wise in-phase fraction (both stance or both swing)
        def inphase(a, b):
            return float(np.mean(contact[:, a] == contact[:, b]))
        diag = 0.5 * (inphase(0, 3) + inphase(1, 2))   # FL-RR, FR-RL   (trot)
        lateral = 0.5 * (inphase(0, 2) + inphase(1, 3))  # FL-RL, FR-RR (pace)
        frontrear = 0.5 * (inphase(0, 1) + inphase(2, 3))  # FL-FR, RL-RR (bound)
        gtype = max([("trot", diag), ("pace", lateral), ("bound", frontrear)],
                    key=lambda kv: kv[1])
        # step frequency from FL contact transitions
        tr = np.abs(np.diff(contact[:, 0].astype(int)))
        cyc = tr.sum() / 2.0
        freq = cyc / (len(contact) * 0.02) if len(contact) else 0.0
        gait = {"duty_FL_FR_RL_RR": [round(float(x), 3) for x in duty],
                "inphase_diagonal_trot": round(diag, 3),
                "inphase_lateral_pace": round(lateral, 3),
                "inphase_frontrear_bound": round(frontrear, 3),
                "dominant_pattern": gtype[0], "pattern_score": round(gtype[1], 3),
                "step_freq_hz": round(freq, 2)}
        print("\n================= GAIT =================")
        print(json.dumps(gait, indent=2))
        print("=======================================\n")

        # gait diagram PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            T = contact.shape[0]
            t = np.arange(T) * 0.02
            fig, ax = plt.subplots(figsize=(12, 2.6))
            for r, name in enumerate(["FL", "FR", "RL", "RR"]):
                stance = contact[:, r]
                ax.fill_between(t, r + 0.05, r + 0.95, where=stance,
                                step="pre", color="#2c7fb8")
            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(["FL", "FR", "RL", "RR"])
            ax.set_xlabel("time (s)")
            ax.set_title(f"{os.path.basename(os.path.dirname(model_dir))} gait "
                         f"({gait['dominant_pattern']}, "
                         f"duty {gait['duty_FL_FR_RL_RR']}, "
                         f"{gait['step_freq_hz']} Hz)  filled=stance")
            ax.set_ylim(0, 4)
            fig.tight_layout()
            png = os.path.join(model_dir, "gait_diagram.png")
            fig.savefig(png, dpi=110)
            print(f"[INFO] wrote {png}")
        except Exception as e:
            print(f"[WARN] gait diagram failed: {e}")

    # --- posture analysis (mean joint angle per joint vs nominal default) -------
    posture = {}
    jp = np.asarray(logs["joint_pos"], dtype=np.float64)
    if jp.ndim == 2 and jp.shape[0] > 40:
        jp = jp[20:]
        mean_ang = jp.mean(axis=0)
        default = robot.data.default_joint_pos[0].detach().cpu().numpy()
        posture = {keys[i]: {"mean_deg": round(float(np.degrees(mean_ang[i])), 1),
                             "default_deg": round(float(np.degrees(default[i])), 1),
                             "dev_deg": round(float(np.degrees(mean_ang[i] - default[i])), 1)}
                   for i in range(12)}
        print("\n================= POSTURE (mean joint angle) =================")
        for name in ["FL_thigh", "RL_thigh", "FL_calf", "RL_calf",
                     "FR_thigh", "RR_thigh", "FR_calf", "RR_calf"]:
            if name in posture:
                p = posture[name]
                print(f"  {name:9s} mean {p['mean_deg']:7.1f}°  "
                      f"default {p['default_deg']:7.1f}°  dev {p['dev_deg']:+6.1f}°")
        print("=============================================================\n")

    summary = {
        "model_dir": model_dir,
        "backend": args_cli.backend,
        "drive_mode": args_cli.drive_mode,
        "stand_gains": [args_cli.stand_kp, args_cli.stand_kd],
        "policy_gains": [args_cli.policy_kp, args_cli.policy_kd],
        "linvel_source": args_cli.linvel_source,
        "stand_steps": args_cli.stand_steps,
        "stand_end_height": stand_h[-1] if stand_h else None,
        "steps": n,
        "joint_order_match": bool(order_match),
        "sim_articulation_order": keys,
        "deploy_assumed_order": ASSUMED_ISAAC_ORDER,
        "selftest_pass": st_ok,
        "selftest_err": st_err,
        "net_export_err": stats(logs["net_export_err"]),
        "obs_block_err": {k: stats(v) for k, v in logs["obs_block_err"].items()},
        "base_height": stats(logs["base_height"]),
        "actual_vx": stats(logs["actual_vx"]),
        "cmd_vx_mean": float(np.mean(logs["cmd_vx"])) if n else None,
        "fell_any": fell_any,
        "first_fall_step": fell_step,
        "gait": gait,
        "posture": posture,
    }
    print("\n================= PARITY SUMMARY =================")
    print(json.dumps(summary, indent=2))
    print("=================================================\n")

    out = args_cli.report or os.path.join(model_dir, "sim_parity_report.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "series": logs}, f, indent=2)
    print(f"[INFO] wrote {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
