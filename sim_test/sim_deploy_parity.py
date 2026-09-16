#!/usr/bin/env python3
"""배포 파리티 테스트 — 실기에 올리기 전에 Isaac Sim 안에서 배포 경로를 그대로 돌립니다.

내보낸 산출물 그 자체와 `sdk_deploy/` 의 관측 조립·관절 리맵·액션 후처리를 학습
환경 안에서 실행해, 학습된 정책을 재현하고 실제로 걷는지 확인합니다.

내보내기 자체의 충실도(참조 정책 vs .pt/.onnx/.npz)는 이제 시뮬 없이
`scripts/export_p3_student.py` 가 reference_io.json 으로 검증합니다. 여기서는
시뮬이 있어야만 확인할 수 있는 세 가지만 봅니다:

  1. 관절 순서 감사   : 시뮬 articulation 관절 순서 vs config.ISAAC_JOINT_NAMES.
                        어긋나면 실기에서 모든 관측과 액션이 조용히 뒤섞입니다.
  2. 관측 조립 충실도 : 하드웨어 센서 신호를 흉내 내 sdk_deploy/observation.py 로
                        관측을 다시 만들고, env 자신의 관측과 블록 단위로 비교.
  3. 폐루프 거동      : 배포 파이프라인으로 시뮬을 구동해 넘어지지 않고 속도
                        명령을 추종하는지 기록.

실행 (go1_lod 의 스크립트 폴더에서 경로를 잡습니다):
  ~/IsaacLab/isaaclab.sh -p sim_test/sim_deploy_parity.py \
      --phase_config_path configs/phase/3/phase3.yaml \
      --checkpoint sdk_deploy/model/P3-final/model_3999.pt \
      --policy sdk_deploy/model/P3-final/exported/policy_numpy.npz \
      --num_steps 600 --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
GO1_LOD_RSL = Path(os.environ.get("GO1_LOD_RSL", Path.home() / "go1_lod/scripts/rsl_rl"))
if not GO1_LOD_RSL.is_dir():
    sys.exit(f"go1_lod 스크립트 폴더를 찾을 수 없습니다: {GO1_LOD_RSL}\n"
             f"GO1_LOD_RSL 로 경로를 지정하세요.")
sys.path.insert(0, str(GO1_LOD_RSL))
sys.path.insert(0, str(REPO / "sdk_deploy"))
# go1_lab 확장이 `pip install -e` 로 등록돼 있지 않아도 돌도록 소스 경로를 받쳐 둡니다.
_GO1_LOD_SRC = GO1_LOD_RSL.parents[1] / "source" / "go1_lab"
if _GO1_LOD_SRC.is_dir():
    sys.path.append(str(_GO1_LOD_SRC))

from utils.eval_common import (  # noqa: E402
    add_common_args, build_agent_cfg, build_env_cfg, load_config, make_gym_env,
    make_policy, resolve_checkpoint, resolve_eval_mode, wrap_rsl,
)

parser = argparse.ArgumentParser(description="Go1 배포 파리티 테스트 (Isaac Sim)")
add_common_args(parser)
parser.add_argument("--policy", type=str, required=True,
                    help="내보낸 번들 (.npz / .onnx / .pt)")
parser.add_argument("--num_steps", type=int, default=600, help="제어 스텝 수 (50 Hz)")
parser.add_argument("--drive", choices=["deploy", "reference"], default="deploy",
                    help="deploy = 배포 파이프라인으로 구동(기본), "
                         "reference = 학습 정책으로 구동(기준선)")
parser.add_argument("--obs_tol", type=float, default=1e-3,
                    help="관측 블록별 최대 허용 오차")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

config = load_config(args)
checkpoint = resolve_checkpoint(args)
eval_mode = resolve_eval_mode(args, config, default="normal")
device = args.device
seed = args.seed if args.seed is not None else config.train.seed

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import go1_lab.tasks  # noqa: F401,E402

import config as C  # noqa: E402  (sdk_deploy/config.py)
from observation import build_obs, peg_leg_one_hot  # noqa: E402
from policy import Policy  # noqa: E402

# 관측 블록 — sdk_deploy/observation.py 의 concat 순서와 같아야 합니다.
OBS_BLOCKS = [
    ("base_ang_vel", 0, 3),
    ("projected_gravity", 3, 6),
    ("velocity_commands", 6, 9),
    ("joint_pos_rel", 9, 21),
    ("joint_vel_rel", 21, 33),
    ("last_actions", 33, 45),
    ("peg_leg_one_hot", 45, 49),
]


def joint_order_audit(robot):
    """시뮬 articulation 관절 순서가 config.py 가 가정한 순서와 같은지."""
    sim_names = list(robot.joint_names)
    # 부목 관절은 정책이 다루지 않으므로 제외
    leg_names = [n for n in sim_names if "_splint_" not in n]
    match = leg_names == C.ISAAC_JOINT_NAMES
    print("\n================ 관절 순서 감사 ================")
    print(f"  sim articulation : {leg_names}")
    print(f"  config.py 가정   : {C.ISAAC_JOINT_NAMES}")
    print(f"  MATCH            : {match}")
    if not match:
        print("  !! 불일치: 실기에서 모든 관절 관측과 액션이 뒤섞입니다.")
    print("================================================\n")
    # 정책 관절의 articulation 인덱스 (부목 관절을 건너뛰기 위함)
    idx = np.array([sim_names.index(n) for n in C.ISAAC_JOINT_NAMES])
    return match, idx


class SimRobotState:
    """sdk_deploy 의 RobotState 와 같은 필드를, 시뮬에서 읽어 채웁니다."""
    __slots__ = ("q", "dq", "quat_wxyz", "gyro", "accel", "foot_force", "rpy")


def read_sim(robot, joint_idx):
    s = SimRobotState()
    f32 = lambda t: t[0].detach().cpu().numpy().astype(np.float32)
    s.q = f32(robot.data.joint_pos)[joint_idx]
    s.dq = f32(robot.data.joint_vel)[joint_idx]
    s.quat_wxyz = f32(robot.data.root_quat_w)
    s.gyro = f32(robot.data.root_ang_vel_b)
    s.accel = np.zeros(3, dtype=np.float32)
    s.foot_force = np.zeros(4, dtype=np.float32)
    s.rpy = np.zeros(3, dtype=np.float32)
    return s


def main():
    env_cfg = build_env_cfg(config, num_envs=args.num_envs or 1, device=device,
                            seed=seed, eval_mode=eval_mode)
    agent_cfg = build_agent_cfg(config, seed=seed, device=device)
    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint, device)

    deploy = Policy(str(REPO / args.policy) if not os.path.isabs(args.policy)
                    else args.policy)

    base = env.unwrapped
    robot = base.scene["robot"]
    order_ok, joint_idx = joint_order_audit(robot)

    # wrap_rsl 생성 시 이미 reset 이 돌아 조건 배정이 끝나 있습니다.
    obs = env.get_observations()
    deploy.reset()

    peg = getattr(base, "_peg_leg_index", None)
    injured = int(peg[0].item()) if peg is not None else -1
    one_hot = peg_leg_one_hot(injured if injured >= 0 else None)
    print(f"[INFO] 조건: {'정상' if injured < 0 else C.LEG_NAMES[injured] + ' 부상'}"
          f"  one_hot={one_hot.astype(int).tolist()}")

    last_action = np.zeros(C.NUM_ACTIONS, dtype=np.float32)
    worst = {name: 0.0 for name, _, _ in OBS_BLOCKS}
    act_err, heights, vx, cmd_vx = [], [], [], []

    for step in range(args.num_steps):
        env_obs = obs["policy"][0].detach().cpu().numpy().astype(np.float32)

        # (1) 관측 조립 충실도 — 배포 경로로 다시 만든 관측 vs env 관측
        # get_command 는 live 참조라 step 안의 재샘플에 덮입니다 — 복사해 둡니다.
        cmd = base.command_manager.get_command("base_velocity")[0, :3]
        cmd = cmd.clone().detach().cpu().numpy().astype(np.float32)
        deploy_obs = build_obs(read_sim(robot, joint_idx), cmd, last_action, one_hot)
        for name, lo, hi in OBS_BLOCKS:
            worst[name] = max(worst[name],
                              float(np.abs(deploy_obs[lo:hi] - env_obs[lo:hi]).max()))

        # (2) 내보내기 충실도 — 같은 env 관측을 두 경로에 먹였을 때의 액션 차이
        with torch.no_grad():
            ref_action = policy_fn(obs)[0].detach().cpu().numpy().astype(np.float32)
        deploy_action = deploy(env_obs).astype(np.float32)
        act_err.append(float(np.abs(ref_action - deploy_action).max()))

        # (3) 폐루프 — 선택한 쪽으로 실제 구동
        drive = ref_action if args.drive == "reference" else deploy_action
        last_action = drive.copy()
        obs, _, dones, _ = env.step(
            torch.from_numpy(drive).to(device).unsqueeze(0))

        heights.append(float(robot.data.root_pos_w[0, 2].item()))
        vx.append(float(robot.data.root_lin_vel_b[0, 0].item()))
        cmd_vx.append(float(cmd[0]))
        policy_module.reset(dones)
        if bool(dones[0]):
            fell = bool(base.reset_terminated[0].item())
            print(f"[{'FAIL' if fell else 'INFO'}] step {step}: "
                  f"{'낙상' if fell else '타임아웃'}으로 에피소드 종료")
            deploy.reset()
            last_action[:] = 0.0
            # 리셋으로 부상 조건이 다시 뽑힙니다 — one_hot 을 새 조건에 맞춥니다.
            peg = getattr(base, "_peg_leg_index", None)
            injured = int(peg[0].item()) if peg is not None else -1
            one_hot = peg_leg_one_hot(injured if injured >= 0 else None)

    print("\n================ 관측 조립 충실도 ================")
    for name, _, _ in OBS_BLOCKS:
        flag = "ok" if worst[name] <= args.obs_tol else "FAIL"
        print(f"  {name:20s} max|diff| = {worst[name]:.3e}  [{flag}]")
    print("\n================ 내보내기 충실도 ================")
    print(f"  action max|ref - deploy| = {max(act_err):.3e}")
    print("\n================ 폐루프 거동 ================")
    print(f"  base height : mean {np.mean(heights):.3f} m  min {np.min(heights):.3f} m")
    print(f"  vx          : mean {np.mean(vx):+.3f} m/s  "
          f"(cmd mean {np.mean(cmd_vx):+.3f})")

    obs_ok = max(worst.values()) <= args.obs_tol
    exp_ok = max(act_err) <= 1e-3
    up_ok = np.min(heights) > 0.15
    print(f"\n판정: 관절순서 {'PASS' if order_ok else 'FAIL'} | "
          f"관측조립 {'PASS' if obs_ok else 'FAIL'} | "
          f"내보내기 {'PASS' if exp_ok else 'FAIL'} | "
          f"기립유지 {'PASS' if up_ok else 'FAIL'}")

    env.close()
    return 0 if (order_ok and obs_ok and exp_ok and up_ok) else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
