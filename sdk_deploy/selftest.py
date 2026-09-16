"""오프라인 자가 검증 — 로봇/SDK 없이 실행 가능해야 합니다.

    python3 selftest.py [--policy model/P3-final/exported/policy_numpy.npz]

검증 항목:
  1. Isaac <-> SDK 관절 순서 매핑 왕복
  2. 관측 조립: 차원과 슬라이스 배치 (49차원 레이아웃)
  3. peg_leg_one_hot 인코딩
  4. 속도 명령 클립
  5. mock 로봇으로 기립 + 정책 루프 전 구간 실행 (정상 / 부목 모드)
  6. --policy 를 주면: 번들 레이아웃이 config.py 와 일치하고 reference 자가검증 통과
"""

import argparse
import os

# numpy 첫 import 전에 설정 (deploy.py 와 동일한 이유 — NX 에서 OpenBLAS
# 멀티스레드가 루프를 50 ms 이상 밀리게 하는 것을 실측).
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import config as C
from observation import build_obs, clip_command, peg_leg_one_hot
from robot_io import MockGo1Interface


def test_joint_remap():
    q_isaac = np.arange(12.0)
    q_sdk = q_isaac[C.SDK_TO_ISAAC]
    assert np.allclose(q_sdk[C.ISAAC_TO_SDK], q_isaac)
    # 개별 확인: Isaac 0 = FL_hip 은 SDK 3 (FL_0)
    assert C.ISAAC_TO_SDK[0] == 3
    # SDK 0 = FR_hip 은 Isaac 1
    assert C.SDK_TO_ISAAC[0] == 1
    print("ok: joint remap")


def test_obs_layout():
    state = MockGo1Interface().read_state()
    cmd = np.array([0.4, 0.0, -0.1])
    last_a = np.arange(12.0)
    obs = build_obs(state, cmd, last_a, peg_leg_one_hot(1))   # FR 부상

    assert obs.shape == (49,), obs.shape
    assert np.allclose(obs[0:3], 0.0)                    # gyro
    assert np.allclose(obs[3:6], [0.0, 0.0, -1.0])       # projected gravity
    assert np.allclose(obs[6:9], cmd)
    assert np.allclose(obs[9:21], 0.0)                   # q - default = 0
    assert np.allclose(obs[21:33], 0.0)                  # dq
    assert np.allclose(obs[33:45], last_a)
    assert np.allclose(obs[45:49], [0, 1, 0, 0])         # FR one-hot
    print("ok: obs layout (49)")


def test_one_hot():
    assert np.allclose(peg_leg_one_hot(None), 0.0)
    for i, name in enumerate(C.LEG_NAMES):
        oh = peg_leg_one_hot(i)
        assert oh.sum() == 1.0 and oh[i] == 1.0, (name, oh)
    print("ok: peg_leg_one_hot")


def test_command_clip():
    assert np.allclose(clip_command(5.0, 0.0, 0.0)[0], C.CMD_VX_RANGE[1])
    assert np.allclose(clip_command(-5.0, 0.0, 0.0)[0], C.CMD_VX_RANGE[0])
    assert np.allclose(clip_command(0.0, 9.0, -9.0)[1:],
                       [C.CMD_VY_RANGE[1], C.CMD_WZ_RANGE[0]])
    print("ok: command clip")


def _mock_loop(injured_leg=None):
    from deploy import Deployer

    args = argparse.Namespace(mock=True, kp=C.KP, kd=C.KD, vx_floor=0.0,
                              stand_kp=C.STAND_KP, stand_kd=C.STAND_KD,
                              injured_leg=injured_leg, log_npz=None)
    robot = MockGo1Interface()
    dep = Deployer(robot, args)

    class _ZeroPolicy:
        hidden = np.zeros(256, dtype=np.float32)

        def __call__(self, obs):
            assert obs.shape == (C.OBS_DIM,), obs.shape
            return np.zeros(C.NUM_ACTIONS)

        def reset(self):
            pass

        def estimate(self):
            return None

    dep.stand_up(duration=0.2)
    dep.run_policy(_ZeroPolicy(), np.zeros(3), duration=0.5)

    assert len(robot.sent) > 20
    lo, hi = C.SOFT_JOINT_LIMITS[:, 0], C.SOFT_JOINT_LIMITS[:, 1]
    for q_des, _, _ in robot.sent:
        assert np.all(q_des >= lo - 1e-9) and np.all(q_des <= hi + 1e-9)
    return robot


def test_mock_deploy_loop():
    robot = _mock_loop()
    # zero action -> 기본 자세
    assert np.allclose(robot.sent[-1][0], C.DEFAULT_JOINT_POS, atol=1e-6)
    print(f"ok: mock deploy loop, {len(robot.sent)} commands sent")


def test_mock_deploy_loop_injured():
    leg = 1  # FR
    robot = _mock_loop(injured_leg=leg)
    q_last = robot.sent[-1][0]
    # 부상 calf 만 부목 고정각, 나머지는 기본 자세
    assert np.isclose(q_last[8 + leg], C.SPLINT_CALF_ANGLE), q_last[8 + leg]
    others = [i for i in range(12) if i != 8 + leg]
    assert np.allclose(q_last[others], C.DEFAULT_JOINT_POS[others], atol=1e-6)
    print(f"ok: mock deploy loop (splinted {C.LEG_NAMES[leg]}), "
          f"calf held at {q_last[8 + leg]:.2f} rad")


def test_policy_bundle(path):
    from policy import Policy

    p = Policy(path)          # reference_io.json 자가검증이 여기서 돈다
    state = MockGo1Interface().read_state()
    obs = build_obs(state, np.zeros(3), np.zeros(12), peg_leg_one_hot(None))
    p.reset()
    a = p(obs)
    assert a.shape == (C.NUM_ACTIONS,), a.shape
    est = p.estimate()
    extra = "" if est is None else f", L_hat={est[0]:.3f} v_hat={est[1].round(2)}"
    print(f"ok: policy bundle, |a|max={np.abs(a).max():.3f}{extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=None,
                    help="추가로 검증할 번들 (.npz / .onnx / .pt)")
    cli = ap.parse_args()

    test_joint_remap()
    test_obs_layout()
    test_one_hot()
    test_command_clip()
    test_mock_deploy_loop()
    test_mock_deploy_loop_injured()
    if cli.policy:
        test_policy_bundle(cli.policy)

    print("\n모든 자가 검증 통과")
