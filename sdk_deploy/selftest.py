"""오프라인 자가 검증 — 로봇/SDK 없이 실행 가능해야 합니다.

  python3 selftest.py

검증 항목:
  1. Isaac<->SDK 관절 순서 매핑 왕복
  2. 기하 자코비안 vs 수치 미분
  3. 기본 자세 FK 타당성 (좌우 대칭, 발 높이)
  4. 칼만 필터: 정지 수렴 + 합성 등속 전진 수렴
  5. 관측 조립: 차원과 슬라이스 배치
  6. mock 로봇으로 기립 + 정책 루프 전 구간 실행
"""

import numpy as np

import config as C
from kinematics import all_foot_pos_body, foot_jacobian_body, foot_pos_body
from observation import build_obs
from robot_io import MockGo1Interface
from state_estimator import LinearKFStateEstimator


def test_joint_remap():
    q_isaac = np.arange(12.0)
    q_sdk = q_isaac[C.SDK_TO_ISAAC]
    assert np.allclose(q_sdk[C.ISAAC_TO_SDK], q_isaac)
    # 개별 확인: Isaac 0 = FL_hip 은 SDK 3 (FL_0)
    assert C.ISAAC_TO_SDK[0] == 3
    # SDK 0 = FR_hip 은 Isaac 1
    assert C.SDK_TO_ISAAC[0] == 1
    print("ok: joint remap")


def test_jacobian():
    rng = np.random.RandomState(0)
    eps = 1e-6
    for leg in range(4):
        for _ in range(20):
            q = rng.uniform(-1.0, 1.0, 3) + np.array([0.0, 0.8, -1.5])
            j_ana = foot_jacobian_body(leg, q)
            j_num = np.empty((3, 3))
            for k in range(3):
                dq = np.zeros(3)
                dq[k] = eps
                j_num[:, k] = (
                    foot_pos_body(leg, q + dq) - foot_pos_body(leg, q - dq)
                ) / (2 * eps)
            assert np.allclose(j_ana, j_num, atol=1e-6), (leg, q)
    print("ok: geometric jacobian == numeric jacobian")


def test_fk_default_pose():
    feet = all_foot_pos_body(C.DEFAULT_JOINT_POS)
    # 좌우 대칭 (y 부호만 반대)
    assert np.allclose(feet[0] * [1, -1, 1], feet[1], atol=1e-9)
    assert np.allclose(feet[2] * [1, -1, 1], feet[3], atol=1e-9)
    # 발은 몸통 아래 0.2~0.35 m
    assert np.all(feet[:, 2] < -0.20) and np.all(feet[:, 2] > -0.35), feet
    print(f"ok: FK default pose, foot z = {feet[:, 2].round(3)}")


def test_kf_stationary():
    est = LinearKFStateEstimator()
    q = C.DEFAULT_JOINT_POS
    dq = np.zeros(12)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    accel = np.array([0.0, 0.0, 9.81])
    contact = np.ones(4, dtype=bool)
    for _ in range(100):
        out = est.update(quat, np.zeros(3), accel, q, dq, contact, 0.02)
    assert np.linalg.norm(out["v_body"]) < 1e-3, out["v_body"]
    assert abs(out["p_world"][2] - 0.31) < 0.05, out["p_world"]
    print(f"ok: KF stationary, |v|={np.linalg.norm(out['v_body']):.1e} "
          f"h={out['p_world'][2]:.3f}")


def test_kf_constant_velocity():
    """지지발 4개가 world 에 고정된 채 몸통이 vx 로 이동하는 합성 데이터."""
    v_body = np.array([0.5, 0.0, 0.0])
    est = LinearKFStateEstimator()
    q = C.DEFAULT_JOINT_POS
    dq = np.zeros(12)
    # J dq = -v_body 가 되도록 관절 속도 구성 (발이 world 고정 → 상대속도 -v)
    for leg in range(4):
        ids = C.LEG_JOINT_IDS[leg]
        j = foot_jacobian_body(leg, q[ids])
        dq[ids] = np.linalg.solve(j, -v_body)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    accel = np.array([0.0, 0.0, 9.81])  # 등속 → 비력은 중력뿐
    contact = np.ones(4, dtype=bool)
    for _ in range(200):
        out = est.update(quat, np.zeros(3), accel, q, dq, contact, 0.02)
    err = np.linalg.norm(out["v_body"] - v_body)
    assert err < 0.02, out["v_body"]
    print(f"ok: KF constant velocity, v_est={out['v_body'].round(3)}")


def test_obs_layout():
    robot = MockGo1Interface()
    state = robot.read_state()
    v = np.array([0.1, 0.2, 0.3])
    cmd = np.array([0.4, 0.0, -0.1])
    last_a = np.arange(12.0)
    obs = build_obs(state, v, cmd, last_a)
    assert obs.shape == (59,)
    assert np.allclose(obs[0:3], v)
    assert np.allclose(obs[3:6], 0.0)                    # gyro
    assert np.allclose(obs[6:9], [0.0, 0.0, -1.0])       # proj gravity
    assert np.allclose(obs[9:12], cmd)
    assert np.allclose(obs[12:24], 0.0)                  # q - default = 0
    assert np.allclose(obs[24:36], 0.0)                  # dq
    assert np.allclose(obs[36:48], last_a)
    assert np.allclose(obs[48:52], 0.0)                  # calf - nominal = 0
    assert np.allclose(obs[52:59], 0.0)                  # privileged tail
    print("ok: obs layout (59)")


def test_mock_deploy_loop():
    import argparse

    from deploy import Deployer

    args = argparse.Namespace(mock=True, kp=C.KP, kd=C.KD)
    robot = MockGo1Interface()
    dep = Deployer(robot, LinearKFStateEstimator(), args)
    dep._estimator_warmup(0.2)
    dep.stand_up(duration=0.2)
    dep.run_policy(lambda obs: np.zeros(12), np.zeros(3), duration=0.5)
    assert len(robot.sent) > 20
    q_last = robot.sent[-1][0]
    assert np.allclose(q_last, C.DEFAULT_JOINT_POS, atol=1e-6)
    lo, hi = C.SOFT_JOINT_LIMITS[:, 0], C.SOFT_JOINT_LIMITS[:, 1]
    for q_des, _, _ in robot.sent:
        assert np.all(q_des >= lo - 1e-9) and np.all(q_des <= hi + 1e-9)
    print(f"ok: mock deploy loop, {len(robot.sent)} commands sent")


if __name__ == "__main__":
    test_joint_remap()
    test_jacobian()
    test_fk_default_pose()
    test_kf_stationary()
    test_kf_constant_velocity()
    test_obs_layout()
    test_mock_deploy_loop()
    print("\n모든 자가 검증 통과")
