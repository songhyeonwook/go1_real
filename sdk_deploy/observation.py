"""49차원 student 관측 조립 — 학습 env 와 순서/정의 일치.

학습 측 기준은 go1_lod `mdp/obs_normalizer.py` 의 span 상수이고, 내보낸 번들의
`policy_io.json` 이 같은 레이아웃을 싣고 있습니다. 셋 중 하나라도 어긋나면
policy.py 의 reference 자가검증이 걸립니다.

concat 순서 (스케일/클립/노이즈 없음 — obs 스케일은 정책 그래프 안에 있습니다):
  base_ang_vel(3)       : IMU 자이로
  projected_gravity(3)  : R^T [0,0,-1]
  velocity_commands(3)  : (vx, vy, wz)
  joint_pos_rel(12)     : q - DEFAULT_JOINT_POS
  joint_vel_rel(12)     : dq (default vel = 0)
  last_action(12)       : 직전 정책 출력 (스케일 전 raw)
  peg_leg_one_hot(4)    : 부상 다리 (FL, FR, RL, RR). 정상 = 전부 0

base_lin_vel 은 더 이상 관측에 없습니다. 학습 env 가 policy 그룹에서 빼고
teacher 전용 privileged 로 옮겼기 때문입니다 (실기 Go1 는 몸통 선속도를 측정할
수 없음). 추정이 필요하면 정책의 vel_head 출력을 쓰세요 — policy.Policy.aux 참고.
"""

import numpy as np

import config as C


def quat_to_rot(q_wxyz):
    """(w,x,y,z) 쿼터니언 → body→world 회전행렬."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def peg_leg_one_hot(injured_leg=None):
    """부상 다리 one-hot. `injured_leg` 는 0=FL, 1=FR, 2=RL, 3=RR 또는 None(정상).

    학습에서 정상 env 는 전부 0 이고 (mdp/observations.py: peg_leg_one_hot),
    부상 env 만 해당 자리가 1 입니다.
    """
    one_hot = np.zeros(C.NUM_LEGS, dtype=np.float32)
    if injured_leg is not None:
        one_hot[int(injured_leg)] = 1.0
    return one_hot


def build_obs(state, cmd, last_action, one_hot):
    rot = quat_to_rot(state.quat_wxyz)
    proj_g = rot.T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate([
        state.gyro,
        proj_g,
        cmd,
        state.q - C.DEFAULT_JOINT_POS,
        state.dq,
        last_action,
        one_hot,
    ]).astype(np.float32)
    assert obs.shape == (C.OBS_DIM,), obs.shape
    return obs


def clip_command(vx, vy, wz):
    """학습 분포(antalgic.yaml commands) 밖 명령을 잘라냅니다."""
    return np.array([
        np.clip(vx, *C.CMD_VX_RANGE),
        np.clip(vy, *C.CMD_VY_RANGE),
        np.clip(wz, *C.CMD_WZ_RANGE),
    ])
