"""59차원 teacher 관측 조립 — 학습 env 와 순서/정의 일치.

concat 순서 (스케일/클립 없음, 노이즈는 학습 전용이므로 미적용):
  base_lin_vel(3)       : 상태 추정기 (body frame)
  base_ang_vel(3)       : IMU 자이로
  projected_gravity(3)  : R^T [0,0,-1]
  velocity_commands(3)  : (vx, vy, wz)
  joint_pos_rel(12)     : q - DEFAULT_JOINT_POS
  joint_vel_rel(12)     : dq (default vel = 0)
  last_action(12)       : 직전 정책 출력 (스케일 전 raw)
  calf_pos_abs(4)       : q_calf - NOMINAL_CALF (FL,FR,RL,RR)
  privileged tail(7)    : one_hot(5)+splint(1)+friction(1) — 정상 로봇 = 0
"""

import numpy as np

import config as C
from robot_io import RobotState
from state_estimator import quat_to_rot


def build_obs(state: RobotState, v_body: np.ndarray, cmd: np.ndarray,
              last_action: np.ndarray) -> np.ndarray:
    rot = quat_to_rot(state.quat_wxyz)
    proj_g = rot.T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate([
        v_body,
        state.gyro,
        proj_g,
        cmd,
        state.q - C.DEFAULT_JOINT_POS,
        state.dq,
        last_action,
        state.q[C.CALF_IDS_ISAAC] - C.NOMINAL_CALF_POS,
        C.HEALTHY_PRIVILEGED_TAIL,
    ]).astype(np.float32)
    assert obs.shape == (C.OBS_DIM,), obs.shape
    return obs


def clip_command(vx: float, vy: float, wz: float) -> np.ndarray:
    """학습 분포(env.yaml commands.ranges) 밖 명령을 잘라냅니다."""
    return np.array([
        np.clip(vx, *C.CMD_VX_RANGE),
        np.clip(vy, *C.CMD_VY_RANGE),
        np.clip(wz, *C.CMD_WZ_RANGE),
    ])
