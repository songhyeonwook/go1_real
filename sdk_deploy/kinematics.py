"""Go1 다리 순운동학 + 기하 자코비안 (body frame).

체인 (URDF 기준):
  trunk --[HIP_OFFSET]--> hip roll(q0, x축)
    --[0, side*L_HIP, 0]--> thigh pitch(q1, y축)
    --[0,0,-L_THIGH]--> calf pitch(q2, y축) --[0,0,-L_CALF]--> foot

상태 추정기의 다리 오도메트리가 쓰는 유일한 기하 모듈입니다. 자코비안은
축/외적 방식의 정확한 기하 자코비안이며 selftest.py 가 수치미분과 대조합니다.
"""

import numpy as np

from config import (
    HIP_OFFSETS, L_CALF, L_HIP, L_THIGH, LEG_JOINT_IDS, LEG_SIDE_SIGN,
)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def foot_pos_body(leg: int, q: np.ndarray) -> np.ndarray:
    """다리 leg(0=FL,1=FR,2=RL,3=RR)의 발 위치 (body frame, 발 중심)."""
    q0, q1, q2 = q
    side = LEG_SIDE_SIGN[leg]
    r0 = _rot_x(q0)
    r1 = r0 @ _rot_y(q1)
    r2 = r1 @ _rot_y(q2)
    p = HIP_OFFSETS[leg].copy()
    p = p + r0 @ np.array([0.0, side * L_HIP, 0.0])
    p = p + r1 @ np.array([0.0, 0.0, -L_THIGH])
    p = p + r2 @ np.array([0.0, 0.0, -L_CALF])
    return p


def foot_jacobian_body(leg: int, q: np.ndarray) -> np.ndarray:
    """기하 자코비안 J (3x3): v_foot_body = J @ dq."""
    q0, q1, q2 = q
    side = LEG_SIDE_SIGN[leg]
    r0 = _rot_x(q0)
    r1 = r0 @ _rot_y(q1)

    o0 = HIP_OFFSETS[leg].copy()
    o1 = o0 + r0 @ np.array([0.0, side * L_HIP, 0.0])
    o2 = o1 + r1 @ np.array([0.0, 0.0, -L_THIGH])
    p_foot = foot_pos_body(leg, q)

    a0 = np.array([1.0, 0.0, 0.0])       # hip roll: body x축
    a1 = r0 @ np.array([0.0, 1.0, 0.0])  # thigh pitch: 회전 y축
    a2 = a1                               # calf pitch: 같은 y축 (R_y 는 y축 불변)

    j = np.empty((3, 3))
    j[:, 0] = np.cross(a0, p_foot - o0)
    j[:, 1] = np.cross(a1, p_foot - o1)
    j[:, 2] = np.cross(a2, p_foot - o2)
    return j


def all_foot_pos_body(q_isaac: np.ndarray) -> np.ndarray:
    """(4,3) 발 위치. q_isaac 은 Isaac 순서 12차원."""
    return np.stack(
        [foot_pos_body(i, q_isaac[LEG_JOINT_IDS[i]]) for i in range(4)]
    )


def all_foot_vel_body(q_isaac: np.ndarray, dq_isaac: np.ndarray) -> np.ndarray:
    """(4,3) 발 속도 (body frame, 몸통 고정 기준): J @ dq."""
    out = np.empty((4, 3))
    for i in range(4):
        ids = LEG_JOINT_IDS[i]
        out[i] = foot_jacobian_body(i, q_isaac[ids]) @ dq_isaac[ids]
    return out
