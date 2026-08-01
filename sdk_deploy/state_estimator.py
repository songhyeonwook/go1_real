"""IMU + 다리 오도메트리 융합 선형 칼만 필터 (base_lin_vel 추정).

정책 관측의 base_lin_vel 은 IMU 만으로는 얻을 수 없습니다(가속도 적분은
바이어스/중력 제거 오차로 수 초 내 발산). 지지 중인 발은 지면에 고정되어
있다는 가정으로 관절 엔코더 + 순운동학에서 드리프트 없는 속도를 얻고,
IMU 가속도 적분과 칼만 필터로 융합합니다.

구조는 공개 구현들을 따릅니다 (18-상태 선형 KF):
  * MIT Mini-Cheetah software: PositionVelocityEstimator (BSD)
  * legged_control: LinearKalmanFilter (BSD)

상태 x(18) = [p(3), v(3), p_foot_FL(3), FR(3), RL(3), RR(3)]  (world)
예측: p += v dt,  v += (R a_imu + g) dt,  발 위치는 상수
측정(28): 발마다
  p - p_f = -R p_rel        (다리 운동학 상대 위치, 12)
  v = -w x (R p_rel) - R v_rel   (지지발 속도 0 가정, 12)
  p_f.z = FOOT_RADIUS       (평지 가정, 4)
스윙 발은 해당 노이즈를 크게 부풀려 사실상 측정에서 제외합니다.
"""

import numpy as np

import config as C
from kinematics import all_foot_pos_body, all_foot_vel_body


def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) 쿼터니언 → body→world 회전행렬."""
    w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class LinearKFStateEstimator:
    def __init__(self):
        self.xhat = np.zeros(18)
        self.P = np.eye(18) * 3.0
        self._initialized = False

    def _init_from_kinematics(self, rot: np.ndarray, p_rel: np.ndarray,
                              contact: np.ndarray) -> None:
        """첫 호출: 지지발이 지면(z=FOOT_RADIUS)에 있다고 보고 높이 초기화."""
        feet_w = (rot @ p_rel.T).T  # (4,3), 몸통 원점 기준 world 방향
        use = contact if contact.any() else np.ones(4, dtype=bool)
        base_z = C.FOOT_RADIUS - feet_w[use, 2].mean()
        self.xhat[:3] = [0.0, 0.0, base_z]
        self.xhat[3:6] = 0.0
        self.xhat[6:] = (self.xhat[:3] + feet_w).reshape(-1)
        self._initialized = True

    def update(self, quat_wxyz, gyro, accel, q_isaac, dq_isaac,
               contact, dt) -> dict:
        """한 제어 주기 갱신.

        Args:
            quat_wxyz: IMU 자세 (w,x,y,z)
            gyro: body frame 각속도 (rad/s)
            accel: 가속도계 비력 (m/s^2, 정지 시 [0,0,+9.81])
            q_isaac/dq_isaac: 관절 상태 12차원 (Isaac 순서)
            contact: (4,) bool — FL,FR,RL,RR 접촉 여부
            dt: 경과 시간 (s)
        Returns:
            dict(v_body, v_world, p_world, contact_count)
        """
        rot = quat_to_rot(np.asarray(quat_wxyz, dtype=float))
        gyro = np.asarray(gyro, dtype=float)
        contact = np.asarray(contact, dtype=bool)

        p_rel = all_foot_pos_body(q_isaac)      # (4,3)
        v_rel = all_foot_vel_body(q_isaac, dq_isaac)

        if not self._initialized:
            self._init_from_kinematics(rot, p_rel, contact)

        # ---- 예측 -----------------------------------------------------
        a_w = rot @ np.asarray(accel, dtype=float) + C.GRAVITY
        A = np.eye(18)
        A[0:3, 3:6] = np.eye(3) * dt
        B = np.zeros((18, 3))
        B[0:3] = np.eye(3) * 0.5 * dt * dt
        B[3:6] = np.eye(3) * dt

        Q = np.zeros((18, 18))
        Q[0:3, 0:3] = np.eye(3) * dt * C.EST_NOISE_P_IMU
        Q[3:6, 3:6] = np.eye(3) * dt * C.EST_NOISE_V_IMU
        for i in range(4):
            s = 6 + 3 * i
            infl = 1.0 if contact[i] else C.EST_SWING_INFLATION
            Q[s:s + 3, s:s + 3] = np.eye(3) * dt * C.EST_NOISE_P_FOOT * infl

        self.xhat = A @ self.xhat + B @ a_w
        self.P = A @ self.P @ A.T + Q

        # ---- 측정 -----------------------------------------------------
        H = np.zeros((28, 18))
        y = np.zeros(28)
        Rn = np.zeros(28)
        w_world = rot @ gyro
        for i in range(4):
            infl = 1.0 if contact[i] else C.EST_SWING_INFLATION
            pw = rot @ p_rel[i]
            # p - p_f = -R p_rel
            r = 3 * i
            H[r:r + 3, 0:3] = np.eye(3)
            H[r:r + 3, 6 + 3 * i:9 + 3 * i] = -np.eye(3)
            y[r:r + 3] = -pw
            Rn[r:r + 3] = C.EST_SENSOR_P_FOOT * infl
            # v = -w x (R p_rel) - R v_rel
            r = 12 + 3 * i
            H[r:r + 3, 3:6] = np.eye(3)
            y[r:r + 3] = -np.cross(w_world, pw) - rot @ v_rel[i]
            Rn[r:r + 3] = C.EST_SENSOR_V_FOOT * infl
            # p_f.z = FOOT_RADIUS
            r = 24 + i
            H[r, 8 + 3 * i] = 1.0
            y[r] = C.FOOT_RADIUS
            Rn[r] = C.EST_SENSOR_H_FOOT * infl

        S = H @ self.P @ H.T + np.diag(Rn)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(28))
        self.xhat = self.xhat + K @ (y - H @ self.xhat)
        self.P = (np.eye(18) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)  # 대칭 유지

        # 전부 스윙(공중)이면 적분 드리프트 방지를 위해 속도를 서서히 감쇠
        if not contact.any():
            self.xhat[3:6] *= 0.97

        v_world = self.xhat[3:6].copy()
        return {
            "v_body": rot.T @ v_world,
            "v_world": v_world,
            "p_world": self.xhat[:3].copy(),
            "contact_count": int(contact.sum()),
        }
