"""unitree_legged_sdk low-level 인터페이스 + 오프라인 mock.

실제 로봇: unitree_legged_sdk 의 파이썬 바인딩(robot_interface)을 사용합니다.
빌드 방법은 README 참고. 이 모듈 바깥의 코드는 전부 Isaac 관절 순서만 다루고,
SDK 순서 변환은 여기서만 일어납니다.

안전:
  * 목표각을 soft joint limit 으로 클램프
  * SDK Safety.PowerProtect 통과 후에만 송신
  * send_damping() 은 Kp=0 / Kd 만 남기는 무릎꿇기(제동) 모드
"""

import dataclasses

import numpy as np

import config as C

# unitree_legged_sdk 상수 (sdk/include/unitree_legged_sdk/comm.h)
_POS_STOP_F = 2.146e9
_VEL_STOP_F = 16000.0


@dataclasses.dataclass
class RobotState:
    """Isaac 순서로 정리된 로봇 상태 스냅샷."""
    q: np.ndarray            # (12,) 관절각 (Isaac 순서)
    dq: np.ndarray           # (12,) 관절 각속도
    quat_wxyz: np.ndarray    # (4,) IMU 자세
    gyro: np.ndarray         # (3,) body 각속도
    accel: np.ndarray        # (3,) 가속도계 비력
    foot_force: np.ndarray   # (4,) FL,FR,RL,RR
    rpy: np.ndarray          # (3,) roll, pitch, yaw


class Go1Interface:
    """실물 Go1 low-level UDP 인터페이스."""

    def __init__(self, power_protect: int = C.POWER_PROTECT_LEVEL):
        import robot_interface as sdk  # unitree_legged_sdk python wrapper

        self._sdk = sdk
        self._power_protect = int(power_protect)
        self.udp = sdk.UDP(
            C.SDK_LOWLEVEL, C.SDK_LOCAL_PORT,
            C.SDK_ROBOT_IP, C.SDK_ROBOT_PORT,
        )
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)
        self.cmd.levelFlag = C.SDK_LOWLEVEL

    def read_state(self) -> RobotState:
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        ms = self.state.motorState
        q_sdk = np.array([ms[j].q for j in range(12)])
        dq_sdk = np.array([ms[j].dq for j in range(12)])
        imu = self.state.imu
        ff_sdk = np.array(
            [self.state.footForce[i] for i in range(4)], dtype=float
        )
        return RobotState(
            q=q_sdk[C.ISAAC_TO_SDK],
            dq=dq_sdk[C.ISAAC_TO_SDK],
            quat_wxyz=np.array(list(imu.quaternion)),
            gyro=np.array(list(imu.gyroscope)),
            accel=np.array(list(imu.accelerometer)),
            foot_force=ff_sdk[C.FOOT_FORCE_SDK_TO_LEG],
            rpy=np.array(list(imu.rpy)),
        )

    def send_positions(self, q_des_isaac: np.ndarray,
                       kp: float, kd: float) -> None:
        """목표 관절각 송신 (온보드 1 kHz PD 가 추종)."""
        q_des = np.clip(
            q_des_isaac, C.SOFT_JOINT_LIMITS[:, 0], C.SOFT_JOINT_LIMITS[:, 1]
        )
        q_sdk = q_des[C.SDK_TO_ISAAC]
        for j in range(12):
            mc = self.cmd.motorCmd[j]
            mc.q = float(q_sdk[j])
            mc.dq = 0.0
            mc.Kp = float(kp)
            mc.Kd = float(kd)
            mc.tau = 0.0
        self.safe.PowerProtect(self.cmd, self.state, self._power_protect)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def send_damping(self, kd: float = C.DAMPING_KD) -> None:
        """비상 정지: 위치 제어를 끊고 관절 제동만 남깁니다."""
        for j in range(12):
            mc = self.cmd.motorCmd[j]
            mc.q = _POS_STOP_F
            mc.dq = _VEL_STOP_F
            mc.Kp = 0.0
            mc.Kd = float(kd)
            mc.tau = 0.0
        self.udp.SetSend(self.cmd)
        self.udp.Send()


class MockGo1Interface:
    """SDK 없이 코드 경로를 검증하기 위한 mock.

    기본자세로 서 있는 로봇을 흉내 냅니다. 보낸 목표각을 1차 지연으로
    따라가므로 deploy.py 의 전체 루프를 오프라인에서 돌려볼 수 있습니다.
    """

    def __init__(self, power_protect: int = 0):
        self._q = C.DEFAULT_JOINT_POS.copy()
        self._dq = np.zeros(12)
        self.sent = []  # (q_des, kp, kd) 기록 — selftest 용

    def read_state(self) -> RobotState:
        return RobotState(
            q=self._q.copy(),
            dq=self._dq.copy(),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            gyro=np.zeros(3),
            accel=np.array([0.0, 0.0, 9.81]),
            foot_force=np.full(4, 50.0),
            rpy=np.zeros(3),
        )

    def send_positions(self, q_des_isaac, kp, kd):
        q_des = np.clip(
            q_des_isaac, C.SOFT_JOINT_LIMITS[:, 0], C.SOFT_JOINT_LIMITS[:, 1]
        )
        self.sent.append((q_des.copy(), kp, kd))
        alpha = 0.3
        self._dq = (q_des - self._q) * alpha / C.CONTROL_DT
        self._q = self._q + (q_des - self._q) * alpha

    def send_damping(self, kd: float = C.DAMPING_KD):
        self._dq = np.zeros(12)
