"""실물 Go1 배포 상수 — 전부 phase1 체크포인트의 params/env.yaml 과 1:1 대응.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 관절 순서
# ---------------------------------------------------------------------------
# Isaac Lab (학습) 순서: 타입별 그룹 [..hips.., ..thighs.., ..calves..]
ISAAC_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
# unitree_legged_sdk 모터 순서: 다리별 [FR, FL, RR, RL] x [hip, thigh, calf]
SDK_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# q_isaac[i] = q_sdk[ISAAC_TO_SDK[i]],  cmd_sdk[j] = cmd_isaac[SDK_TO_ISAAC[j]]
ISAAC_TO_SDK = np.array([SDK_JOINT_NAMES.index(n) for n in ISAAC_JOINT_NAMES])
SDK_TO_ISAAC = np.array([ISAAC_JOINT_NAMES.index(n) for n in SDK_JOINT_NAMES])

# 다리 순서는 이 저장소 전체와 동일하게 FL, FR, RL, RR 로 통일합니다.
LEG_NAMES = ["FL", "FR", "RL", "RR"]
# 다리 i 의 (hip, thigh, calf) 관절이 Isaac 벡터에서 차지하는 인덱스
LEG_JOINT_IDS = np.array([[i, 4 + i, 8 + i] for i in range(4)])
# SDK footForce 배열은 [FR, FL, RR, RL] → FL,FR,RL,RR 로 재배열
FOOT_FORCE_SDK_TO_LEG = np.array([1, 0, 3, 2])

# ---------------------------------------------------------------------------
# 기본 자세 / 액션 (env.yaml: init_state.joint_pos, actions.joint_pos)
# ---------------------------------------------------------------------------
# 주의: 뒷다리 thigh 는 1.0 으로 앞다리(0.8)와 다릅니다.
DEFAULT_JOINT_POS = np.array([
    0.1, -0.1, 0.1, -0.1,       # hips   (L +0.1 / R -0.1)
    0.8, 0.8, 1.0, 1.0,         # thighs (front 0.8 / rear 1.0)
    -1.5, -1.5, -1.5, -1.5,     # calves
])
# calf_pos_abs 관측의 nominal (mdp/observations.py: calf_pos_nominal_rel).
# 건강한 로봇 배포에서는 default 와 같습니다 (부상 시에만 lock 각으로 바뀜).
NOMINAL_CALF_POS = np.array([-1.5, -1.5, -1.5, -1.5])  # FL, FR, RL, RR
CALF_IDS_ISAAC = np.array([8, 9, 10, 11])              # FL, FR, RL, RR

ACTION_SCALE = 0.25          # actions.joint_pos.scale
# target_q = DEFAULT_JOINT_POS + ACTION_SCALE * action (use_default_offset)

# ---------------------------------------------------------------------------
# 제어 (env.yaml: sim.dt=0.005, decimation=4, actuators: DCMotor Kp/Kd)
# ---------------------------------------------------------------------------
CONTROL_DT = 0.02            # 정책 50 Hz
KP = 20.0
KD = 0.5
STAND_KP = 60.0
STAND_KD = 1.0
GAIN_BLEND_TIME = 3.0        # 정책 인계 후 STAND_KP → KP 블렌딩 시간 (s)

# 정상 종료 시 천천히 주저앉기(lie_down) — damping 직행은 Kp=0 이라 뚝 떨어짐.
# 비상정지/기울임 가드는 여전히 즉시 damping (제어된 하강이 낙하와 싸우면 안 됨).
LIE_DOWN_POS = np.array([
    0.1, -0.1, 0.1, -0.1,     # hips (기립 기본값 유지)
    1.3, 1.3, 1.3, 1.3,       # thighs 접기
    -2.6, -2.6, -2.6, -2.6,   # calves 접기 (soft limit -2.72 안쪽)
])
LIE_DOWN_TIME = 2.5
LIE_DOWN_KP = 25.0
LIE_DOWN_KD = 1.0
DAMPING_KD = 2.0             # 비상/종료 시 damping 모드
TORQUE_LIMIT = 23.7          # DCMotor effort_limit (모든 관절 동일하게 학습됨)

# ---------------------------------------------------------------------------
# 관절 한계 (Go1 URDF) — soft factor 0.9 를 적용해 목표각을 클램프
# ---------------------------------------------------------------------------
_HIP_LIM = (-0.863, 0.863)
_THIGH_LIM = (-0.686, 4.501)
_CALF_LIM = (-2.818, -0.888)
JOINT_LIMITS = np.array([_HIP_LIM] * 4 + [_THIGH_LIM] * 4 + [_CALF_LIM] * 4)
SOFT_LIMIT_FACTOR = 0.9      # env.yaml: soft_joint_pos_limit_factor
_mid = JOINT_LIMITS.mean(axis=1)
_half = (JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]) / 2.0
SOFT_JOINT_LIMITS = np.stack(
    [_mid - SOFT_LIMIT_FACTOR * _half, _mid + SOFT_LIMIT_FACTOR * _half],
    axis=1,
)

# ---------------------------------------------------------------------------
# 속도 명령 한계 (env.yaml commands.ranges — 학습 분포 밖 금지)
# ---------------------------------------------------------------------------
CMD_VX_RANGE = (0.0, 1.0)    # 학습은 0.1~1.0; 0 은 램프업 통과점으로만 사용
CMD_VY_RANGE = (0.0, 0.0)
CMD_WZ_RANGE = (-0.15, 0.15)

# ---------------------------------------------------------------------------
# 다리 기하 (Go1 URDF)
# ---------------------------------------------------------------------------
HIP_OFFSETS = np.array([     # trunk → hip roll 축 원점 (body frame), FL FR RL RR
    [0.1881, 0.04675, 0.0],
    [0.1881, -0.04675, 0.0],
    [-0.1881, 0.04675, 0.0],
    [-0.1881, -0.04675, 0.0],
])
LEG_SIDE_SIGN = np.array([1.0, -1.0, 1.0, -1.0])  # 왼쪽 +y
L_HIP = 0.08                 # hip → thigh 횡방향 오프셋
L_THIGH = 0.213
L_CALF = 0.213
FOOT_RADIUS = 0.02

# ---------------------------------------------------------------------------
# 상태 추정기 (legged_control / MIT Mini-Cheetah linear KF 계열 기본값)
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD = 20.0   # footForce (N 근사) 접촉 판정 — bias 차감 후 값 기준
# Go1 발 압력 센서는 무부하에서도 발마다 큰 오프셋을 출력합니다.
# 실측 (2026-08-01, 로봇을 하네스에 매달아 완전 무부하, dry-run ff, FL FR RL RR):
#   [125, 114, 114, 116]  ← 이 값이면 임계 20 으로는 항상 contact=4
# 재측정 방법: 로봇을 매달고 `deploy.py --mode dry-run` 의 ff 값을 그대로 적기.
FOOT_FORCE_BIAS = np.array([125.0, 114.0, 114.0, 116.0])  # FL, FR, RL, RR
EST_NOISE_P_IMU = 0.02           # process: 위치
EST_NOISE_V_IMU = 0.02           # process: 속도
EST_NOISE_P_FOOT = 0.002         # process: 발 위치
EST_SENSOR_P_FOOT = 0.005        # measurement: 다리 운동학 발 위치
EST_SENSOR_V_FOOT = 0.1          # measurement: 다리 운동학 발 속도
EST_SENSOR_H_FOOT = 0.01         # measurement: 발 높이(평지 가정)
EST_SWING_INFLATION = 1e4        # 스윙 발 노이즈 팽창 계수

GRAVITY = np.array([0.0, 0.0, -9.81])

# ---------------------------------------------------------------------------
# 관측 — teacher actor 는 두 그룹을 연결해 받습니다 (agent.yaml obs_groups:
# policy = [policy, privileged_obs], 이 순서).
#
# policy 그룹 (52, env.yaml observations.policy 순서):
#   base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
#   + velocity_commands(3) + joint_pos_rel(12) + joint_vel_rel(12)
#   + last_action(12) + calf_pos_abs(4)
# privileged_obs 그룹 (7): peg_leg_one_hot(5: FL,FR,RL,RR,injured_flag)
#   + splint_length(1) + foot_friction(1)
#
# phase1 은 부상 0%(prob_peg_leg=0)로 학습됐으므로 학습 내내 privileged
# 꼬리는 전부 0 이었습니다. 정상 로봇 배포에서도 0 을 넣는 것이 정확히
# 학습 분포와 일치합니다.
# ---------------------------------------------------------------------------
POLICY_OBS_DIM = 52
PRIVILEGED_OBS_DIM = 7
OBS_DIM = POLICY_OBS_DIM + PRIVILEGED_OBS_DIM   # actor 입력 59
HEALTHY_PRIVILEGED_TAIL = np.zeros(PRIVILEGED_OBS_DIM)
NUM_ACTIONS = 12

# ---------------------------------------------------------------------------
# 통신 (unitree_legged_sdk low-level)
# ---------------------------------------------------------------------------
SDK_LOWLEVEL = 0xFF
SDK_LOCAL_PORT = 8080
SDK_ROBOT_IP = "192.168.123.10"
SDK_ROBOT_PORT = 8007
POWER_PROTECT_LEVEL = 5      # 1(보수적)~10; 처음엔 낮게

# 안전 가드
TILT_LIMIT_RAD = 0.7         # roll/pitch 초과 시 즉시 damping
LOOP_OVERRUN_LIMIT = 0.5     # 한 주기가 dt 의 (1+이 값)배를 넘기면 경고
