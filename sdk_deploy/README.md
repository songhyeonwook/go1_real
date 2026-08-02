# sdk_deploy — phase1 정책 실물 Unitree Go1 배포 (SDK 직결, ROS 불필요)

시뮬레이션(Isaac Lab)에서 학습한 phase1 보행 정책
(`go1_compar/models/phase1_mlp_s42/model_5999.pt`)을 실물 Go1에 올리는
스택입니다. 이 저장소 루트의 ROS 기반 배포(`scripts/deploy_policy.py`)와는
별개의 독립 스택으로, unitree_legged_sdk 에 직접 붙습니다.
전부 오픈소스 기반입니다:

| 구성 요소 | 기반 오픈소스 | 라이선스 |
|---|---|---|
| 로봇 통신 ([robot_io.py](robot_io.py)) | [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk) python wrapper | BSD-3 |
| 속도 추정 ([state_estimator.py](state_estimator.py)) | MIT Mini-Cheetah software / [legged_control](https://github.com/qiayuanl/legged_control) 의 18-상태 선형 KF 구조 | BSD |
| 정책 추론 ([policy.py](policy.py)) | onnxruntime | MIT |

## 구조

```
deploy.py          메인 루프 (dry-run / hang / stand / walk + 안전장치)
config.py          학습 설정(env.yaml)에서 옮겨 적은 모든 상수
observation.py     59차원 teacher 관측 조립 (학습과 동일한 순서/정의)
state_estimator.py IMU + 다리 오도메트리 칼만 필터 → base_lin_vel
kinematics.py      Go1 다리 FK + 기하 자코비안
robot_io.py        SDK 인터페이스 (Isaac↔SDK 관절 순서 변환은 여기서만)
policy.py          ONNX / TorchScript 정책 로더
selftest.py        로봇 없이 실행하는 자가 검증
```

핵심 설계: **관측에 필요한 base_lin_vel 은 IMU 만으로는 얻을 수 없으므로**
"지지발은 지면에 고정"이라는 가정으로 관절 엔코더+운동학에서 속도를 얻고,
IMU 가속도 적분과 칼만 필터로 융합합니다.

## 1. 준비 — 정책 내보내기 (시뮬 PC에서)

phase1 teacher 의 actor 입력은 59차원입니다: policy 그룹 52
(`GO1_ABS_JOINT_OBS=1`) + privileged 그룹 7 (부상 one_hot 5, 부목 길이 1,
발 마찰 1 — phase1 은 부상 0% 학습이라 항상 0). 내보낼 때도 같은 env
설정이 필요합니다:

```bash
cd /home/shw/go1_compar/scripts/rsl_rl
GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 \
GO1_ABS_JOINT_OBS=1 GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
python3 export_policy_onnx.py \
    --checkpoint /home/shw/go1_compar/models/phase1_mlp_s42/model_5999.pt \
    --task Template-Go1-Lab-v0 \
    --agent rsl_rl_teacher_mlp_cfg_entry_point \
    --phase teacher --headless
# → go1_compar/models/phase1_mlp_s42/exported/policy.onnx (+ policy.pt)
# 산출물을 이 스택의 모델 폴더로 복사:
cp /home/shw/go1_compar/models/phase1_mlp_s42/exported/policy.* \
   /home/shw/go1_compar/models/phase1_mlp_s42/exported/policy_io.json \
   /home/shw/go1_real/sdk_deploy/model/phase1/
```

## 2. 준비 — 배포 PC 설정

배포 PC(노트북 권장, 로봇 내장 Pi 도 가능)를 로봇 이더넷에 연결하고
고정 IP `192.168.123.162/24` 로 설정합니다.

```bash
# unitree_legged_sdk 파이썬 바인딩 빌드 (v3.8.x 가 Go1 지원)
git clone https://github.com/unitreerobotics/unitree_legged_sdk
cd unitree_legged_sdk && mkdir build && cd build
cmake -DPYTHON_BUILD=TRUE .. && make
# 생성된 robot_interface*.so 경로를 PYTHONPATH 에 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)/../lib/python/amd64

pip install -r requirements.txt
python3 selftest.py   # 로봇 없이 통과해야 함
```

### 2-b. Go1 온보드 NX 에서 직접 실행하는 경우 (오프라인 빌드, 2026-08-01 적용됨)

NX(192.168.123.15)는 Python 3.6.9 에 인터넷이 없습니다. SDK 에 동봉된
프리빌트 `robot_interface.cpython-38-*.so` 는 3.8 전용이라 import 되지 않으므로,
온보드 SDK 소스로 3.6용을 빌드합니다 (pybind11 동봉, 이미 NX 에 적용 완료):

```bash
cd ~/go1_ws/src/unitree_ros_to_real/unitree_legged_sdk/python_wrapper
# python_interface.cpp 의 `#include <msgpack.hpp>` 는 실제로 사용되지 않는데
# NX 에 msgpack 헤더가 없어 빌드를 막습니다 → 주석 처리 (NX 에 적용됨)
mkdir -p build && cd build && cmake .. && make -j4
# → ../../lib/python/arm64/robot_interface.cpython-36m-aarch64-linux-gnu.so
```

`robot_io.py` 가 이 경로를 자동으로 sys.path 에 추가하므로 PYTHONPATH 설정 없이
`import robot_interface` 가 됩니다 (다른 위치라면 `UNITREE_SDK_PYTHON_PATH` 로 지정).

⚠️ NX 에서 SDK low-level UDP 를 쓰려면 ROS 브리지(`ros_udp lowlevel` 노드)가
떠 있으면 안 됩니다. ros_udp 는 `/low_cmd` 퍼블리셔가 없어도 500 Hz 로
zero-torque 명령을 계속 송신하므로, sdk_deploy 와 동시에 켜면 MCU 가 두
명령 스트림을 번갈아 받아 모터가 덜덜 떨립니다 (실측). 한 번에 하나만.

NX 에는 onnxruntime/torch 가 없지만, `model/phase1/policy_numpy.npz`
(순수 NumPy 백엔드, `scripts/export_policy_numpy.py` 로 변환)를 쓰면
`--mode hang/walk` 도 온보드에서 실행할 수 있습니다. 모델 옆
`reference_io.json` (개발 PC onnxruntime 출력 6쌍)과 자동 대조하는
self-test 가 로드 시 실행되어, 불일치 시 모터를 건드리기 전에 중단합니다.

## 3. 로봇을 low-level 모드로 전환

1. 로봇을 **하네스에 매달거나** 들 수 있는 상태로 전원 인가, 기립 대기.
2. 조종기에서 `L2+A` (엎드림) → `L2+B` (모터 이완) →
   `L1+L2+Start` (low-level 모드). 이후 조종기 보행은 불가능해집니다.

## 4. 시험 절차 — 반드시 순서대로

**모든 단계에서 Enter = 비상정지(damping).** roll/pitch 0.7 rad 초과 시
자동으로 damping 됩니다.

```bash
# (0) SDK 없이 코드 경로만 검증
python3 deploy.py --mode dry-run --mock

# (1) 로봇 매단 채 센서 방향/순서 검증.
#roslaunch lowlevel 안키고 해야됌.
python3 deploy.py --mode dry-run

# (2) 매단 채 기립 자세 추종 (정책 미실행)
#roslaunch lowlevel 안키고 해야됌.
#contact 이 0인지 확인
python3 deploy.py --mode stand --duration 10

# (3) 매단 채 정책 실행 (명령 0) — 발산 없이 안정적인 다리 움직임 확인
python3 deploy.py --mode hang --policy model/phase1/policy_numpy.npz

# (4) 지면에서 제자리 (명령 0 램프만) → 짧은 전진
#메달아두지만, stand 상태에서 발이 땅에 닿게.
#contac이 3~4로 바뀌는 거 확인.
python3 deploy.py --mode walk --policy model/phase1/policy_numpy.npz \
    --vx 0.3 --duration 10
```

phase3 LSTM student (obs 52, `model/phase3_antalgic_s42/`)는 **vx 0.3~1.0 로만
학습**됐으므로 `--vx-floor 0.3` 으로 램프가 분포 밖(vx<0.3)을 통과하지 않게 합니다.
같은 이유로 cmd=0 인 hang 은 이 모델에겐 분포 밖이라 움직임이 이상해도 모델
결함이 아닙니다 — 지면 walk 로 판단하세요:
```bash
python3 deploy.py --mode walk --policy model/phase3_antalgic_s42/policy_numpy.npz \
    --vx 0.3 --vx-floor 0.3 --lin-vel kf --duration 10
```
student 는 조립된 59차원 관측의 앞 52차원(policy 그룹)만 소비하고, LSTM
hidden/cell 은 정책 인계 시점에 자동으로 0 리셋됩니다 (policy.py / deploy.py).

### 실시간 부상 추정 표시 (injury probe)

student 는 부상 정보를 입력받지 않고 LSTM hidden(256)에 스스로 추론합니다.
그 hidden 을 해독하는 선형 probe(`injury_probe.npz`: W/b/names)를 모델 폴더에
두면, walk/hang 중 1초마다 추정치가 표시됩니다:

```
v=(...) contact=3 cmd=(0.30,+0.00) |a|max=2.9
    [EST] peg_FL=-0.02 peg_FR=+0.01 peg_RL=+0.03 peg_RR=+0.94 splint_len=+0.21 friction=+0.55
```

probe 는 정답 라벨이 있는 **시뮬 롤아웃으로만 학습 가능**합니다
(`scripts/train_injury_probe.py`, ridge 닫힌형 — 입력 형식은 그 파일 docstring
참고). `--log-npz` 는 실기 hidden 궤적도 `h` 키로 함께 저장하므로, 실기
부상(부목) 실험 데이터에 probe 를 적용해 오프라인 검증도 할 수 있습니다.

base_lin_vel 관측은 기본적으로 **속도 명령 proxy** 입니다 (`--lin-vel cmd`) —
이 저장소의 모든 시뮬 검증(sim_test/deploy_core.py, ROS 스택)과 같은 방식입니다.
온보드 KF 추정치는 실보행에서 과소추정이 확인돼(cmd 0.3 에서 0.02~0.15)
관측으로는 기본 사용하지 않고 텔레메트리에만 씁니다. `--lin-vel kf` 로 전환 가능.

속도 명령은 학습 분포로 클램프됩니다: vx 0–1.0 m/s, vy 0, wz ±0.15 rad/s
(config.py `CMD_*_RANGE`, 학습 env.yaml 의 command ranges 와 동일).

## 학습 설정과의 대응 (건드리면 안 되는 값)

| 항목 | 값 | 출처 |
|---|---|---|
| 정책 주기 | 50 Hz | sim dt 0.005 × decimation 4 |
| PD 게인 | Kp 20 / Kd 0.5 | actuators DCMotor |
| 액션 | q_des = default + 0.25·a | actions.joint_pos |
| 기본자세 | hip ±0.1, thigh 0.8(앞)/1.0(뒤), calf −1.5 | init_state |
| 관절 순서 | 타입별 그룹 (hips→thighs→calves) | Isaac Lab |
| 관측 59차원 | lin_vel, ang_vel, gravity, cmd, q_rel, dq, last_a, calf_abs + privileged 7(=0) | obs_groups: policy+privileged |

## 튜닝 포인트

* `CONTACT_FORCE_THRESHOLD` (기본 20 N): dry-run 에서 footForce 를 보고
  조정. 서 있을 때 다리당 ~30 N 이상, 스윙 시 ~0 이어야 합니다.
* 추정기 노이즈 (`EST_*`): 기본값은 legged_control 계열 관행값.
  v_body 가 정지 상태에서 0.05 m/s 이상 떠돌면 `EST_SENSOR_V_FOOT` 을
  줄이거나 접촉 임계값을 올리세요.
* `--power-protect` (1~10): 처음엔 5 이하 권장. 토크가 부족해 보이면
  단계적으로 올립니다.

## 알려진 한계

* 발 높이 측정은 **평지 가정** (`p_f.z = FOOT_RADIUS`). 실내 평지에서만.
* phase1 은 vy=0, wz±0.15 로 학습됨 — 제자리 회전/횡보행 불가.
* 게임패드 미지원 (CLI 고정 명령 + 램프). 필요 시 `run_policy` 의
  `cmd_target` 을 조이스틱 입력으로 바꾸면 됩니다.
* IMU 가속도계 바이어스 추정 없음(구조상 KF 가 흡수). 드리프트가 심하면
  로봇을 몇 초 정지 상태로 두고 시작하세요.
