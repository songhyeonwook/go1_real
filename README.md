# Go1 Real-World Policy Deployment Package

이 패키지는 Isaac Lab에서 학습된 Go1 Quadruped의 강화학습 정책(Policy)을 실제 Go1 로봇(ROS Melodic 환경)에서 원활하고 안전하게 실행하기 위해 작성되었습니다. 
특히, 로봇의 특정 다리가 다치거나 의족 상태인 경우(Peg-Leg)에 적응하여 동작하는 복합 환경 보행 시나리오를 지원합니다.

---

## 📂 폴더 및 파일 구조

```text
go1_real/
├── model/
│   ├── policy.pt        # 학습이 완료된 PyTorch JIT 모델 파일
│   └── policy.onnx      # (선택사항) ONNX 모델 파일
├── scripts/
│   ├── deploy_policy.py # 실시간 ROS 제어 루프 파이썬 스크립트
│   └── deploy.launch    # ROS 파라미터 및 노드 실행용 Launch 파일
└── README.md            # 사용 가이드 및 필수 지침
```

---

## 🚀 사용 방법 (Deployment Workflow)

### 1. 환경 준비 및 설치
Go1의 온보드 PC(또는 라즈베리 파이) 환경에서 다음 의존 패키지가 필요합니다:
* **ROS Melodic**
* `unitree_legged_msgs` (Unitree ROS SDK)
* Python 3 & Numpy
* **Inference Runtime:**
  * **PyTorch** 가 이미 설치된 경우: 별도 작업 필요 없음.
  * **ONNX Runtime** (권장 - 라즈베리 파이 등 저사양 환경 효율화):
    ```bash
    pip3 install onnxruntime
    ```

### 2. 모델 이동
학습 완료된 학생(Student) 정책의 내보내기(Export) 결과물을 `model` 폴더에 저장합니다.
* `policy.pt` 또는 `policy.onnx` (경로: `/home/shw/go1_real/model/`)

### 3. 실행 방법 (ROS Launch)
실행 스크립트는 Unitree ROS가 가동 중인 환경(기본 low-level SDK 구동) 위에서 동작해야 합니다.

**A. 기본 정상(Healthy) 보행 테스트:**
```bash
rosrun go1_real deploy_policy.py _model_path:=/home/shw/go1_real/model/policy.pt
```

**B. 특정 다리가 다친(Peg-Leg) 시나리오 테스트 (예: 우측 전방 FR 다리):**
```bash
rosrun go1_real deploy_policy.py _injured_leg_idx:=1 _model_path:=/home/shw/go1_real/model/policy.pt
```
*(인덱스 정보: 0=FL, 1=FR, 2=RL, 3=RR)*

---

## 🛡️ 안전 사양 (Safety Features)

실제 로봇 하드웨어를 보호하기 위해 스크립트 내에 다음과 같은 강력한 안전 로직이 탑재되어 있습니다.

1. **부드러운 기립(Safe Stand Up Phase):**
   * 노드가 시작되면 즉시 격렬하게 정책이 반응하지 않고, 로봇의 **현재 관절 각도**에서부터 **기본 설계 서 있는 각도**까지 **4초 동안 서서히 선형 보간(Interpolation)**하여 일어섭니다.
   * 서는 과정에서 관절 게인($K_p$)을 부드럽게 램프업(Ramp-up)하여 스냅 현상을 방지합니다.

2. **기울임 자동 감지 셧다운(Orientation Safety Stop):**
   * 실행 도중 로봇이 뒤집히거나 한쪽으로 심하게 기울어지는 경우(약 60도 이상), 내부 중력 벡터 방향 변화를 즉각 감지하여 **모든 제어 명령을 차단**합니다.
   * 차단 즉시 모든 관절 강성을 0으로 하고 중간 수준의 감쇠력만 유지하는 **안전 댐핑(Dampening Mode)**으로 전환되어 로봇이 스스로 사뿐히 주저앉으며 모터 과부하를 방지합니다.

3. **관절 가동 범위 제한(Joint Limits Clipping):**
   * 정책에서 출력되는 임의의 폭주 행동을 방지하기 위해, 실제 Unitree Go1 하드웨어 가동 범위를 기반으로 계산된 Soft Joint Range 밖으로 벗어나는 명령을 사전 차단(Clamp)합니다.

---

## ⚙️ 기술 사양 정보

### 🔄 관절 순서(Remapping) 및 중력 투영
* **순서 변환:** Unitree 하드웨어 모터 배치(`FR, FL, RR, RL`)와 Isaac Lab 시뮬레이터 내 배치(`FL, FR, RL, RR`) 차이를 코드 내부에서 실시간으로 자동 정합시켜 변환합니다.
* **중력 투영:** 내부 IMU 쿼터니언을 기저 프레임으로 변환하여 계산한 정밀 물리 수식 기반의 Projected Gravity 계산식(Numpy 최적화)을 사용합니다.

### 📐 고정 파라미터 (Config Match)
* **루프 동작 주기:** `50Hz` (0.02초 dt)
* **행동 스케일(Action Scale):** `0.25`
* **제어 게인:** $K_p = 25.0$, $K_d = 0.5$ (학습은 ActuatorNetMLP 기반이라 정확한 PD 등가값은 없으며, 이 값은 관례적 Go1 sim-to-real 게인입니다. 하드웨어에서 미세조정 권장)
* **입력 차원:** 51차원 (Phase-1 Healthy 정책 기준)
  * 기본 상태 48차원: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos_rel(12) + joint_vel(12) + last_action(12)
  * Peg-Leg Privileged 3차원: `peg_leg_index, peg_leg_splint_length, peg_leg_foot_friction` → Healthy 기본값 `[0, 0, 1]` 고정
  * ⚠️ 지형 스캔(height_scan)은 이 export에 포함되지 않습니다. 모델 옆 `deployment_config.json`이 실제 입력 레이아웃의 기준입니다.
* **정책 구조:** Feed-forward ActorCritic MLP (hidden `[512, 256, 128]`, `elu`) — 순환(LSTM) 아님
* **base_lin_vel 주의:** 실제 Go1는 몸체 선속도를 직접 측정할 수 없어 0으로 입력합니다 (sim-to-real 근사).
* **Healthy 전용:** 이 정책은 정상 보행만 학습되어 다친 다리에 맞춰 보행을 적응시키지 않습니다. `injured_leg_idx`를 지정하면 해당 종아리 모터만 물리적으로 풀어(스플린트 고정용) 줄 뿐이며, 실제 부상 적응은 Phase-2/Student 정책이 필요합니다.
