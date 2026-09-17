# go1_real — Phase-3 student 실물 Unitree Go1 배포

`go1_lod` (Isaac Lab 5.1 + RSL-RL)에서 학습한 Phase-3 student 정책을 실물 Go1에
올리는 스택입니다. 부목(splint)을 채운 다리에 적응하는 antalgic 보행을 지원합니다.

배포 경로는 **하나**입니다: `sdk_deploy/` 가 unitree_legged_sdk 에 직접 붙습니다
(ROS 불필요). 예전 ROS 노드(`scripts/deploy_policy.py` + `launch/`)는 같은 일을
두 번 하던 중복이라 제거했습니다.

```
scripts/
  export_p3_student.py     체크포인트 -> 배포 번들 (torch 만 필요, Isaac Sim 불필요)
  export_policy_numpy.py   구버전 ONNX -> NumPy 번들 (레거시 모델용)
  sync_to_robot.sh         이 저장소를 온보드 NX 로 rsync
sdk_deploy/
  deploy.py                메인 루프 (dry-run / hang / stand / walk + 안전장치)
  config.py                학습 설정에서 옮겨 적은 모든 상수
  observation.py           49차원 관측 조립
  policy.py                번들 로더 (.npz / .onnx / .pt) + 보조 헤드
  robot_io.py              SDK 인터페이스 (Isaac <-> SDK 관절 순서 변환은 여기서만)
  selftest.py              로봇 없이 도는 자가 검증
  model/P3-final/exported/ 배포 번들
sim_test/
  sim_deploy_parity.py     실기 전 Isaac Sim 파리티 테스트
  run_sim_test.sh          실행 래퍼
```

---

## 관측 규격 (49차원)

학습 측 기준은 `go1_lod` 의 `mdp/obs_normalizer.py` 이고, 내보낸
`policy_io.json` 이 같은 레이아웃을 싣습니다. 셋이 어긋나면
`reference_io.json` 자가검증이 모터를 건드리기 전에 걸립니다.

| 구간 | 항목 | 실기 소스 |
|---|---|---|
| `[0:3]` | `base_ang_vel` | IMU 자이로 |
| `[3:6]` | `projected_gravity` | IMU 쿼터니언 → `R^T [0,0,-1]` |
| `[6:9]` | `velocity_commands` | `(vx, vy, wz)` |
| `[9:21]` | `joint_pos_rel` | `q - default` |
| `[21:33]` | `joint_vel_rel` | `dq` |
| `[33:45]` | `last_actions` | 직전 정책 출력 (raw) |
| `[45:49]` | `peg_leg_one_hot` | 부상 다리 `[FL, FR, RL, RR]`, 정상 = 전부 0 |

주의할 점 세 가지:

* **`base_lin_vel` 이 없습니다.** 실기 Go1 는 몸통 선속도를 측정할 수 없어 학습
  env 가 policy 그룹에서 빼고 teacher 전용으로 옮겼습니다. 예전에 쓰던 칼만 필터
  상태추정기는 그래서 제거했습니다. 추정치가 필요하면 정책의 `vel_head` 출력을
  쓰세요 (`policy.estimate()`).
* **부상 다리를 정책이 봅니다.** `peg_leg_one_hot` 이 policy 그룹에 있으므로 이
  student 는 proprioception 전용이 아닙니다. 배포할 때 `--injured-leg` 로
  물리적으로 부목을 채운 다리와 반드시 일치시켜야 합니다.
* **관측 스케일이 정책 안에 있습니다.** 학습의 `train.normalize` (joint_vel × 0.23)
  는 `.pt`/`.onnx` 는 그래프에, `.npz` 는 `lstm_weight_ih` 에 접혀 들어갑니다.
  배포 코드는 **raw 관측**을 그대로 먹입니다.

정책 구조는 `LSTM(hidden 256, 1층) → MLP [512, 256, 128] elu`. hidden state 가
제어 스텝 간에 이어지고, 기립 완료 후 정책 인계 시점에 0으로 리셋됩니다.
출력은 raw action 입니다 (`mse_norm` 은 학습 loss 스케일일 뿐 역정규화 불필요).

---

## 1. 내보내기 (개발 PC)

```bash
/home/shw/miniconda3/envs/isaac/bin/python scripts/export_p3_student.py \
    --checkpoint sdk_deploy/model/P3-final/model_3999.pt
```

체크포인트 state_dict 에서 student 추론 경로를 직접 재조립하므로 **torch 만**
있으면 됩니다 — Isaac Sim / Isaac Lab / GPU 모두 불필요합니다. 세 백엔드가
torch 레퍼런스와 1e-5 안에서 일치하지 않으면 내보내기를 거부합니다.

산출물 (`sdk_deploy/model/P3-final/exported/`):

| 파일 | 용도 |
|---|---|
| `policy_numpy.npz` | **온보드 NX 에서 실제로 도는 것** (torch/onnxruntime 없음) |
| `policy.onnx`, `policy.pt` | 개발 PC 검증용 |
| `aux_heads.npz` | `vel_head`(몸통 선속도) / `splint_head`(부목 길이) |
| `policy_io.json` | 관측 레이아웃 + 번들 메타데이터 |
| `reference_io.json` | hidden=0 에서 시작하는 20스텝 레퍼런스 (자가검증 기준) |

## 2. 검증 (로봇 없이)

```bash
cd sdk_deploy
python3 selftest.py --policy model/P3-final/exported/policy_numpy.npz
```

관절 순서 왕복, 49차원 관측 배치, one-hot 인코딩, 명령 클립, mock 로봇 기립+정책
루프(정상/부목 양쪽), 번들 레퍼런스 자가검증을 확인합니다.

## 3. 검증 (Isaac Sim 파리티)

```bash
sim_test/run_sim_test.sh                    # 정상 조건
PEG_LEG=rl sim_test/run_sim_test.sh         # RL 다리 부상 조건
```

시뮬이 있어야만 볼 수 있는 세 가지를 확인합니다: 관절 순서 감사, 관측 조립
충실도(env 관측 vs 배포 경로 재조립), 폐루프 거동. 내보내기 충실도는 이미
`reference_io.json` 이 시뮬 없이 확인합니다.

## 4. 실기

### 4.1 네트워크 / 접속

Go1 내부망은 `192.168.123.0/24` 입니다. 개발 PC 의 유선 IP 를 `192.168.123.99` 로
고정한 뒤 접속합니다.

| 장비 | 주소 | 용도 |
|---|---|---|
| 개발 PC | `192.168.123.99` | 내보내기 / rsync |
| Raspberry Pi | `pi@192.168.123.161` | 상위 제어 보드 (배포에는 사용 안 함) |
| NX | `unitree@192.168.123.15` | **배포 실행 대상** |

```bash
ssh pi@192.168.123.161        # Raspberry Pi
ssh unitree@192.168.123.15    # NX (여기서 deploy.py 실행)
```

`scripts/sync_to_robot.sh` 는 기본으로 NX(`unitree@192.168.123.15`) 로 보냅니다.

### 4.2 NX 사전 준비

```bash
./scripts/sync_to_robot.sh --go            # 개발 PC 에서

ssh unitree@192.168.123.15                 # 이후는 NX 에서
cd ~/go1_ws/src/go1_real/sdk_deploy
```

ROS 는 필요 없습니다 (`deploy.py` 는 unitree_legged_sdk UDP 에 직접 붙습니다).
`roscore` 나 `source /opt/ros/...` 없이 실행합니다.

**리모컨으로 하위제어(low-level) 모드 진입** — 이걸 하지 않으면 상위 보행
컨트롤러가 관절을 잡고 있어 `deploy.py` 의 관절 명령이 무시됩니다.

1. **L2+A** — 앉기 (서 있으면 두 번)
2. **L2+B** — damping, 완전히 바닥에 엎드림
3. **L1+L2+START** — 하위제어 모드 진입. 관절 힘이 빠져 손으로 자유롭게
   움직여지면 들어간 것입니다.
4. 로봇을 매단 뒤 4.3 의 dry-run 부터 시작합니다.

하위 모드에서는 리모컨으로 상위 모드로 돌아갈 수 없습니다 — 복귀하려면 로봇을
재부팅합니다.

* `roslaunch unitree_legged_real real.launch ctrl_level:=lowlevel` 은 **실행하지 않습니다.**
  `deploy.py` 가 unitree_legged_sdk 에 직접 붙기 때문에, lowlevel 노드가 같이 돌면
  모터 명령이 충돌합니다.

### 4.3 실행 순서

**반드시 이 순서로** 진행합니다. 각 단계가 통과해야 다음으로 넘어갑니다.

| 단계 | 로봇 상태 | 명령 | 확인할 것 |
|---|---|---|---|
| (1) dry-run | 매단 채 | `python3 deploy.py --mode dry-run` | 다리를 손으로 움직여 센서 부호 / 관절 순서 |
| (2) stand | 매단 채 | `python3 deploy.py --mode stand --duration 10` | 기립 자세 추종, 정책 미실행, 발 contact 가 0 |
| (3) hang | 매단 채 | `python3 deploy.py --mode hang --policy model/P3-final/exported/policy_numpy.npz` | 명령 0 으로 정책 실행, 발산 없이 트로트 비슷한 다리 움직임 |
| (4) walk | 지면 | `python3 deploy.py --mode walk --policy model/P3-final/exported/policy_numpy.npz --vx 0.3 --duration 10` | 제자리(명령 0 램프) → 짧은 전진 |

```bash
# (1) 로봇 매단 채 센서 방향/순서 검증
python3 deploy.py --mode dry-run

# (2) 매단 채 기립 자세 추종 (정책 미실행) — contact 이 0 인지 확인
python3 deploy.py --mode stand --duration 10

# (3) 매단 채 정책 실행 (명령 0) — 발산 없이 안정적인 다리 움직임 확인
python3 deploy.py --mode hang --policy model/P3-final/exported/policy_numpy.npz --duration 10

# (4) 지면에서 제자리 (명령 0 램프만) → 짧은 전진
python3 deploy.py --mode walk --policy model/P3-final/exported/policy_numpy.npz \
    --vx 0.3 --duration 10
```

속도는 0.3 으로 시작해 안정되면 0.4 까지 올립니다 (아래 실기 기록 참고).

### 4.4 시간 옵션 — `--duration` 은 무엇을 재나

stand / hang / walk 는 전부 같은 타임라인으로 돕니다:

```
상태 워밍업 1 s → 기립 --stand-time (기본 5 s) → 유지 1 s → 게인 블렌딩 --gain-blend (1 s, hang/walk 만)
  → [모드 본체 --duration] → lie_down 2.5 s → damping
```

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--duration` | **20 s** | 모드 본체의 길이. dry-run = 관측 출력, stand = 기립 유지, hang/walk = 정책 실행. **모든 모드에 적용되며 생략하면 20 초** 돕니다. 처음 시도는 10 으로 짧게. |
| `--stand-time` | 5 s | 엎드린 자세 → 기본 자세 보간 시간. `--duration` 과 별개. |
| `--gain-blend` | 1 s | (hang/walk) 정책 인계 **전**, 기본 자세를 홀드한 채 Kp 60/Kd 1 → 20/0.5 로 내리는 시간. 정책은 첫 스텝부터 학습 게인에서 돕니다. |
| Enter | — | 어느 단계에서든 즉시 종료. 기립/유지/정책 중이면 damping, 정책이 정상 종료되면 lie_down 후 damping. |

즉 `--mode stand --duration 10` 은 "기립 5 초 + 유지 1 초 + **10 초 더 유지** + 주저앉기 2.5 초"
이고, `--mode walk --vx 0.3 --duration 10` 은 "기립 후 **10 초 동안 정책 실행**" 입니다.
hang/walk 에서 `--duration` 을 빼면 20 초 동안 정책이 돕니다.

부목을 채운 경우:

```bash
python3 deploy.py --mode walk --policy model/P3-final/exported/policy_numpy.npz \
    --vx 0.4 --injured-leg FR
```

`--injured-leg` 는 (1) 관측의 `peg_leg_one_hot` 을 켜고, (2) 학습과 동일하게 그
calf 의 action 을 0 으로 마스킹한 뒤 부목 고정각(-2.55 rad)으로 유지합니다.

---

## 안전 사양

1. **관측 차원 / 레퍼런스 검증** — 기립을 시작하기 **전에** 번들 입력 차원이
   `config.OBS_DIM` 과 맞는지, `reference_io.json` 의 행동값과 1e-3 안에서
   일치하는지 확인하고, 아니면 모터를 건드리지 않고 즉시 중단합니다.
2. **상태 워밍업 후 기립** — Go1 MCU 는 패킷을 보낸 클라이언트에게만 low-level
   상태를 회신하므로, 기립 전 1 초간 zero-torque 패킷으로 관절각을 받아 둡니다.
   그래도 관절각이 전부 0 이면 (하위제어 모드가 아니거나 회신 없음) 모터를
   건드리지 않고 중단합니다. 이걸 빼먹으면 엎드린 로봇에 다리를 뻗은 자세를
   Kp 60 으로 명령해 점프 후 전복합니다 (2026-09-17 실기).
3. **부드러운 기립** — 현재 관절각에서 기본 자세까지 `--stand-time` 동안 선형
   보간하며 게인을 램프업합니다.
4. **기울임 자동 셧다운** — roll/pitch 가 0.7 rad 를 넘으면 모든 제어를 끊고
   Kp=0 / Kd 만 남기는 damping 으로 전환합니다.
5. **관절 가동범위 클램프** — Go1 URDF 한계에 soft factor 0.9 를 적용해 목표각을
   사전 차단합니다.
6. **명령 클립** — 학습 분포 밖 속도 명령을 잘라냅니다.
7. **정상 종료 시 lie_down** — damping 직행은 Kp=0 이라 뚝 떨어지므로, 엎드림
   자세로 천천히 보간한 뒤 damping 합니다.

## 기술 사양

* **제어 주기** 50 Hz (`sim.dt 0.005 × decimation 4`)
* **action scale** 0.25, `target_q = default_q + 0.25 × action`
* **게인** 정책 Kp=20 / Kd=0.5 (학습에 쓰인 `DCMotor` 실제 값)
* **기립 게인** Kp=60 / Kd=1 → 정책 인계 **전**, 기본 자세 홀드 중 `--gain-blend`(기본 1초)에
  걸쳐 20/0.5 로 블렌딩. 정책은 Kp 60 을 한 스텝도 보지 않습니다.

  기립 게인이 높은 이유 (NX 실측): `stand_up()` 은 적분항도 중력 보상도 없는 순수
  P 제어라 정상상태 오차가 정확히 `필요토크 / Kp` 입니다. Kp=20 이면 뒷무릎 오차
  0.66 rad 로 바닥에 닿아 기립에 실패하고, Kp=60 이면 0.13 rad 로 수평 기립합니다.
  붕괴 자세의 뒷무릎 필요 토크는 12.9 Nm 지만 기립 후에는 5.1 Nm 로 떨어져서,
  같은 Kp=20 으로도 버틸 수 있게 됩니다.
* **인계 방식** 정책 출력은 인계 순간부터 100% 반영합니다 (행동 권한 램프 없음).
  학습 에피소드도 그렇게 시작하기 때문입니다. 게인은 인계 전에 이미 학습값으로 내려가
  있으므로, 인계 시점의 로봇 상태(기본 자세, Kp 20, hidden 0)가 학습 에피소드 시작과
  같습니다 — 아래 실측 참고.
* **관절 순서** Unitree SDK 는 다리별 `FR, FL, RR, RL`, Isaac Lab 은 타입별
  `[hips][thighs][calves]`. 변환은 `robot_io.py` 안에서만 일어납니다.
* **온보드 NX 제약** Python 3.6.9 / numpy 1.13.3, torch 도 onnxruntime 도 없고
  인터넷도 없습니다. 하드웨어에서 도는 백엔드는 순수 NumPy 번들(`.npz`) 하나뿐이고,
  추론은 3.4 ms/step (50 Hz 예산 20 ms 의 17%) 입니다.

## 실기 기록 (2026-08-03, 구 52차원 antalgic student)

P3-final 이전 모델로 얻은 결과입니다. 모델에 따라 달라지는 값도 있지만, 인계
방식과 로그 처리 교훈은 현재 코드에 반영돼 있습니다.

* **첫 15초 완주** (낙상 없음, 직진 방향 오차 ±7°, 2.5 m 이상).
* **행동 권한 램프는 쓰지 마세요.** 인계 후 3초 동안 정책 출력을 0→100% 로 올렸더니,
  "명령해도 몸이 안 움직이는" 학습에 없던 이력이 LSTM 에 쌓였습니다. 그 결과 멀쩡한
  RR 을 다친 다리로 추정했고(probe peg_RR 0.76), RR 을 아끼는 3족 보행에 고착돼 두
  번 넘어졌습니다. 램프를 없애자 해결됐습니다. 현재 코드에는 램프가 없습니다.
* **게인 블렌딩은 1초.** 인계 순간의 충격은 보통 크기의 보행 행동이 기립 강성(Kp=60)
  에서 실행돼서 생긴 것이었습니다. 행동을 줄이지 말고 고강성 구간을 3초→1초로
  줄였습니다. 현재 `GAIN_BLEND_TIME` 기본값입니다.
  * 2026-09-17 (P3-final, hang): 1초로도 인계 순간 한 번 "발작" 후 안정화. 블렌딩이
    정책 실행 중에 돌아서 첫 스텝은 여전히 Kp 60 이었기 때문입니다. 블렌딩을 인계
    **전** 홀드 구간으로 옮겨 정책이 Kp 60 을 아예 보지 않게 했습니다
    (`blend_gains()`).
* **넘어질 때 로그를 남기세요.** 기울임 가드로 루프가 끊기면서 로그가 유실된 적이
  있습니다. 지금은 루프가 어떻게 끝나든 `--log-npz` 가 저장됩니다.
* **추정치는 평활해서 보세요.** 50 Hz 순간값은 출렁이고, 인계 직후에는 일시적인 가짜
  스파이크가 나옵니다. 텔레메트리의 `L_hat` / `v_hat` 은 1초 EMA 로 평활되고, 인계
  후 `게인 블렌딩 + 1초` 동안은 `(수렴 중)` 이 붙습니다.
* **토크 여유** `--power-protect 9` 에서 토크 피크 24 Nm / 한계 35.5 Nm 로 포화 없음.
* 구 모델 전용 값 (P3-final 에는 그대로 적용하지 마세요):
  `--vx-floor 0.3` (그 모델은 vx 0.3–1.0 으로만 학습), `--wz 0.08` (그 모델의 우측
  요 드리프트 −0.05 rad/s 보정). 속도는 0.3 안정 / 0.4 권장 / 0.5 좌측 롤 발산 낙상.
