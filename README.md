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

```bash
./scripts/sync_to_robot.sh --go
# NX 에서:
cd ~/go1_ws/src/go1_real/sdk_deploy
python3 deploy.py --mode dry-run                      # 손으로 움직여 부호/순서 확인
python3 deploy.py --mode stand                        # 기립만
python3 deploy.py --mode hang  --policy model/P3-final/exported/policy_numpy.npz
python3 deploy.py --mode walk  --policy model/P3-final/exported/policy_numpy.npz --vx 0.4
```

**반드시 이 순서로** 진행하세요. `hang` 은 로봇을 매단 상태에서 정책을 돌려
발산 없이 트로트 비슷하게 움직이는지 보는 단계입니다.

부목을 채운 경우:

```bash
python3 deploy.py --mode walk --policy .../policy_numpy.npz --vx 0.4 --injured-leg FR
```

`--injured-leg` 는 (1) 관측의 `peg_leg_one_hot` 을 켜고, (2) 학습과 동일하게 그
calf 의 action 을 0 으로 마스킹한 뒤 부목 고정각(-2.55 rad)으로 유지합니다.

---

## 안전 사양

1. **관측 차원 / 레퍼런스 검증** — 기립을 시작하기 **전에** 번들 입력 차원이
   `config.OBS_DIM` 과 맞는지, `reference_io.json` 의 행동값과 1e-3 안에서
   일치하는지 확인하고, 아니면 모터를 건드리지 않고 즉시 중단합니다.
2. **부드러운 기립** — 현재 관절각에서 기본 자세까지 `--stand-time` 동안 선형
   보간하며 게인을 램프업합니다.
3. **기울임 자동 셧다운** — roll/pitch 가 0.7 rad 를 넘으면 모든 제어를 끊고
   Kp=0 / Kd 만 남기는 damping 으로 전환합니다.
4. **관절 가동범위 클램프** — Go1 URDF 한계에 soft factor 0.9 를 적용해 목표각을
   사전 차단합니다.
5. **명령 클립** — 학습 분포 밖 속도 명령을 잘라냅니다.
6. **정상 종료 시 lie_down** — damping 직행은 Kp=0 이라 뚝 떨어지므로, 엎드림
   자세로 천천히 보간한 뒤 damping 합니다.

## 기술 사양

* **제어 주기** 50 Hz (`sim.dt 0.005 × decimation 4`)
* **action scale** 0.25, `target_q = default_q + 0.25 × action`
* **게인** 정책 Kp=20 / Kd=0.5 (학습에 쓰인 `DCMotor` 실제 값)
* **기립 게인** Kp=60 / Kd=1 → 정책 인계 후 3초에 걸쳐 20/0.5 로 블렌딩

  기립 게인이 높은 이유 (NX 실측): `stand_up()` 은 적분항도 중력 보상도 없는 순수
  P 제어라 정상상태 오차가 정확히 `필요토크 / Kp` 입니다. Kp=20 이면 뒷무릎 오차
  0.66 rad 로 바닥에 닿아 기립에 실패하고, Kp=60 이면 0.13 rad 로 수평 기립합니다.
  붕괴 자세의 뒷무릎 필요 토크는 12.9 Nm 지만 기립 후에는 5.1 Nm 로 떨어져서,
  같은 Kp=20 으로도 버틸 수 있게 됩니다.
* **관절 순서** Unitree SDK 는 다리별 `FR, FL, RR, RL`, Isaac Lab 은 타입별
  `[hips][thighs][calves]`. 변환은 `robot_io.py` 안에서만 일어납니다.
* **온보드 NX 제약** Python 3.6.9 / numpy 1.13.3, torch 도 onnxruntime 도 없고
  인터넷도 없습니다. 하드웨어에서 도는 백엔드는 순수 NumPy 번들(`.npz`) 하나뿐이고,
  추론은 3.4 ms/step (50 Hz 예산 20 ms 의 17%) 입니다.
