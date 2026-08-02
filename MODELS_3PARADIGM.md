# 3-paradigm 모델 (제안 + 비교군 2종)

논문의 세 패러다임 최종 체크포인트. **동일 환경·부상모델·RMA·PD·DR·viability floor**,
**환부 다리 보상항만** 다름. (seed 42 대표 체크포인트)

---

# ★ 최신 배포 모델 — phase3_antalgic_s42 (2026-08-02)

**관측이 R⁴⁸ → R⁵² 로 바뀌었습니다.** 아래 3-paradigm 절의 모델들과 관측 차원이
다르므로 섞어 쓸 수 없습니다. 배포 스택(`sdk_deploy/observation.py`, `config.py`)은
이미 52 기준(`POLICY_OBS_DIM = 52`)으로 갱신돼 있습니다.

```
antalgic/                             ← deploy_student.launch 가 찾는 위치
  exported/policy_numpy.npz     ★ 실기에서 실제로 도는 것 (LSTM+MLP, obs 52)
  exported/policy.pt|.onnx        개발 PC 검증·변환용
  exported/policy_io.json         관측 레이아웃·아키텍처 메타데이터
  antalgic_student_deploy.pt      원본 체크포인트 (model_3200)
  params_student/                 학습 config (agent.yaml + env.yaml)

sdk_deploy/model/phase3_antalgic_s42/  ← 버전 보관용 동일 사본
```

실행:
```bash
./scripts/sync_to_robot.sh --go
roslaunch go1_real deploy_student.launch paradigm:=antalgic
```
(`deploy.launch` 는 Kp=30/Kd=1.5 와 action multiplier 0.2 를 강제하므로 student 에 쓰지 마세요.)

## 관측 R⁵² 레이아웃

| 구간 | 항목 | 차원 |
|---|---|---|
| 0:3 | base_lin_vel | 3 |
| 3:6 | base_ang_vel | 3 |
| 6:9 | projected_gravity | 3 |
| 9:12 | velocity_commands | 3 |
| 12:24 | joint_pos_rel | 12 |
| 24:36 | joint_vel_rel | 12 |
| 36:48 | last_actions | 12 |
| **48:52** | **calf_pos_abs** (신규) | **4** |

`calf_pos_abs = q_calf − NOMINAL_CALF` (FL,FR,RL,RR). 표준 `joint_pos_rel` 은
`default_joint_pos` 를 빼는데, 부상 시 그 기준이 부목 각도로 바뀌어 **부목 길이 정보가
관측에서 상쇄되어 사라집니다**. 이 채널은 부상 이전 nominal 을 기준으로 빼므로 각도가
그대로 드러납니다 (부목 0.20 m → −0.66 rad, 0.30 m → −0.08 rad).

## 실기 주의사항

**LSTM 상태 리셋이 필수입니다.** phase1 은 feed-forward 라 상태가 없었지만 이 모델은
순환입니다. 기립 후 정책을 켤 때와 중단 후 재개할 때 hidden/cell 을 0 으로 초기화하세요.
안 하면 이전 세션의 기억이 남습니다.

**부상 상태 파악에 1~4초 걸립니다.** 시뮬레이션 실측(LSTM hidden → 파라미터 선형 probe):

| 리셋 후 경과 | 부목 길이 R² | 마찰 R² |
|---|---|---|
| 0~10 스텝 | 0.86 | −0.05 |
| 50~100 | **0.96** | 0.59 |
| 100~200 | 0.92 | **0.62** |
| 500+ | 0.57 | 0.27 |

부목 길이는 관절 엔코더로 즉시 읽히지만 **마찰은 미끄러짐을 관찰해야 하므로 1~4초의
보행이 필요**합니다. 첫 몇 초 거동이 불안정할 수 있으니 안전 확보 후 시작하세요.

**보행 양상**: 부상 다리 하중 duty 0.03~0.14 로 **3족 보행에 가깝습니다** — 발을 땅에
대지만 체중은 거의 싣지 않습니다. 하중 경감 75~82%, 네 부상 조건 모두 명령 속도 95%+
추종(0.3/0.6/1.0 m/s 검증). 실기에서도 "부상 다리를 들고 세 다리로 걷는" 모습이 예상됩니다.

## 검증 완료

`policy_numpy.npz` 와 PyTorch 원본(`policy.pt`)의 출력이 30스텝 시퀀스에서
**최대 절대오차 9.5e-07** 로 일치합니다. NumPy 백엔드가 시뮬레이션과 동일하게 동작합니다.

## 재생성 절차

```bash
# 1) JIT/ONNX 내보내기 (학습과 동일한 GO1_* 필수, 특히 GO1_ABS_JOINT_OBS=1)
cd /home/shw/go1_compar/scripts/rsl_rl
GO1_PHASE=student GO1_ABS_JOINT_OBS=1 GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 \
GO1_FLAT_TERRAIN=1 GO1_PHASE2_GAIT_TUNING=1 GO1_PD_ACTUATOR=1 \
python play.py --task Template-Go1-Lab-v0 --agent rsl_rl_distill_cfg_entry_point \
  --checkpoint <student model_N.pt> --headless --num_envs 1
# exported/ 생성 직후 Ctrl+C (play 는 시뮬 루프를 계속 돕니다)

# 2) NumPy 번들 변환 (개발 PC, onnx 필요)
python3 /home/shw/go1_real/scripts/export_policy_numpy.py <run>/exported/policy.onnx
```

**출처**: student `phase3_ws_antalgic_s42/model_3200` ←
teacher `phase2_speedfix_antalgic_s42/model_13950` ← phase1 `phase1_mlp_s42/model_5999`

---

# (이전) R⁴⁸ 3-paradigm 세트

아래는 `calf_pos_abs` 도입 이전의 R⁴⁸ 모델들입니다. **위 모델과 관측 차원이 다릅니다.**

## 파일

```
antalgic/                                    ← 제안 알고리즘 (통각 보상)
  antalgic_student_deploy.pt      ★ 배포용: proprioception R⁴⁸만 → 실기 Go1 배포 가능
  antalgic_teacher_widespeed.pt     넓은속도 teacher (privileged, 학습용)
  antalgic_teacher_s42_compare.pt   저속 teacher (비교 기준, seed 42)
  params_student/  params_teacher/  (agent.yaml + env.yaml = 로드 config)
fault_tolerant/
  faulttol_teacher_s42.pt           비교군: 통각 없음(alive bonus). teacher(privileged)
  params/
symmetry/
  symmetry_teacher_s42.pt           비교군: 좌우대칭 penalty. teacher(privileged)
  params/
```

## ★ 배포 가능성 (중요)

| 모델 | 관측 | 실기 배포 |
|---|---|---|
| **antalgic_student_deploy.pt** | proprioception R⁴⁸만 (LSTM) | ✅ **배포 가능** |
| **faulttol / symmetry student** | proprioception R⁴⁸만 (LSTM) | ✅ **배포 가능** |
| 각 teacher 체크포인트 | + privileged z (I_idx, L_peg, μ) | ❌ 학습·비교 전용 |

→ **세 패러다임 모두 실기 배포 가능**합니다. 비교군 2종도 Phase-3 distillation으로
proprioception-only student가 제작되어 `fault_tolerant/` · `symmetry/`에 포함돼 있습니다
(`*_student_deploy.pt` + `exported/policy.onnx` + `policy_numpy.npz` + config/metadata).
teacher 체크포인트는 privileged z를 받으므로 논문 §2.3 비교 목적에 한해 사용합니다.

## 배포 절차 (student → 실기)

⚠️ Go1 온보드 NX에는 **PyTorch도 ONNX Runtime도 없고 인터넷도 없습니다** (Python 3.6.9 /
numpy 1.13.3). 실기에서 도는 백엔드는 순수 NumPy 번들(`policy_numpy.npz`)뿐이며,
`deploy_policy.py`가 LSTM(hidden 256)을 NumPy로 직접 실행합니다 (NX 실측 3.4 ms/step).

1. **NumPy 번들 생성** (개발 PC, `pip install onnx` 필요) — 이미 커밋돼 있으므로 새 체크포인트를
   내보낼 때만 필요합니다:
   ```bash
   python3 scripts/export_policy_numpy.py antalgic/exported/policy.onnx \
       --env-yaml antalgic/params_student/env.yaml
   ```
2. **로봇으로 전송**: `./scripts/sync_to_robot.sh --go`
3. **실행**: `roslaunch go1_real deploy_student.launch paradigm:=antalgic`
   (`deploy.launch`는 Kp=30/Kd=1.5와 action multiplier 0.2를 강제하므로 student에 쓰지 마세요.)

### 실기에서 확인된 필수 설정

| 항목 | 값 | 근거 |
|---|---|---|
| `Kp` / `Kd` | 20.0 / 0.5 | `params_student/env.yaml`의 DCMotor 실제 학습 게인 |
| `stand_up_Kp` | 60.0 | Kp=20으로 기립 시 뒷무릎 0.66 rad 새그 → 바닥 접촉. 60에서 0.13 rad |
| `action_scale_multiplier` | 1.0 | 0.2에서는 0/4 다리, 보행 자체가 발화하지 않음 |

기립 후 `policy_ramp_time`(8초) 동안 Kp가 60 → 20으로 블렌딩되며, 램프 종료 시점에 Kp=20에서
`projected_gravity_z = -1.000`으로 정책이 스스로 몸을 지탱하는 것을 확인했습니다.

### 진단
```bash
rosrun go1_real check_stand.py      # 기립 품질 (목표 대비 오차, tauEst/Kp)
rosrun go1_real check_gait.py 5     # 보행 여부 (다리별 진폭, N/4 legs moving)
```

## 성능 (n=10, §2.3)

| 지표 | antalgic | fault-tol | symmetry |
|---|---|---|---|
| GRF 감소% | **81.9±2.0** | −30.4±55 | −32.3±48 |
| SI (eq.7)% | **139.6±5.1** | 4.8±32 | −7.8±22 |
| direction% | **82.5±5.8** | 51.7±13 | 71.4±7.5 |

antalgic이 3지표 모두 유의 우월 (Mann–Whitney U, p_adj≤0.003, Cliff's δ=+1.00).

**출처 체크포인트**:
- antalgic student: phase3_uc_s42/model_5600 · teacher: phase2_uc3_s42/model_32000, phase2_uni2_s42/model_17999
- fault-tol: phase2_cmp_faulttol_s42/model_17999 · symmetry: phase2_cmp_symmetry_s42/model_17999
