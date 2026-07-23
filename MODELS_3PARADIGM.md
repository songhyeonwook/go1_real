# 3-paradigm 모델 (제안 + 비교군 2종)

논문의 세 패러다임 최종 체크포인트. **동일 환경·부상모델·RMA·PD·DR·viability floor**,
**환부 다리 보상항만** 다름. (seed 42 대표 체크포인트)

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
