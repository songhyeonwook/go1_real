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
| **antalgic_student_deploy.pt** | proprioception R⁴⁸만 (LSTM) | ✅ **배포 가능** (privileged 불필요) |
| antalgic_teacher_* | + privileged z (I_idx, L_peg, μ) | ❌ 학습·비교 전용 |
| faulttol / symmetry teacher | + privileged z | ❌ **비교군은 teacher만** — 실기 배포하려면 별도 student distillation 필요 |

→ **실기 배포는 `antalgic/antalgic_student_deploy.pt` 하나**. 두 비교군은 **논문 §2.3
비교 목적**(privileged teacher)이며, 그대로는 실기 배포 불가(proprioception-only student
미제작). 필요 시 각 baseline도 Phase-3 distillation으로 student를 만들면 배포 가능.

## 배포 절차 (antalgic student → 실기)

이 repo의 `deploy_policy.py`는 `model/policy.pt`(또는 .onnx) + `deployment_config.json`을
로드한다. 현재 model/은 **Phase-1 healthy** 정책. antalgic student로 교체하려면:

1. **ONNX/JIT export** (RSL-RL 체크포인트 → 배포형식):
   ```bash
   # go1_antalgic (또는 go1_peg) 환경에서
   GO1_PHASE=student python3 export_policy_onnx.py \
     --checkpoint /home/shw/go1_real/antalgic/antalgic_student_deploy.pt \
     --agent rsl_rl_distill_cfg_entry_point --phase student --headless
   ```
2. 산출된 `policy.onnx` + `deployment_config.json`을 `model/`에 배치(또는 model_path 지정).
3. `deploy_policy.py`의 obs 순서·PD gain(Kp20/Kd0.5)·action scale을 학습과 일치시킴
   (`params_student/env.yaml` 참조).

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
