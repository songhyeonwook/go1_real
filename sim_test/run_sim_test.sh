#!/usr/bin/env bash
# Deployment-parity test (Test B) in Isaac Sim for the three deployable students.
#
# Usage:
#   sim_test/run_sim_test.sh                 # all 3, headless, numpy backend
#   sim_test/run_sim_test.sh antalgic        # just one
#   BACKEND=torch VIDEO=1 sim_test/run_sim_test.sh antalgic
#
# Env vars: ISAACLAB (default ~/IsaacLab), BACKEND (numpy|torch), VIDEO (1 to record),
#           NUM_STEPS, CMD_VX, DRIVE_WITH (deploy|reference).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Isaac Lab lives in the 'isaac' conda env (py3.11), same one used for training.
PY="${PY:-$HOME/miniconda3/envs/isaac/bin/python}"
BACKEND="${BACKEND:-numpy}"
NUM_STEPS="${NUM_STEPS:-600}"
CMD_VX="${CMD_VX:-0.5}"
DRIVE_MODE="${DRIVE_MODE:-fixed}"   # asis | fixed | reference
# Real bring-up: stand at STAND_KP/KD, then run the policy at POLICY_KP/KD.
# Hardware currently stands + runs the policy at Kp60/Kd1 (training used Kp20/Kd0.5).
STAND_KP="${STAND_KP:-60}"; STAND_KD="${STAND_KD:-1}"
POLICY_KP="${POLICY_KP:-20}"; POLICY_KD="${POLICY_KD:-0.5}"   # policy near training gains
STAND_STEPS="${STAND_STEPS:-100}"
# base_lin_vel source (hardware can't measure it): command proxy = the deployed fix.
LINVEL_SOURCE="${LINVEL_SOURCE:-command}"

# model_dir : reference_checkpoint
declare -A CKPT=(
  [antalgic]="antalgic/exported:antalgic/antalgic_student_deploy.pt"
  [fault_tolerant]="fault_tolerant/exported:fault_tolerant/faulttol_student_deploy.pt"
  [symmetry]="symmetry/exported:symmetry/symmetry_student_deploy.pt"
)

# Reproduce the exact config the students were trained/evaluated under
# (see go1_peg/scripts/rsl_rl/eval_metrics_lo.sh). PROPRIO_ONLY -> policy obs = 48,
# INJURY_ONEHOT -> teacher privileged = 7 (needed for checkpoint load), FLAT_TERRAIN,
# and the PD actuator (Kp=20/Kd=0.5) the policy was trained against.
export GO1_PHASE="${GO1_PHASE:-student}"
export GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1
export GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8
export GO1_PD_ACTUATOR=1 GO1_PD_KP="${GO1_PD_KP:-20.0}" GO1_PD_KD="${GO1_PD_KD:-0.5}"
export GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.0

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(antalgic fault_tolerant symmetry)

VIDEO_FLAG=""
[ "${VIDEO:-0}" = "1" ] && VIDEO_FLAG="--video"

for m in "${MODELS[@]}"; do
  entry="${CKPT[$m]:-}"
  if [ -z "$entry" ]; then echo "unknown model: $m" >&2; exit 2; fi
  model_dir="${entry%%:*}"
  ckpt="${entry##*:}"
  echo "==================================================================="
  echo ">>> ${m}  (backend=${BACKEND}, drive_mode=${DRIVE_MODE})"
  echo "==================================================================="
  "${PY}" "${REPO_ROOT}/sim_test/sim_deploy_parity.py" \
    --task Template-Go1-Lab-v0 \
    --agent rsl_rl_distill_cfg_entry_point \
    --model_dir "${REPO_ROOT}/${model_dir}" \
    --checkpoint "${REPO_ROOT}/${ckpt}" \
    --backend "${BACKEND}" \
    --num_envs 1 \
    --num_steps "${NUM_STEPS}" \
    --cmd_vx "${CMD_VX}" \
    --drive_mode "${DRIVE_MODE}" \
    --stand_kp "${STAND_KP}" --stand_kd "${STAND_KD}" \
    --policy_kp "${POLICY_KP}" --policy_kd "${POLICY_KD}" \
    --stand_steps "${STAND_STEPS}" --linvel_source "${LINVEL_SOURCE}" \
    --report "${REPO_ROOT}/${model_dir}/sim_parity_report_${DRIVE_MODE}.json" \
    --headless ${VIDEO_FLAG}
done

echo "All done. Reports: <model>/exported/sim_parity_report.json"
