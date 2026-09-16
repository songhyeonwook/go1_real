#!/usr/bin/env bash
# Isaac Sim 배포 파리티 테스트 실행 래퍼.
#
#   sim_test/run_sim_test.sh                         # 기본 (정상 조건, npz 백엔드)
#   BACKEND=onnx sim_test/run_sim_test.sh            # 다른 백엔드
#   PEG_LEG=rl sim_test/run_sim_test.sh              # 부상 조건 (fl|fr|rl|rr|normal|balanced)
#   HEADLESS=0 NUM_STEPS=1200 sim_test/run_sim_test.sh
#
# 학습 설정은 phase yaml 이 기준입니다 (GO1_* 환경변수 방식은 go1_lod 에서
# 없어졌습니다). 모델을 바꾸려면 MODEL_DIR 만 바꾸면 됩니다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/isaac/bin/python}"
GO1_LOD_RSL="${GO1_LOD_RSL:-$HOME/go1_lod/scripts/rsl_rl}"

MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/sdk_deploy/model/P3-final}"
BACKEND="${BACKEND:-npz}"
case "${BACKEND}" in
  npz|numpy) BUNDLE="policy_numpy.npz" ;;
  onnx)      BUNDLE="policy.onnx" ;;
  torch|pt)  BUNDLE="policy.pt" ;;
  *) echo "unknown BACKEND: ${BACKEND} (npz|onnx|torch)" >&2; exit 2 ;;
esac

PHASE_CFG="${PHASE_CFG:-${GO1_LOD_RSL}/configs/phase/3/phase3.yaml}"
CKPT="${CKPT:-${MODEL_DIR}/model_3999.pt}"
NUM_STEPS="${NUM_STEPS:-600}"
PEG_LEG="${PEG_LEG:-normal}"
DRIVE="${DRIVE:-deploy}"

HEADLESS_FLAG=""
[ "${HEADLESS:-1}" = "1" ] && HEADLESS_FLAG="--headless"

for f in "${CKPT}" "${MODEL_DIR}/exported/${BUNDLE}" "${PHASE_CFG}"; do
  [ -f "$f" ] || { echo "not found: $f" >&2; exit 2; }
done

echo "==================================================================="
echo ">>> ${MODEL_DIR##*/}  backend=${BACKEND}  peg_leg=${PEG_LEG}  drive=${DRIVE}"
echo "==================================================================="

GO1_LOD_RSL="${GO1_LOD_RSL}" "${PY}" "${REPO_ROOT}/sim_test/sim_deploy_parity.py" \
  --phase_config_path "${PHASE_CFG}" \
  --checkpoint "${CKPT}" \
  --policy "${MODEL_DIR}/exported/${BUNDLE}" \
  --peg_leg "${PEG_LEG}" \
  --num_envs 1 \
  --num_steps "${NUM_STEPS}" \
  --drive "${DRIVE}" \
  --clean \
  ${HEADLESS_FLAG}
