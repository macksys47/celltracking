#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/dingyi/ct
DATASET="${1:?dataset is required}"
GPU="${2:?gpu id is required}"
SPLIT="${3:-test}"
LINK_THRESHOLD="${4:?link threshold is required}"
TRACKING_ROOT="${5:-${ROOT}/results/celldino_tracking_fixed}"
PYTHON_BIN="${PYTHON_BIN:-/home/dingyi/miniconda3/envs/ct/bin/python}"
SAFE_TH="${LINK_THRESHOLD/./p}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG_DIR="${TRACKING_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "infer calibrated CellDINO ${DATASET}/${SPLIT} link=${LINK_THRESHOLD} mode=global gpu=${GPU}"
"${PYTHON_BIN}" -m celldino_tracking.infer \
  --dataset "${DATASET}" \
  --data-dir "${ROOT}/data" \
  --output-dir "${TRACKING_ROOT}/${DATASET}/${SPLIT}" \
  --checkpoint "${ROOT}/results/celldino_tracking/${DATASET}/checkpoint_best.pth" \
  --split "${SPLIT}" \
  --link-mode global \
  --link-threshold "${LINK_THRESHOLD}" \
  >"${LOG_DIR}/inf_${DATASET}_${SPLIT}_global_link_${SAFE_TH}.log" 2>&1
