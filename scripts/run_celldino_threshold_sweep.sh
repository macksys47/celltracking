#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/dingyi/ct
DATASET="${1:?dataset is required}"
GPU="${2:?gpu id is required}"
SPLIT="${3:-val}"
TRACKING_ROOT="${4:-${ROOT}/results/celldino_tracking_sweep}"
PYTHON_BIN="${PYTHON_BIN:-/home/dingyi/miniconda3/envs/ct/bin/python}"
THRESHOLDS="${THRESHOLDS:-0.0 0.02 0.05 0.1 0.2 0.3 0.5}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

for TH in ${THRESHOLDS}; do
  SAFE_TH="${TH/./p}"
  OUT_DIR="${TRACKING_ROOT}/link_${SAFE_TH}/${DATASET}/${SPLIT}"
  LOG_DIR="${TRACKING_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  echo "infer CellDINO ${DATASET}/${SPLIT} link=${TH} mode=global gpu=${GPU}"
  "${PYTHON_BIN}" -m celldino_tracking.infer \
    --dataset "${DATASET}" \
    --data-dir "${ROOT}/data" \
    --output-dir "${OUT_DIR}" \
    --checkpoint "${ROOT}/results/celldino_tracking/${DATASET}/checkpoint_best.pth" \
    --split "${SPLIT}" \
    --link-mode global \
    --link-threshold "${TH}" \
    >"${LOG_DIR}/inf_${DATASET}_${SPLIT}_link_${SAFE_TH}.log" 2>&1
done
