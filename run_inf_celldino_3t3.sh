#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/dingyi/ct
EXP=3t3
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m celldino_tracking.infer \
  --dataset "${EXP}" \
  --data-dir "${ROOT}/data" \
  --output-dir "${ROOT}/results/celldino_tracking/${EXP}/test" \
  --checkpoint "${ROOT}/results/celldino_tracking/${EXP}/checkpoint_best.pth" \
  --split test >"inf_celldino_${EXP}.log" 2>&1
