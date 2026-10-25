#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/dingyi/ct
EXP=3t3
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m trackastra_tracking.train \
  --dataset "${EXP}" \
  --data-dir "${ROOT}/data" \
  --output-dir "${ROOT}/results/trackastra_tracking/${EXP}" \
  --epochs 30 \
  --train-backbone >"train_trackastra_${EXP}.log" 2>&1
