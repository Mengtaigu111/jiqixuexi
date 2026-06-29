#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?MODEL is required}"
OUTPUT_LEN="${OUTPUT_LEN:?OUTPUT_LEN is required}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-2026 2027 2028 2029 2030}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-auto}"
DATA_DIR="${DATA_DIR:-data/processed}"
METRICS_DIR="${METRICS_DIR:-results/metrics}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-results/predictions}"
FIGURES_DIR="${FIGURES_DIR:-results/figures}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"

for SEED in ${SEEDS}; do
  ${PYTHON} -m src.train \
    --model "${MODEL}" \
    --output_len "${OUTPUT_LEN}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --data_dir "${DATA_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --device "${DEVICE}"

  ${PYTHON} -m src.evaluate \
    --checkpoint "${SAVE_DIR}/${MODEL}_${OUTPUT_LEN}_seed${SEED}.pt" \
    --model "${MODEL}" \
    --output_len "${OUTPUT_LEN}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --data_dir "${DATA_DIR}" \
    --predictions_dir "${PREDICTIONS_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --figures_dir "${FIGURES_DIR}" \
    --device "${DEVICE}"
done
