#!/usr/bin/env bash
set -euo pipefail

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MEMORY_FREE_THRESHOLD_MB="${MEMORY_FREE_THRESHOLD_MB:-1024}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"
PYTHON="${PYTHON:-conda run -n qwen3meld-run python}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-2026 2027 2028 2029 2030}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_LOG="${RUN_LOG:-logs/full_experiments_gpu_$(date -u +%Y%m%dT%H%M%SZ).log}"

mkdir -p logs

echo "Watcher started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"
echo "Checking GPU every ${CHECK_INTERVAL_SECONDS}s; free threshold=${MEMORY_FREE_THRESHOLD_MB}MB, util<=${MAX_GPU_UTIL}%" | tee -a "${RUN_LOG}"

while true; do
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) nvidia-smi not found; waiting" | tee -a "${RUN_LOG}"
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  IFS=',' read -r USED_MEM TOTAL_MEM GPU_UTIL < <(
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
      | head -n 1 \
      | tr -d ' '
  )
  FREE_MEM=$((TOTAL_MEM - USED_MEM))
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu used=${USED_MEM}MB free=${FREE_MEM}MB util=${GPU_UTIL}%" | tee -a "${RUN_LOG}"

  if [[ "${FREE_MEM}" -ge "${MEMORY_FREE_THRESHOLD_MB}" && "${GPU_UTIL}" -le "${MAX_GPU_UTIL}" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU is available; starting full experiments" | tee -a "${RUN_LOG}"
    PYTHON="${PYTHON}" \
      EPOCHS="${EPOCHS}" \
      SEEDS="${SEEDS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      DEVICE=cuda \
      bash scripts/run_all_experiments.sh 2>&1 | tee -a "${RUN_LOG}"
    STATUS=${PIPESTATUS[0]}
    echo "EXIT_CODE=${STATUS}" | tee -a "${RUN_LOG}"
    exit "${STATUS}"
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
