#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHON

bash scripts/run_lstm_90.sh
bash scripts/run_lstm_365.sh
bash scripts/run_transformer_90.sh
bash scripts/run_transformer_365.sh
bash scripts/run_hybrid_90.sh
bash scripts/run_hybrid_365.sh
${PYTHON} -m src.summarize_results --models lstm transformer hybrid
