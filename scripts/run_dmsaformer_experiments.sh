#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHON

bash scripts/run_dmsaformer_90.sh
bash scripts/run_dmsaformer_365.sh
${PYTHON} -m src.calibrated_dmsaformer
${PYTHON} -m src.summarize_results
${PYTHON} -m src.export_dmsaformer_artifacts
