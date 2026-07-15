#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHON

bash scripts/run_dmsaformer_90.sh
bash scripts/run_dmsaformer_365.sh
# Fair comparison: writes RAW (uncalibrated) test metrics for all three formal
# models into results/metrics/*_test_metrics.csv, plus the raw-vs-calibrated
# ablation table (results/metrics/fair_comparison_summary.csv). This keeps the
# main table on a symmetric raw footing for every model. The old
# `src.calibrated_dmsaformer` step is intentionally NOT run here, because it
# only calibrated DMSAFormer and left the baselines raw, which biased the table.
${PYTHON} -m src.fair_comparison
${PYTHON} -m src.summarize_results --models lstm transformer dmsaformer
${PYTHON} -m src.export_dmsaformer_artifacts
