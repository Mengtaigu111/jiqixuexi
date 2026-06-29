#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_TESTS="${SKIP_TESTS:-0}"
ALLOW_PLACEHOLDERS="${ALLOW_PLACEHOLDERS:-0}"
read -r -a PYTHON_CMD <<< "$PYTHON"

cd "$(dirname "$0")/.."

echo "Checking current submission artifacts..."

"${PYTHON_CMD[@]}" -c '
from pathlib import Path
import re
import pandas as pd


def pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


summary = pd.read_csv("results/metrics/summary.csv")
models = sorted(summary["Model"].unique())
expected_models = ["dmsaformer", "hybrid", "lstm", "transformer"]
if len(summary) != 8 or models != expected_models or not bool((summary["Runs"] == 5).all()):
    runs = summary["Runs"].tolist()
    raise SystemExit(f"Unexpected summary.csv contents: rows={len(summary)} models={models} runs={runs}")
print(f"summary.csv: {len(summary)} rows, models={models}, Runs all 5")

for pdf_path in ["report/report.pdf", "report/ML_household_power_report.pdf"]:
    path = Path(pdf_path)
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty PDF: {pdf_path}")
    pages = pdf_page_count(path)
    if pages != 10:
        raise SystemExit(f"Unexpected page count for {pdf_path}: {pages}")
    print(f"{pdf_path}: {pages} pages")

for image_path in [
    "results/figures/metric_bar_mse.png",
    "results/figures/metric_bar_mae.png",
    "results/figures/prediction_comparison_90.png",
    "results/figures/prediction_comparison_365.png",
]:
    path = Path(image_path)
    if not path.exists() or path.stat().st_size <= 0:
        raise SystemExit(f"Missing or empty figure: {image_path}")
    print(f"{image_path}: {path.stat().st_size} bytes")
'

for path in \
  "2026-专硕机器学习课程考核.pdf" \
  "data/raw/household_power_consumption.txt" \
  "data/processed/train_90.npz" \
  "data/processed/scaler.pkl" \
  "checkpoints/lstm_90_seed2026.pt" \
  "logs/full_experiments_gpu_20260618T132345Z.log" \
  "results/predictions/lstm_90_seed2026.csv"; do
  if git check-ignore -q -- "$path"; then
    echo "ignored: $path"
  else
    echo "Expected ignored path is not ignored: $path" >&2
    exit 1
  fi
done

if [[ "$ALLOW_PLACEHOLDERS" != "1" ]]; then
  if rg -q "你的用户名|你的仓库名|待填写" README.md report src/generate_report_pdf.py; then
    echo "Placeholder GitHub/author fields remain. Set ALLOW_PLACEHOLDERS=1 only for pre-fill checks." >&2
    rg -n "你的用户名|你的仓库名|待填写" README.md report src/generate_report_pdf.py >&2
    exit 1
  fi
else
  echo "placeholder check: skipped by ALLOW_PLACEHOLDERS=1"
fi

if [[ "$SKIP_TESTS" != "1" ]]; then
  "${PYTHON_CMD[@]}" -m pytest tests -q
else
  echo "pytest: skipped by SKIP_TESTS=1"
fi

echo "Submission readiness checks completed."
