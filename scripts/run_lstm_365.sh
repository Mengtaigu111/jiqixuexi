#!/usr/bin/env bash
set -euo pipefail

MODEL=lstm OUTPUT_LEN=365 bash scripts/run_one_experiment.sh
