#!/usr/bin/env bash
set -euo pipefail

MODEL=lstm OUTPUT_LEN=90 bash scripts/run_one_experiment.sh
