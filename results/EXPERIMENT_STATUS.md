# Experiment Status

The project has been preprocessed with the real UCI household power dataset:

- Raw file: `data/raw/household_power_consumption.txt` (ignored by git)
- Daily data: `data/processed/daily_power.csv`
- Date range: 2006-12-16 to 2010-11-26
- Daily rows: 1442
- `train_90/valid_90/test_90`: 353/366/366 samples
- `train_365/valid_365/test_365`: 78/91/91 samples

Synthetic smoke outputs were removed before the formal run. The GPU watcher waited until the card was idle, then completed the formal full experiment matrix.

Watcher command used:

```bash
tmux new-session -d -s jiqixuexi_gpu_watch \
  "cd /home/myluo/jiqixuexi && CHECK_INTERVAL_SECONDS=300 MEMORY_FREE_THRESHOLD_MB=20000 MAX_GPU_UTIL=5 PYTHON='conda run -n qwen3meld-run python' EPOCHS=30 SEEDS='2026 2027 2028 2029 2030' BATCH_SIZE=64 bash scripts/watch_gpu_and_run_full_experiments.sh"
```

The watcher wrote `logs/full_experiments_gpu_*.log`; the final log contains `EXIT_CODE=0`.

Formal result status:

- `results/metrics/summary.csv`: generated
- `results/metrics/*_test_metrics.csv`: 30 files
- `results/predictions/*.csv`: 30 files
- `results/figures/*_curve.png`: 30 files
- `results/figures/*_error.png`: 30 files
- `results/screenshots/*.png`: 64 files
- `summary.csv`: every model/horizon has `Runs=5`
