# Goal 2 Plan: DMSAFormer

## Research objective

Add and verify a third custom improved model, DMSAFormer, for multivariate household power consumption forecasting. The model must use 90 days of historical daily features to predict `global_active_power` for 90-day and 365-day horizons.

## Code deliverable

- `src/models/dmsaformer.py`: MovingAverage, SeriesDecomposition, VariableAttention, DMSAFormer.
- Training/evaluation integration through the existing CLI where practical, plus compatibility scripts matching the requested examples if needed.
- Results for five seeds each at `pred_len=90` and `pred_len=365`.
- Summary CSV and prediction plots using the requested DMSAFormer filenames or clearly documented equivalent paths.

## Scope

- Reuse existing UCI household power preprocessing, dataset, training, evaluation, plotting, and summary infrastructure.
- Add only the DMSAFormer model and the minimum integration needed to train and evaluate it.
- Preserve completed LSTM, Transformer, and Hybrid work from `goal-1`.

## Non-goals

- Do not rerun or alter completed baseline LSTM/Transformer/Hybrid experiments unless needed for compatibility checks.
- Do not change metric definitions, data split protocol, target variable, or processed data semantics.
- Do not overwrite raw data or remove existing results.

## Relevant assets

- Objective: `goal/goal-2.md`
- Existing project task file: `tasks/todo.md`
- Data: `data/raw/household_power_consumption.txt`, `data/processed/*.npz`, `data/processed/daily_power.csv`
- Existing code: `src/data_preprocess.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/summarize_results.py`, `src/models/`
- Existing tests: `tests/`

## Assumptions

- The first course-project goal is complete and should be treated as prior work.
- Existing train/evaluate infrastructure can be extended by registering a new model name.
- Existing processed data already satisfies the goal-2 data requirements unless audit proves otherwise.

## Risks

- Requested file names differ from current infrastructure paths (`results/metrics/*`, `results/figures/*`).
- Full 10-run training may take significant GPU time; if the GPU is busy, use tmux watcher rather than CPU full experiments.
- Existing code may not expose enough metadata to confirm scaler and original-scale evaluation without inspection.

## Implementation strategy

1. Audit existing model, dataset, training, evaluation, summary, and tests.
2. Add failing tests for DMSAFormer components and CLI registration.
3. Implement DMSAFormer with decomposition, trend branch, variable attention, multi-scale convolution, Transformer encoder, and prediction head.
4. Reuse existing training/evaluation scripts for 90/365 runs.
5. Add DMSAFormer-specific experiment runner or wrappers only if existing scripts cannot meet requested commands/outputs.
6. Run smoke verification, then full five-seed experiments for both horizons when resources allow.

## Verification strategy

- Unit tests: random tensor output shape `[batch, pred_len]` for pred_len 90 and 365; decomposition and variable attention shape checks.
- CLI smoke: train/evaluate one short DMSAFormer run with a small epoch count.
- Full verification: five seeds for pred_len 90 and five seeds for pred_len 365 with test MSE/MAE CSVs, summary mean/std, prediction CSVs, and plots.
- Artifact audit: confirm requested or documented result paths exist and are non-empty.

## Reproducibility strategy

- Use conda environment `qwen3meld-run` unless a better project environment is identified.
- Record every train/evaluate/summarize command in `experiment_log.md`.
- Prefer tmux for long-running experiments and keep logs under `logs/`.
- Use seeds `2026,2027,2028,2029,2030`.

## Rollback strategy

- Code changes are limited to new model file, model registry, tests, scripts, docs, and DMSAFormer result artifacts.
- Existing baseline artifacts are preserved.
- If a run fails, keep logs and do not delete previous valid artifacts unless replacing only DMSAFormer failed outputs.

## Current findings

- `tasks/todo.md` indicates `goal-1` is complete: data preprocessing, LSTM, Transformer, Hybrid, 30 formal experiments, plots, and report were verified.
- `goal/goal-2.md` is a new objective for DMSAFormer only; it explicitly says baseline LSTM and baseline Transformer do not need to be implemented again.
- Current result directories contain LSTM, Transformer, and Hybrid outputs, but no DMSAFormer artifacts were observed in the initial file listing.

## Optimization extension: DMSAFormer performance improvement

### Research objective

Improve the third model after its first formal result underperformed the completed LSTM/Transformer/Hybrid baselines, while preserving the original DMSAFormer requirements and evaluation protocol.

### Evidence

- Original DMSAFormer 90-day MSE mean: `203046.764429`; best 90-day baseline: Hybrid `155633.435294`.
- Original DMSAFormer 365-day MSE mean: `398765.923895`; best 365-day baseline: LSTM `316352.062831`.
- 365-day training set has only `78` windows, making high-capacity attention-only forecasting fragile.

### Literature-informed strategy

- Use DLinear-style decomposition linear mapping as the main low-variance target-channel forecast.
- Keep DMSAFormer decomposition, variable attention, 3/7/30-day multi-scale convolution, and Transformer encoder as residual correction modules.
- Avoid changing data splits, metrics, target scaling, or baseline results.

### Verification strategy

- Add tests proving the improved DMSAFormer exposes the new decomposition-linear backbone and still returns `[batch, pred_len]`.
- Run a 90/365 smoke experiment in isolated optimization output directories.
- If smoke works, run formal five-seed DMSAFormer reruns and compare against previous DMSAFormer and baselines.
