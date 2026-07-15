# 基于深度学习的家庭电力消耗多变量时间序列预测

本项目用于 2026 年专硕机器学习课程项目：使用家庭电力消耗多变量时间序列，根据过去 90 天数据分别预测未来 90 天和 365 天 `global_active_power` 曲线。最终报告正式比较 LSTM、Transformer 和 DMSAFormer 三个模型。三模型在 2 个预测长度 x 5 个 seed 上训练与评估，共 30 次正式对比实验；HybridTCNTransformer 仅作为 DMSAFormer 改进过程中的中间结构/消融对照，不进入主结果表。

**正式主表口径**：三个模型全部报告未经任何后处理的原始 test MSE/MAE，确保完全对称的公平对比。结果如实报告：90 天最优模型是 Transformer，365 天最优模型是 LSTM，本文提出的 DMSAFormer 在两个 horizon 上都居中、未超过最强 baseline。作为附加消融，`src/fair_comparison.py` 对三个模型施加同一套 validation-only 仿射校准（`y = a * pred + b`，仅在 valid 上拟合、不接触 test 标签），结果表明校准可小幅降低误差但不改变模型排名。

## 数据来源

电力数据使用 UCI Machine Learning Repository 的 Individual household electric power consumption 数据集：

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

课程允许融合天气变量，天气数据可来自：

https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles

将数据放入 `data/raw/`。脚本会优先识别：

- `train.csv`
- `test.csv`
- `tes.csv`
- `household_power_consumption.txt`
- `household_power_consumption.csv`
- `weather.csv`

`tes.csv` 和 `test.csv` 均兼容。大数据文件不会提交到 GitHub。

## 环境安装

推荐使用已有 conda 环境。本机验证可用环境为 `qwen3meld-run`：

```bash
conda activate qwen3meld-run
```

也可以新建环境：

```bash
conda create -n household-power python=3.11 -y
conda activate household-power
pip install -r requirements.txt
```

如需 GPU 训练，请安装与本机 CUDA 匹配的 PyTorch。

## 数据预处理

真实数据：

```bash
python -m src.data_preprocess --data_dir data/raw --output_dir data/processed
```

无真实数据时只做链路 smoke test：

```bash
python -m src.data_preprocess --data_dir data/raw --output_dir data/processed --generate_smoke_data
```

预处理会处理缺失值，将 `?`、空字符串和非法数值转为 NaN。核心电力传感器列采用按时间排序后的前向填充和后向填充；若核心列仍存在不可恢复缺失，脚本会报错而不是将功率缺失直接补 0。可选天气列缺失时仅作为占位字段补 0。分钟级数据按天聚合：功率和分表能耗求和，电压和电流取平均，天气变量取当天第一个非空值，并计算 `sub_metering_remainder`。

输出文件：

- `data/processed/daily_power.csv`
- `data/processed/train_90.npz`
- `data/processed/valid_90.npz`
- `data/processed/test_90.npz`
- `data/processed/train_365.npz`
- `data/processed/valid_365.npz`
- `data/processed/test_365.npz`
- `data/processed/scaler.pkl`

## 训练

单个实验示例：

```bash
python -m src.train --model lstm --output_len 90 --seed 2026 --epochs 30 --device auto
```

正式报告支持模型：

- `lstm`
- `transformer`
- `dmsaformer`

`hybrid` 代码仍保留用于内部消融和 DMSAFormer 局部时序主干探索，但不是正式第三个提交模型。

支持预测长度：

- `90`
- `365`

训练日志保存到 `results/metrics/{model}_{output_len}_seed{seed}.csv`，最佳 checkpoint 保存到 `checkpoints/{model}_{output_len}_seed{seed}.pt`。

## 测试与评估

```bash
python -m src.evaluate \
  --checkpoint checkpoints/lstm_90_seed2026.pt \
  --model lstm \
  --output_len 90 \
  --seed 2026
```

评估会保存：

- `results/metrics/{model}_{output_len}_seed{seed}_test_metrics.csv`
- `results/predictions/{model}_{output_len}_seed{seed}.csv`
- `results/figures/{model}_{output_len}_seed{seed}_curve.png`
- `results/figures/{model}_{output_len}_seed{seed}_error.png`

## 复现实验

基础与中间改进实验脚本默认运行 LSTM、Transformer 和 HybridTCNTransformer。正式报告主表只采用 LSTM、Transformer 和 DMSAFormer；HybridTCNTransformer 结果只作为中间改进/消融记录使用。

```bash
bash scripts/run_all_experiments.sh
```

如果没有先 `conda activate`，可显式指定解释器：

```bash
PYTHON="conda run -n qwen3meld-run python" bash scripts/run_all_experiments.sh
```

如果 GPU 正在被占用，可用 tmux 轮询等待 GPU 空闲后自动启动完整实验：

```bash
tmux new-session -d -s jiqixuexi_gpu_watch \
  "cd /home/myluo/jiqixuexi && CHECK_INTERVAL_SECONDS=300 MEMORY_FREE_THRESHOLD_MB=20000 MAX_GPU_UTIL=5 PYTHON='conda run -n qwen3meld-run python' EPOCHS=30 SEEDS='2026 2027 2028 2029 2030' BATCH_SIZE=64 bash scripts/watch_gpu_and_run_full_experiments.sh"
```

watcher 日志写入 `logs/full_experiments_gpu_*.log`。默认每 300 秒检查一次 `nvidia-smi`，当空闲显存达到阈值且 GPU 利用率足够低时，用 `DEVICE=cuda` 启动完整实验。

默认 seeds 为 `2026 2027 2028 2029 2030`。快速 smoke：

```bash
PYTHON="conda run -n qwen3meld-run python" EPOCHS=1 SEEDS="2026" BATCH_SIZE=16 bash scripts/run_lstm_90.sh
conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer dmsaformer
```

运行 DMSAFormer 并重新生成三模型正式结果：

```bash
PYTHON="conda run -n qwen3meld-run python" EPOCHS=30 SEEDS="2026 2027 2028 2029 2030" BATCH_SIZE=32 bash scripts/run_dmsaformer_experiments.sh
```

正式主表对三个模型（LSTM、Transformer、DMSAFormer）统一采用未经任何后处理的原始 test MSE/MAE，三者完全站在同一起跑线上，不对任何单一模型施加额外校准。若 checkpoints 已存在，用 `src.fair_comparison` 直接重导三模型的 raw test 指标并刷新主表：

```bash
conda run -n qwen3meld-run python -m src.fair_comparison --write_raw_test_metrics
conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer dmsaformer
```

`src.fair_comparison` 还会额外输出一个校准消融：对三个模型施加完全相同的 validation-only 仿射校准 `y = a * pred + b`（a、b 只在 valid 预测上拟合，只应用于 test 预测，绝不接触 test 标签），结果写入 `results/metrics/fair_comparison_summary.csv`。该消融表明校准对三个模型都能小幅降低误差，但不改变模型间的相对排名。历史脚本 `src.calibrated_dmsaformer` 仅对 DMSAFormer 单独校准，会造成对比不对称，已不再用于生成主表：

DMSAFormer 单次实验示例：

```bash
python -m src.train --model dmsaformer --output_len 90 --seed 2026 --epochs 30 --device auto
python -m src.evaluate --checkpoint checkpoints/dmsaformer_90_seed2026.pt --model dmsaformer --output_len 90 --seed 2026
```

兼容目标说明中的命令名：

```bash
python train_dmsaformer.py --pred_len 90 --seed 2026
python train_dmsaformer.py --pred_len 365 --seed 2026
python run_experiments.py
```

## 结果说明

汇总命令：

```bash
python -m src.summarize_results --models lstm transformer dmsaformer
```

输出：

- `results/metrics/summary.csv`
- `results/metrics/summary.md`
- `results/figures/metric_bar_mse.png`
- `results/figures/metric_bar_mae.png`
- `results/figures/prediction_comparison_90.png`
- `results/figures/prediction_comparison_365.png`
- `results/screenshots/`

最终报告使用显式三模型过滤生成，`summary.csv` 包含 LSTM、Transformer、DMSAFormer 三个正式模型在两个预测长度上的 MSE/MAE mean/std。若需要查看 HybridTCNTransformer 的内部消融结果，可从 `results/metrics/hybrid_*_test_metrics.csv` 或另行运行 `python -m src.summarize_results --models lstm transformer hybrid dmsaformer` 查看，但不要把它放入正式主表。
