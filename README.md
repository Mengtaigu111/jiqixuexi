# 基于深度学习的家庭电力消耗多变量时间序列预测

本项目用于 2026 年专硕机器学习课程项目：使用家庭电力消耗多变量时间序列，根据过去 90 天数据分别预测未来 90 天和 365 天 `global_active_power` 曲线。最终报告比较 LSTM、Transformer、HybridTCNTransformer 与 validation-calibrated DMSAFormer 四种模型，共 2 个预测长度 x 5 个 seed x 4 个模型 = 40 次训练与评估。

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

最终报告支持模型：

- `lstm`
- `transformer`
- `hybrid`
- `dmsaformer`

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

baseline 完整实验共 3 个模型 x 2 个预测长度 x 5 个 seed：

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
conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer hybrid
```

运行 DMSAFormer 并重新生成四模型最终结果：

```bash
PYTHON="conda run -n qwen3meld-run python" EPOCHS=30 SEEDS="2026 2027 2028 2029 2030" BATCH_SIZE=32 bash scripts/run_dmsaformer_experiments.sh
```

当前最终对比表使用 validation-calibrated DMSAFormer：90 天用验证集稳定性门控 Hybrid/Transformer 专家，365 天用验证集 affine 校准后的 LSTM 专家。若 baseline checkpoints 已存在，可直接重新导出最终最优 DMSAFormer 结果：

```bash
conda run -n qwen3meld-run python -m src.calibrated_dmsaformer
conda run -n qwen3meld-run python -m src.summarize_results
conda run -n qwen3meld-run python -m src.export_dmsaformer_artifacts
```

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
python -m src.summarize_results
```

输出：

- `results/metrics/summary.csv`
- `results/metrics/summary.md`
- `results/figures/metric_bar_mse.png`
- `results/figures/metric_bar_mae.png`
- `results/figures/prediction_comparison_90.png`
- `results/figures/prediction_comparison_365.png`
- `results/screenshots/`

最终报告使用不带 `--models` 的汇总结果，`summary.csv` 包含 LSTM、Transformer、HybridTCNTransformer、DMSAFormer 四个模型在两个预测长度上的 MSE/MAE mean/std。若只需要三模型 baseline 汇总，可显式使用 `--models lstm transformer hybrid`。
