# Project Checklist

## 课程要求逐项检查

- [x] 完成 LSTM：`src/models/lstm.py`
- [x] 完成 Transformer：`src/models/transformer.py`
- [x] 完成最终改进模型 DMSAFormer：`src/models/dmsaformer.py`；HybridTCNTransformer 仅作为中间改进/消融代码保留。
- [x] 完成 90 天预测代码路径：`output_len=90`
- [x] 完成 365 天预测代码路径：`output_len=365`
- [x] 每组至少 5 个 seed：正式三模型 `lstm/transformer/dmsaformer` 在 2 个预测长度上均为 `Runs=5`。
- [x] 计算 MSE 和 MAE：`src/utils.py`、`src/evaluate.py`
- [x] 计算 mean 和 std：`src/summarize_results.py`
- [x] 生成正式预测曲线和 Ground Truth 对比图：`results/figures/*_curve.png`
- [x] 生成正式结果截图素材：`results/screenshots/`
- [x] 生成 PDF 报告草稿：`report/report.pdf` 与 `report/ML_household_power_report.pdf`
- [x] PDF 包含 GitHub 链接占位符；正式提交前需替换为真实仓库 URL。
- [x] PDF 包含四部分：问题介绍、模型、结果与分析、讨论。
- [x] README 可指导他人运行代码。
- [x] 本地 git 仓库已初始化，默认分支为 `main`。
- [x] GitHub 远程仓库已配置：`https://github.com/Mengtaigu111/jiqixuexi.git`。

## 当前验证与运行状态

- [x] `PYTHON='/home/myluo/miniconda3/bin/conda run -n qwen3meld-run python' bash scripts/check_submission_ready.sh` 通过，内部 `pytest` 结果为 `25 passed`。
- [x] `python -m src.data_preprocess --generate_smoke_data` 生成 90/365 训练和测试窗口。
- [x] `PYTHON="conda run -n qwen3meld-run python" EPOCHS=1 SEEDS=2026 BATCH_SIZE=64 DEVICE=auto bash scripts/run_all_experiments.sh` 跑通 LSTM/Transformer/Hybrid 基础与中间模型 smoke 组合；正式三模型汇总另含 DMSAFormer。
- [x] 真实 UCI 数据已下载并预处理完成。
- [x] smoke 输出已清理，避免混入正式结果。
- [x] tmux watcher `jiqixuexi_gpu_watch` 已等待 GPU 空闲并完成 full 实验，完成后已停止。
- [x] `results/metrics/summary.csv` 已按正式三模型生成，包含 `dmsaformer/lstm/transformer` 共 6 行，且每行 `Runs=5`。
- [x] `results/figures/metric_bar_mse.png` 已生成。
- [x] `results/figures/metric_bar_mae.png` 已生成。
- [x] `results/figures/prediction_comparison_90.png` 已生成。
- [x] `results/figures/prediction_comparison_365.png` 已生成。
- [x] `results/EXPERIMENT_STATUS.md` 已标注真实数据与正式实验状态。
- [x] PDF 已用 HTML/CSS + Playwright 链路重新生成并核验为 12 页；`report/report.pdf` 与 `report/ML_household_power_report.pdf` 均存在。
- [x] 关键图像文件已核验为非空白。
- [x] `scripts/check_submission_ready.sh` 可检查三模型 summary、PDF 页数、关键图片、ignore 规则、占位符和 pytest。

## 正式提交前必须完成

- [x] 获取并放入真实课程数据。
- [x] 删除或覆盖 smoke 结果。
- [x] 用真实数据完成正式三模型 30 次训练/评估汇总；Hybrid 结果作为内部消融记录保留。
- [x] 检查 `summary.csv` 每个组合 `Runs=5`。
- [x] 更新报告中的结果表和分析文字。
- [ ] 填写真实 GitHub 链接、作者贡献和研究领域。
- [ ] 在用户确认后 commit/push 当前未提交改动。

## 公平对比口径修正与诊断分析（2026-07-06）

- [x] 修正主表口径不对称问题：原主表用「校准后 DMSAFormer」对比「未校准 baseline」，已改为三模型统一原始（raw、未校准）test 指标。`src/fair_comparison.py` 对三模型施加同一套 validation-only 仿射校准并输出 raw 与 calibrated 两套指标作为消融。
- [x] 诚实结论：公平对比下 DMSAFormer 两个 horizon 均未超过最强 baseline。90 天最优为 Transformer（DMSAFormer 排第三，MSE 高 6.02%）；365 天最优为 LSTM（DMSAFormer 排第二，MSE 高 9.21%）。
- [x] 诊断分析（原因分析，`src/diagnostics.py`、`src/ablation_dmsaformer.py`、`src/diagnostics_figures.py`）：
  - 朴素基线地板：`results/metrics/naive_floor_summary.csv`（持续法/窗口均值/周季节 naive）。
  - 参数量代价：`results/metrics/capacity_summary.csv`，DMSAFormer 参数量为 baseline 的 4–6 倍却更差。
  - 过拟合缺口：`results/metrics/overfitting_gap_summary.csv`，365 天 DMSAFormer valid/train 比值 2.53 为三者最高。
  - 逐模块消融：`results/metrics/ablation_dmsaformer_summary.csv`，当前架构从头重训、5 seed；消融数值只用于模块间相对比较，不替代主表。
  - 诊断图：`results/figures/diag_naive_floor.png`、`diag_capacity_vs_error.png`、`diag_overfitting_gap.png`、`diag_ablation.png`。
- [x] 报告正文新增 3.4/3.5 诊断章节与 3.6 诊断图表页，PDF 已重新生成为 14 页（`report/ML_household_power_report.pdf`）。
- [x] 复现脚本 `scripts/run_dmsaformer_experiments.sh` 改用 `src.fair_comparison` 写 raw 主表，不再用只校准 DMSAFormer 的旧流程。
- [x] 过时的 Word 报告 (`report/*.docx`) 已删除，避免与诚实版 PDF 口径矛盾；正式交付只用 PDF。
