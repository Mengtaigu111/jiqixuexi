# Project Checklist

## 课程要求逐项检查

- [x] 完成 LSTM：`src/models/lstm.py`
- [x] 完成 Transformer：`src/models/transformer.py`
- [x] 完成改进模型 HybridTCNTransformer：`src/models/hybrid_model.py`
- [x] 完成 90 天预测代码路径：`output_len=90`
- [x] 完成 365 天预测代码路径：`output_len=365`
- [x] 每组至少 5 个 seed：goal-1 三模型真实数据 full 实验已完成，6 个模型/预测长度组合均为 `Runs=5`。
- [x] 计算 MSE 和 MAE：`src/utils.py`、`src/evaluate.py`
- [x] 计算 mean 和 std：`src/summarize_results.py`
- [x] 生成正式预测曲线和 Ground Truth 对比图：`results/figures/*_curve.png`
- [x] 生成正式结果截图素材：`results/screenshots/`
- [x] 生成 PDF 报告草稿：`report/report.pdf` 与 `report/ML_household_power_report.pdf`
- [x] PDF 包含 GitHub 链接占位符；正式提交前需替换为真实仓库 URL。
- [x] PDF 包含四部分：问题介绍、模型、结果与分析、讨论。
- [x] README 可指导他人运行代码。
- [x] 本地 git 仓库已初始化，默认分支为 `main`。
- [ ] GitHub 远程仓库尚未创建，当前仅准备本地可提交文件。

## 当前验证与运行状态

- [x] `conda run -n qwen3meld-run python -m pytest tests -q` 通过：19 passed。
- [x] `python -m src.data_preprocess --generate_smoke_data` 生成 90/365 训练和测试窗口。
- [x] `PYTHON="conda run -n qwen3meld-run python" EPOCHS=1 SEEDS=2026 BATCH_SIZE=64 DEVICE=auto bash scripts/run_all_experiments.sh` 跑通 6 个组合。
- [x] 真实 UCI 数据已下载并预处理完成。
- [x] smoke 输出已清理，避免混入正式结果。
- [x] tmux watcher `jiqixuexi_gpu_watch` 已等待 GPU 空闲并完成 full 实验，完成后已停止。
- [x] `results/metrics/summary.csv` 已按 goal-1 三模型范围生成，包含 `hybrid/lstm/transformer` 共 6 行，且每行 `Runs=5`。
- [x] `results/figures/metric_bar_mse.png` 已生成。
- [x] `results/figures/metric_bar_mae.png` 已生成。
- [x] `results/figures/prediction_comparison_90.png` 已生成。
- [x] `results/figures/prediction_comparison_365.png` 已生成。
- [x] `results/EXPERIMENT_STATUS.md` 已标注真实数据与正式实验状态。
- [x] PDF 已重新生成并核验为 10 页；`report/report.pdf` 与 `report/ML_household_power_report.pdf` 均存在。
- [x] 关键图像文件已核验为非空白。
- [x] `scripts/check_submission_ready.sh` 可检查三模型 summary、PDF 页数、关键图片、ignore 规则、占位符和 pytest；当前默认模式因 GitHub/作者占位符未填写而失败并列出文件/行号，`ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1` 预检查通过。

## 正式提交前必须完成

- [x] 获取并放入真实课程数据。
- [x] 删除或覆盖 smoke 结果。
- [x] 用真实数据完成 30 次 full 实验。
- [x] 检查 `summary.csv` 每个组合 `Runs=5`。
- [x] 更新报告中的结果表和分析文字。
- [ ] 填写真实 GitHub 链接、作者贡献和研究领域。
- [ ] 配置 GitHub remote，并在用户确认后 commit/push。
