# GitHub Release Checklist

## 当前状态

- 代码结构、训练评估链路、真实 UCI 数据预处理、30 次 goal-1 full 实验、图表生成和 PDF 草稿已完成。
- `results/metrics/summary.csv` 已按 goal-1 三模型范围生成：`hybrid/lstm/transformer` x `90/365` 共 6 行，每个模型/预测长度组合均为 `Runs=5`。
- 发布前仍需填写真实 GitHub 链接、作者贡献和研究领域。

## 发布前检查

- [ ] 填写 `submission_info_template.md`：GitHub 仓库 URL、作者姓名、研究领域、贡献，以及是否提交 DMSAFormer 扩展和课程 PDF 原件。
- [x] 将真实 `train.csv`、`tes.csv`/`test.csv` 或 UCI 原始数据放入 `data/raw/`。
- [x] 运行 `python -m src.data_preprocess --data_dir data/raw --output_dir data/processed`。
- [x] 运行 `bash scripts/run_all_experiments.sh`，完成 3 个模型 x 2 个预测长度 x 5 个 seed。
- [x] 如果 GPU 被占用，运行 `scripts/watch_gpu_and_run_full_experiments.sh` 或 README 中的 tmux watcher 命令，等待 GPU 空闲后自动启动 full 实验。
- [x] 确认 `results/metrics/summary.csv` 每个 goal-1 三模型组合 `Runs=5`。
- [x] 确认 `results/figures/metric_bar_mse.png` 和 `metric_bar_mae.png` 已更新。
- [x] 确认 `results/figures/prediction_comparison_90.png` 和 `prediction_comparison_365.png` 已更新。
- [x] 更新 `report/report_draft.md`，去掉 smoke 说明并写入真实实验分析。
- [x] 重新生成 `report/report.pdf` 和 `report/ML_household_power_report.pdf`。
- [ ] 在 README 和报告中填入真实 GitHub 链接。
- [ ] 在报告中填入作者贡献和研究领域。
- [ ] 配置 GitHub remote，并在用户确认后 commit/push。
- [x] 运行最终测试：`conda run -n qwen3meld-run python -m pytest tests -q`，结果 `19 passed`。
- [x] 新增并运行提交前检查脚本：`PYTHON="conda run -n qwen3meld-run python" ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1 bash scripts/check_submission_ready.sh` 通过；默认模式会因 GitHub/作者占位符保留而失败。
- [x] 默认提交前检查失败时会列出占位符文件和行号，便于替换真实 GitHub 链接、作者贡献和研究领域。
- [x] 若 DMSAFormer 流程覆盖了 `summary.csv`，发布前需重新运行 `conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer hybrid` 并重新生成 PDF。
- [x] 确认参考文献真实可查。
- [x] 确认报告注明 ChatGPT/DeepSeek 等工具使用情况。

## Git 暂存审计

- [x] 已用 `git check-ignore -v` 核对课程 PDF 原件、raw data、processed `.npz`、`scaler.pkl`、checkpoints、logs、prediction CSV 均会被 `.gitignore` 排除。
- [x] 已补强 `.gitignore`，避免嵌套 smoke 目录中的 `predictions/`、`checkpoints/` 和 `results/goal2_smoke/`、`results/goal2_v2_smoke/` 被误提交。
- [ ] 不要直接执行裸 `git add .` 后提交；提交前需由用户决定 DMSAFormer/goal-2 相关产物是否进入仓库，并先用 `git add -n .` 预演。
- [ ] 严格 goal-1 提交建议先使用 `git add -n <paths>` 预演，再去掉不希望提交的文件后执行真实 `git add`。
- [ ] 当前代码、测试和 README 已集成 DMSAFormer 作为可选扩展模型；推荐提交策略是保留 DMSA 代码入口，但课程主报告和默认 `summary.csv` 仍按 LSTM/Transformer/Hybrid 三模型生成。
- [ ] 若用户要求纯 goal-1 仓库，需要另做一次回退清理：移除 DMSA 模型、DMSA 脚本、DMSA 测试、DMSA 结果与 `goal/goal-2/`，再重新跑测试和生成报告。
- [ ] 课程 PDF 原件 `2026-专硕机器学习课程考核.pdf` 默认已 ignore；若用户明确要求提交，需要使用 `git add -f 2026-专硕机器学习课程考核.pdf`。

## 不应提交

- [x] `data/raw/`
- [x] `2026-专硕机器学习课程考核.pdf`
- [x] `data/processed/*.npz`
- [x] `data/processed/scaler.pkl`
- [x] `checkpoints/`
- [x] `results/**/checkpoints/`
- [x] `*.pt`
- [x] `*.pth`
- [x] `__pycache__/`
- [x] `.pytest_cache/`
- [x] `logs/`
- [x] `results/predictions/`
- [x] `results/**/predictions/`
- [x] `results/goal2_smoke/`
- [x] `results/goal2_v2_smoke/`

## 可提交核心文件

- [x] `src/data_preprocess.py`
- [x] `src/dataset.py`
- [x] `src/models/lstm.py`
- [x] `src/models/transformer.py`
- [x] `src/models/hybrid_model.py`
- [x] `src/train.py`
- [x] `src/evaluate.py`
- [x] `src/summarize_results.py`
- [x] `src/generate_report_pdf.py`
- [x] `scripts/check_submission_ready.sh`
- [x] `scripts/run_all_experiments.sh`
- [x] `submission_info_template.md`
- [x] `README.md`
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `report/report_draft.md`
- [x] `report/report.pdf`
- [x] `report/ML_household_power_report.pdf`
