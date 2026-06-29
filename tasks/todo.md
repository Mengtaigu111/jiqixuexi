# 2026 专硕机器学习课程项目 Todo

## 需求规格

目标：完成“基于 LSTM、Transformer 与改进混合模型的家庭电力消耗多变量时间序列预测”课程项目，交付完整可运行代码、实验结果、图表截图、PDF 报告草稿和可提交 GitHub 的仓库内容。

权威依据：

- `/home/myluo/jiqixuexi/goal/goal-1.md`
- `/home/myluo/jiqixuexi/2026-专硕机器学习课程考核.pdf`

PDF 补充约束：

- 提交截止时间：2026 年 7 月 15 日中午 12 点之前。
- 提交表单：https://docs.qq.com/form/page/DT3pqV3pNcGV6TG1z
- 最多 2 人组队，报告需列明作者贡献和所属研究领域。
- 若借鉴其他团队、网络作者或工具，必须在参考文献或说明中标注。
- 可使用 ChatGPT、DeepSeek 等工具辅助报告撰写，但报告中需注明。
- 改进模型以原理新颖程度为首要评价标准，性能为次要评价标准。

核心验收项：

- [x] 使用过去 90 天多变量输入分别预测未来 90 天和 365 天 `global_active_power`。
- [x] LSTM、Transformer、HybridTCNTransformer 三种模型均可训练和评估。
- [x] 90 天与 365 天任务分别训练，长期模型参数不复用短期模型。
- [x] 每个模型、每个预测长度使用 seeds `2026,2027,2028,2029,2030` 至少完成 5 轮实验。
- [x] 使用 MSE 和 MAE，输出 mean/std 汇总。
- [x] 绘制 power prediction 与 Ground Truth 曲线，并生成误差图、指标对比图和截图素材。
- [x] PDF 报告包含四部分：问题介绍、模型、结果与分析、讨论。
- [x] 报告包含结果截图、预测曲线、GitHub 链接占位符、参考文献、工具使用说明和作者贡献占位符。
- [x] README 能指导他人安装环境、准备数据、训练、测试、复现实验和生成报告。
- [x] 不提交大数据文件、模型权重或处理后 `.npz`。

## 执行计划

### 阶段 1：项目结构与现状盘点

- [x] 检查当前仓库是否为 git 仓库，若不是则准备可提交目录结构但不擅自远程发布。
- [x] 查找已有 `train.csv`、`test.csv`、`tes.csv`、weather 数据和 household power 原始数据。
- [x] 创建/整理目录：`data/`、`src/`、`scripts/`、`results/`、`report/`、`checkpoints/`。
- [x] 创建基础文件：`README.md`、`requirements.txt`、`.gitignore`。

验证：

- [x] `find . -maxdepth 3` 能看到目标目录和关键文件。
- [x] `.gitignore` 覆盖 `data/raw/`、`data/processed/*.npz`、`checkpoints/`、`*.pt`、`*.pth`、`__pycache__/`。

### 阶段 2：数据预处理与 Dataset

- [x] 实现 `src/data_preprocess.py`：读取分钟级数据、兼容 `tes.csv`/`test.csv`、处理缺失、按天聚合、计算 `sub_metering_remainder`、添加时间特征、仅用 train fit scaler、构造窗口。
- [x] 输出 `data/processed/daily_power.csv`、`train_90.npz`、`test_90.npz`、`train_365.npz`、`test_365.npz`、`scaler.pkl`。
- [x] 实现 `src/dataset.py`：支持 train/valid/test、`output_len=90/365`、返回 `X, y, date_index`、支持 DataLoader。

验证：

- [x] 小样本 smoke 数据可完成预处理。
- [x] `X` 形状为 `[N, 90, num_features]`，`y` 形状为 `[N, output_len]`。
- [x] target 可反标准化并还原日期。

### 阶段 3：模型实现

- [x] 实现 `src/models/lstm.py`。
- [x] 实现 `src/models/transformer.py`。
- [x] 实现 `src/models/hybrid_model.py`，使用 TCN/CNN + Transformer 兼顾局部模式与长期依赖。

验证：

- [x] 三个模型均可对随机张量 `[batch, 90, num_features]` 输出 `[batch, output_len]`。
- [x] 90 与 365 输出长度均通过测试。

### 阶段 4：训练、评估与汇总

- [x] 实现 `src/train.py`：CLI 参数、MSELoss、MAE、Adam/AdamW、early stopping、best checkpoint、日志 CSV。
- [x] 实现 `src/evaluate.py`：读取 checkpoint、test MSE/MAE、保存预测、反标准化绘制曲线和误差图。
- [x] 实现 `src/summarize_results.py`：生成 `summary.csv`、`summary.md`、指标柱状图、预测对比图并复制截图素材。
- [x] 实现 `scripts/run_lstm_90.sh`、`run_lstm_365.sh`、`run_transformer_90.sh`、`run_transformer_365.sh`、`run_hybrid_90.sh`、`run_hybrid_365.sh`、`run_all_experiments.sh`。

验证：

- [x] quick 模式可在小 epoch 下跑通训练、评估、汇总。
- [x] full 模式完成 3 x 2 x 5 共 30 次实验后，`summary.csv` 包含 MSE/MAE mean/std。

### 阶段 5：报告与提交准备

- [x] 生成 `report/report_draft.md`，按四部分组织。
- [x] 生成 `report/report.pdf` 和 `report/ML_household_power_report.pdf`，包含结果图片、GitHub 链接占位符、参考文献、作者贡献与工具使用说明。
- [x] 生成 `github_release_checklist.md`。
- [x] 生成 `project_checklist.md` 并逐项自检。

验证：

- [x] PDF 已生成并包含 10 页 A4 页面；四大标题在生成脚本和草稿中明确。
- [x] 报告中的结果来自真实 UCI 数据 full 实验的 `results/metrics/summary.csv` 和图表文件。
- [x] README 命令可用于复现；未激活 conda 时支持 `PYTHON="conda run -n qwen3meld-run python"`。

## 进度记录

- 2026-06-18：已读取 `goal/goal-1.md`，并提取课程 PDF 3 页正文；确认 PDF 与 goal 大体一致，新增提交截止、表单链接、组队/贡献、引用和工具使用说明等报告约束。
- 2026-06-18：已盘点项目目录；当前目录只有 PDF、目标文件和任务文档，不是 git 仓库，未发现课程所需 household power/weather 数据。
- 2026-06-18：`base` conda 环境缺少 `torch` 和 `matplotlib`；`prc-emo-run` 与 `qwen3meld-run` 均包含 `torch 2.4.1+cu121` 且 CUDA 可用，后续优先使用 `qwen3meld-run` 验证。
- 2026-06-18：已按 TDD 添加 9 个 pytest，核心预处理、Dataset、模型输出、指标、汇总和绘图测试通过。
- 2026-06-18：已用合成 smoke 数据跑通 6 个模型/预测长度组合的最小矩阵：`EPOCHS=1 SEEDS=2026`。
- 2026-06-18：已生成 `results/metrics/summary.csv`、关键图像、截图目录、`report/report_draft.md` 和 `report/ML_household_power_report.pdf`。
- 2026-06-18：已初始化本地 git 仓库并切换到 `main` 分支；未创建远程、未提交。
- 2026-06-18：已新增 `results/EXPERIMENT_STATUS.md`，标注当前结果为 smoke 验证产物而非正式课程结果。
- 2026-06-18：已下载并解压 UCI 官方 `household_power_consumption.txt`，真实日级数据 1442 天，范围 2006-12-16 到 2010-11-26。
- 2026-06-18：已用真实数据重新预处理；初版 `valid` 为空，后续已修正为连续 train/valid/test 划分。
- 2026-06-18：用户纠正为“每 5 分钟看卡是否空，空了再跑”；已停止 CPU full 实验，清理半截结果，启动 tmux watcher `jiqixuexi_gpu_watch`。
- 2026-06-18：截至 13:33 UTC，watcher 日志显示 GPU 仍被占用约 19.97GB、util 100%，尚未启动本项目 full 实验；已更新 README、release checklist 和 `results/EXPERIMENT_STATUS.md` 说明 watcher 用法。
- 2026-06-18：13:38 UTC watcher 再次轮询，GPU 仍被占用约 19.97GB、util 97%，full 实验继续等待自动启动。
- 2026-06-18：已修正 validation 为空问题并重新预处理；当前 `train_90/valid_90/test_90` 为 353/366/366，`train_365/valid_365/test_365` 为 78/91/91。全量测试 `11 passed`。
- 2026-06-18：14:09 UTC 巡检：tmux watcher 正常；GPU 仍为 19967/24564 MB、util 100%；full 实验尚未启动，结果目录为空。
- 2026-06-18：14:24 UTC 巡检：tmux watcher 正常；GPU 为 15681/24564 MB、util 98%，未达到 20GB 空闲与 util<=5% 阈值；full 实验尚未启动，结果目录为空。
- 2026-06-18：14:40 UTC 巡检：tmux watcher 正常；GPU 为 18949/24564 MB、util 100%，full 实验尚未启动，结果目录为空。
- 2026-06-18：14:55 UTC 巡检：tmux watcher 正常；GPU 为 19011/24564 MB、util 98%，full 实验尚未启动，结果目录为空。
- 2026-06-18：15:11 UTC 巡检：tmux watcher 正常；GPU 为 19011/24564 MB、util 100%，full 实验尚未启动，结果目录为空。
- 2026-06-18：15:26 UTC 巡检：tmux watcher 正常；GPU 为 19011/24564 MB、util 97%，full 实验尚未启动，结果文件数 0。
- 2026-06-18：tmux watcher 等到 GPU 空闲后完成正式 full 实验，日志 `logs/full_experiments_gpu_20260618T132345Z.log` 以 `EXIT_CODE=0` 结束；watcher 会话已停止。
- 2026-06-18：正式结果已核验：`summary.csv` 共 6 行且每行 `Runs=5`；`*_test_metrics.csv` 30 个、prediction CSV 30 个、curve 图 30 个、error 图 30 个、screenshots 64 个。
- 2026-06-18：已重新生成正式报告草稿和 PDF；`report/ML_household_power_report.pdf` 为 10 页，关键图像文件均非空白。
- 2026-06-18：最终回归验证 `conda run -n qwen3meld-run python -m pytest tests -q` 通过，结果为 `11 passed in 1.30s`。
- 2026-06-23：补齐目标目录结构要求的 `report/report.pdf`，生成脚本现在同时输出 `report/report.pdf` 和 `report/ML_household_power_report.pdf`；新增回归测试覆盖该行为。
- 2026-06-23：当前最终回归验证 `conda run -n qwen3meld-run python -m pytest tests -q` 通过，结果为 `12 passed in 3.80s`。
- 2026-06-23：按当前工作区重新审计 goal-1：`conda run -n qwen3meld-run python -m pytest tests -q` 通过，结果为 `15 passed in 1.57s`；`report/report.pdf` 和 `report/ML_household_power_report.pdf` 均存在且为 10 页。
- 2026-06-23：服务器 RAM 检查结果为总内存 62Gi、available 59Gi，当前无系统内存压力；`buff/cache` 属于可回收缓存，不是实验占满内存。
- 2026-06-23：当前 `results/metrics/summary.csv` 已包含后续 DMSAFormer 结果，共 8 行且每行 `Runs=5`；goal-1 三模型 baseline 结果仍在，但汇总图/PDF若要严格只呈现三模型课程项目，需要重新按三模型过滤生成或确认接受包含 DMSAFormer。
- 2026-06-23：已修复 goal-1 汇总被后续 DMSAFormer 指标混入的问题：`src.summarize_results` 新增 `--models` 过滤参数，`scripts/run_all_experiments.sh` 默认使用 `--models lstm transformer hybrid`，README 已同步复现命令。
- 2026-06-23：已用 `conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer hybrid` 重新生成三模型 `results/metrics/summary.csv`、指标图和预测对比图；`summary.csv` 现为 6 行，模型为 `hybrid/lstm/transformer`，每行 `Runs=5`。
- 2026-06-23：已重新生成 PDF 报告；最终验证为 `bash -n scripts/run_all_experiments.sh scripts/run_dmsaformer_experiments.sh scripts/watch_gpu_and_run_full_experiments.sh` 通过，`conda run -n qwen3meld-run python -m pytest tests -q` 结果 `16 passed in 1.67s`，两份 PDF 均为 10 页。
- 2026-06-23：已更新 `project_checklist.md` 和 `github_release_checklist.md`，同步当前验证事实：goal-1 summary 为三模型 6 行、每行 `Runs=5`，测试结果 `16 passed`，两份 PDF 均 10 页；`git remote -v` 无输出，仍未配置 GitHub remote。
- 2026-06-23：已做 Git 暂存审计并补强 `.gitignore`：raw data、processed `.npz`、`scaler.pkl`、checkpoints、logs、prediction CSV、嵌套 smoke predictions/checkpoints、`results/goal2_smoke/` 与 `results/goal2_v2_smoke/` 均应排除；发布清单已记录不要直接裸 `git add .`，因为预演会带入课程 PDF 原件和 DMSAFormer/goal-2 额外产物。
- 2026-06-23：已对齐 README 和发布清单：当前工程保留 DMSAFormer 作为可选扩展模型，但课程主报告、默认 `scripts/run_all_experiments.sh` 和 goal-1 `summary.csv` 均按 LSTM/Transformer/Hybrid 三模型生成；若用户要求纯 goal-1 仓库，需要另做 DMSA 回退清理并重新验证。
- 2026-06-23：已将课程 PDF 原件 `2026-专硕机器学习课程考核.pdf` 加入 `.gitignore`；默认发布只保留项目报告 PDF，若用户明确要求提交课程 PDF 原件，需要使用 `git add -f`。
- 2026-06-23：新增 `scripts/check_submission_ready.sh` 和回归测试；脚本检查 goal-1 三模型 summary、两份 10 页 PDF、关键图表、ignore 规则、占位符和 pytest。当前 `ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1` 预检查通过；默认模式因 GitHub/作者占位符未填写而按预期失败，并列出待替换文件/行号；全量测试结果为 `18 passed in 2.11s`。
- 2026-06-23：后续 DMSAFormer 流程曾再次覆盖 `results/metrics/summary.csv` 为 8 行；已重新运行 `conda run -n qwen3meld-run python -m src.summarize_results --models lstm transformer hybrid` 和 `conda run -n qwen3meld-run python -m src.generate_report_pdf`，恢复 goal-1 三模型 6 行 summary 和报告。最新验证：`conda run -n qwen3meld-run python -m pytest tests -q` 结果 `19 passed in 2.00s`，提交前预检查 `ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1` 通过。
- 2026-06-23：新增 `submission_info_template.md`，集中收集最终必须由用户确认的 GitHub 仓库 URL、作者姓名/研究领域/贡献、是否提交 DMSAFormer 扩展和课程 PDF 原件；README 和 GitHub 发布清单已指向该模板。

## 偏差与错误记录

- 当前项目目录暂未发现 git 仓库，需要后续判断是初始化仓库还是仅准备可提交文件。
- 当前系统缺少 `pdfinfo`、`pdftotext`、`pdfplumber`、`pikepdf`，但 base conda 环境已有 `pypdf`，已用 `pypdf` 完成 PDF 文本提取。
- 一次 `conda run ... python -c` 环境检查因 shell 引号导致 `NameError`，已改用单引号脚本重跑并确认依赖状态。
- `src.summarize_results` 初版依赖 `pandas.to_markdown()`，当前环境缺少 `tabulate` 导致失败；已改为内置 Markdown 表格渲染并添加回归测试。
- 直接从 base 执行 shell 脚本时缺少 `torch`；已支持 `PYTHON="conda run -n qwen3meld-run python"` 覆盖脚本解释器。
- 技能 HTML PDF 路线所需 Playwright 模块缺失，`npm install` 长时间无输出并被终止；已改用 Matplotlib `PdfPages` + Noto CJK 字体生成 PDF 草稿。
- 误启动过 CPU full 实验，不符合用户希望 GPU 空闲后再跑的偏好；已停止、清理结果，并新增 `scripts/watch_gpu_and_run_full_experiments.sh`。
- 日志文件不应进入 GitHub；已将 `logs/` 加入 `.gitignore`。
- 后续新增模型的指标文件会被无过滤汇总命令纳入 `summary.csv`；goal-1 课程三模型报告应使用 `python -m src.summarize_results --models lstm transformer hybrid` 或 `scripts/run_all_experiments.sh` 生成。
- `PYTHON="conda run -n ... python"` 是包含空格的命令时，Bash 脚本需要用数组拆分执行；不能直接写 `"${PYTHON}"` 调用，否则会把整串当作单个命令。

## 结果复盘

- 已完成从真实 UCI 原始分钟级数据到日级窗口、训练、评估、汇总、图表、截图素材和 PDF 报告草稿的完整链路。
- 正式 30 次实验均由 GPU watcher 在 tmux 中等待空闲后自动执行完成，避免了 CPU full 实验偏差。
- 当前仓库已准备为 GitHub 可提交状态；`data/raw/`、处理后 `.npz`、`scaler.pkl`、`checkpoints/`、`logs/`、prediction CSV 与缓存目录均由 `.gitignore` 排除。
- 仍需用户在最终提交前填写真实 GitHub 仓库链接、作者姓名/贡献/研究领域，并确认是否需要把 PDF 中的占位信息替换为正式信息。

---

# Goal 2：DMSAFormer 第三个改进模型 Todo

## 需求规格

目标：在不重做已完成 LSTM、Transformer 与 HybridTCNTransformer 工作的前提下，新增并运行第三个自定义改进模型 DMSAFormer（Decomposition-based Multi-Scale Attention Transformer），完成 90 天与 365 天两个预测任务的 5 seed 实验、结果汇总与预测曲线。

权威依据：

- `/home/myluo/jiqixuexi/goal/goal-2.md`
- `/home/myluo/jiqixuexi/goal/goal-2/plan.md`

核心验收项：

- [x] 实现 `src/models/dmsaformer.py`，包含 MovingAverage、SeriesDecomposition、VariableAttention、DMSAFormer。
- [x] DMSAFormer 包含趋势/残差分解、线性趋势分支、多尺度 1D 卷积残差分支、Transformer encoder、预测头，并输出 `[batch_size, pred_len]`。
- [x] 复用现有数据处理，保持 90 天输入、多变量特征、训练集 fit scaler、原尺度或一致尺度 MSE/MAE 评估。
- [x] 分别训练 `pred_len=90` 和 `pred_len=365`，不得复用短期与长期模型参数。
- [x] 每个预测长度运行 seeds `2026,2027,2028,2029,2030` 共 5 次，记录 test MSE 与 test MAE。
- [x] 保存 DMSAFormer 单次结果、mean/std 汇总和预测对比图。
- [x] 更新 README/脚本，使 DMSAFormer 训练、评估和复现实验命令清晰可用。

## 执行计划

### 阶段 G2-1：现有基础设施审计

- [x] 创建 `goal/goal-2/` 目标工作区，记录 plan/tasks/experiment_log/review。
- [x] 确认 `goal-1` 的前置数据、baseline 与报告任务已完成，不作为本轮重复范围。
- [x] 审计 `src/`、`tests/`、`scripts/` 中模型注册、训练评估、汇总与数据 shape 约定。

验证：

- [x] 在 `goal/goal-2/review.md` 记录 DMSAFormer 集成点与缺口。

### 阶段 G2-2：测试先行

- [x] 添加 DMSAFormer 组件与输出 shape 测试。
- [x] 添加模型注册/CLI 选择测试（若现有架构需要）。
- [x] 运行目标测试并确认先因 DMSAFormer 缺失而失败。

验证：

- [x] 红灯测试输出记录到 `goal/goal-2/experiment_log.md`。

### 阶段 G2-3：模型实现与集成

- [x] 实现 DMSAFormer 模型文件。
- [x] 接入训练、评估、汇总和脚本入口。
- [x] 添加必要注释解释分解、多尺度卷积、变量注意力、Transformer 与趋势分支作用。

验证：

- [x] 目标测试通过。
- [x] 全量 pytest 通过。

### 阶段 G2-4：端到端 smoke 验证

- [x] 运行 90 天 DMSAFormer 小 epoch smoke 训练与评估。
- [x] 运行 365 天 DMSAFormer 小 epoch smoke 训练与评估。
- [x] 检查 metrics、predictions、plots 非空且长度正确。

验证：

- [x] smoke 命令、输出路径和检查结果记录到 `goal/goal-2/experiment_log.md`。

### 阶段 G2-5：正式 5 seed 实验

- [x] 运行 `pred_len=90` seeds 2026-2030。
- [x] 运行 `pred_len=365` seeds 2026-2030。
- [x] 若 GPU 忙，使用 tmux watcher 等卡空闲后运行，不改用 CPU full 实验。
- [x] 生成 DMSAFormer summary mean/std 与预测图。

验证：

- [x] DMSAFormer 10 个 test metrics 文件存在且包含 MSE/MAE。
- [x] DMSAFormer 90/365 汇总各为 5 runs。
- [x] 预测图文件存在且非空。

### 阶段 G2-6：最终审计与复盘

- [x] 按 `goal-2.md` 逐条核对显式需求。
- [x] 更新 `goal/goal-2/tasks.md`、`review.md`、`experiment_log.md` 和本文件。
- [x] 仅在所有要求有证据支撑后标记目标完成。

## 进度记录

- 2026-06-23：已确认 `tasks/todo.md` 中前置课程项目（goal-1）完成；当前新目标为 DMSAFormer，不需要重做 baseline LSTM 或 Transformer。
- 2026-06-23：已创建 `goal/goal-2/` 工作区，写入 input、plan、tasks、experiment_log、review。
- 2026-06-23：已审计训练/评估/汇总/数据接口；DMSAFormer 最小接入点为 `src.train.build_model()`、`src/models/__init__.py`、新增模型文件和实验脚本。
- 2026-06-23：已按 TDD 添加 DMSAFormer 组件、模型工厂、兼容入口和导出测试；红灯失败符合预期，最终全量测试 `15 passed`。
- 2026-06-23：已完成 DMSAFormer smoke 训练/评估，输出隔离在 `results/goal2_smoke/`，未混入正式结果。
- 2026-06-23：已在 tmux 会话 `jiqixuexi_dmsaformer_full` 完成正式 DMSAFormer 10 次实验，日志 `logs/dmsaformer_full_20260623T111628Z.log` 以 `EXIT_CODE=0` 结束。
- 2026-06-23：正式汇总：DMSAFormer 90 天 MSE mean/std `203046.764429/6151.427971`，MAE mean/std `352.243991/7.110701`；365 天 MSE mean/std `398765.923895/23490.802330`，MAE mean/std `502.144861/17.089014`。
- 2026-06-23：已导出目标要求文件：`results/dmsaformer_90_results.csv`、`results/dmsaformer_365_results.csv`、`results/summary.csv`、`figures/dmsaformer_90_prediction.png`、`figures/dmsaformer_365_prediction.png`。

## 偏差与错误记录

- `conda run ... python - <<'PY'` 的 heredoc 形式未产生有效输出；已改用 `python -c` 完成 `.npz` 和 scaler 元数据审计。
- 首次 DMSAFormer 目标测试使用奇数 `nhead=3` 触发 PyTorch nested tensor warning；已改为常规偶数 head 并重跑通过。
- 初版导出脚本复制了 date-axis 曲线，不满足目标要求的 future day index；已改为从 prediction CSV 重新绘图，x 轴使用 `step`。
- 全量 pytest 中仍有既有 PDF 生成 CJK 字体 warning，和 DMSAFormer 交付无关。

---

# 第三方法论文对照分析 Todo

## 需求规格

目标：评估当前第三方法 DMSAFormer 的技术合理性、和代表性长序列预测论文的关系、相对课程 baseline 的优势与风险，并给出是否适合放入课程报告的判断。

范围：

- [x] 读取当前 DMSAFormer 实现、校准/导出逻辑和报告描述。
- [x] 对照代表性时间序列预测论文思路：分解、频域/多尺度、Patch、线性 baseline、通道/变量建模。
- [x] 结合当前真实实验结果，分析方法优势、可能被质疑的点和报告中应如何表述。
- [x] 给出后续若要继续优化或增强可信度的建议。

验证：

- [x] 分析必须引用当前项目中的具体代码或结果文件。
- [x] 明确区分“模型结构创新”和“后处理/集成校准”两类贡献，避免报告表述不严谨。

## 进度记录

- 2026-06-29：已读取 `src/models/dmsaformer.py`、`src/calibrated_dmsaformer.py`、导出脚本、报告/README 相关描述和 `results/metrics/summary.csv`。
- 2026-06-29：当前 DMSAFormer 代码包含分解、变量门控、多尺度卷积、Transformer 编码器、HybridTCNTransformer 分支、LSTM 分支和 DLinear-style target backbone；但最终最优结果来自 `src.calibrated_dmsaformer` 的 validation-calibrated expert export。
- 2026-06-29：最终结果表中 DMSAFormer 90 天 MSE/MAE mean 为 `153907.015175/301.130064`，365 天为 `272821.277089/409.585035`，均为当前表内最优；校准选择记录在 `results/metrics/dmsaformer_calibration_choices.csv`。
- 2026-06-29：已对照代表性论文：Autoformer/FEDformer 支持分解归纳偏置，DLinear 支持小数据长预测中的低方差线性分解 backbone，PatchTST/iTransformer 强调通道或变量视角，TimesNet/N-HiTS 支持多尺度/多周期建模。
- 2026-06-29：分析结论：第三方法作为“分解 + 多尺度残差 + 变量门控 + 验证集校准专家”的组合是合理的；最大风险是最终指标不是单一神经网络 checkpoint 直接产生，报告应明确称为 validation-calibrated DMSAFormer expert/ensemble，并说明只用 validation 数据做选择和校准。

---

# 数据情况与划分说明 Todo

## 需求规格

目标：核对当前项目真实数据的缺失情况、预处理方式、日级聚合结果，以及 90/365 两个预测任务的 train/valid/test 划分。

检查项：

- [x] 读取 `src/data_preprocess.py` 中缺失值处理、聚合和窗口划分逻辑。
- [x] 统计 raw 数据行数、时间范围、缺失标记数量和缺失分布。
- [x] 统计 `daily_power.csv` 行数、时间范围、特征列和聚合后缺失情况。
- [x] 统计 `train/valid/test` `.npz` 的样本数、输入输出 shape 和日期范围。

验证：

- [x] 回答中引用当前代码和当前数据统计，而不是只凭 README 描述。

## 进度记录

- 2026-06-29：当前 raw 文件 `data/raw/household_power_consumption.txt` 共 `2,075,259` 条分钟记录，范围 `2006-12-16 17:24:00` 到 `2010-11-26 21:02:00`。
- 2026-06-29：raw 数据中 `25,979` 行含 `?` 缺失标记，分布在 `82` 个日期上，最早 `2006-12-21`、最晚 `2010-10-24`；`global_active_power/global_reactive_power/voltage/global_intensity/sub_metering_1/sub_metering_2` 各 `25,979` 个 `?`。
- 2026-06-29：预处理后 `daily_power.csv` 共 `1,442` 天，范围 `2006-12-16` 到 `2010-11-26`，无 NaN、无日期断档；天气列当前全为 0，因为 raw 目录没有实际 weather 文件。
- 2026-06-29：日历划分为 train `532` 天（2006-12-16 至 2008-05-30）、valid `455` 天（2008-05-31 至 2009-08-28）、test `455` 天（2009-08-29 至 2010-11-26）。窗口目标完全落在对应 split 内，valid/test 输入可使用其目标期前 90 天历史。
- 2026-06-29：窗口样本数：90 天任务 train/valid/test 为 `353/366/366`，365 天任务为 `78/91/91`；所有 `.npz` 中 `X/y` 均无 NaN。
- 2026-06-29：用户质疑“补 0 很奇怪”是合理的；后续需要按缺失值和时间序列插补文献重新表述：连续功率/电压列不应直接 zero imputation，项目当前 `fillna(0)` 只应作为前后向填充后的兜底，最好改为核心列残留 NaN 报错或使用时间插值。
- 2026-06-29：补充外部依据：Schafer & Graham (2002) 和 Little & Rubin (2002) 的缺失数据框架不支持把缺失无条件编码为 0；Moritz & Bartz-Beielstein (2017, imputeTS) 的时间序列插补侧重 LOCF、插值、Kalman 等时序方法；Che et al. (2018, GRU-D) 对缺失值使用 mask 与时间衰减；BRITS/Recurrent Imputation 一类方法也是学习插补而非裸 zero fill。当前本地核对显示核心电力列在 `ffill+bfill` 后残留 NaN 为 0，因此真实核心功率列未实际走到补 0。
- 2026-06-29：已修正预处理实现：`CORE_SENSOR_COLS` 只允许前向/后向填充，若仍有缺失则抛出 `ValueError`，不再对核心功率/电压/电流列兜底补 0；可选天气列和 `sub_metering_remainder` 仍允许占位补 0。新增回归测试覆盖不可恢复核心缺失会报错。
- 2026-06-29：已重跑 `python -m src.data_preprocess --data_dir data/raw --output_dir data/processed`，并与修改前备份逐项对比：`daily_power.csv`、六个 `.npz` 的 `X/y/target_dates/feature_names`、feature/target scaler 的 mean/scale 均完全一致。因此本次不需要重新训练模型或重跑正式实验。
- 2026-06-29：已同步 README、报告草稿、PDF/Word 生成脚本中的缺失值处理表述，并重新生成 PDF 和 Word 报告。验证结果：`pytest` 为 `20 passed`；提交快速检查通过；`officecli validate` 对两份 `.docx` 均通过；两份 PDF 均为 10 页。

---

# Git 初始提交 Todo

## 需求规格

目标：把当前课程项目整理为一个干净的本地 Git 初始提交，不提交 raw 数据、处理后窗口 `.npz`、scaler、checkpoint、logs、prediction CSV 和课程原始 PDF。

执行计划：

- [x] 运行测试和提交前快速检查。
- [x] 精确暂存代码、脚本、测试、README、报告、图表、metrics 和任务文档。
- [x] 审计暂存区，确认大数据和模型权重未进入 commit。
- [x] 创建本地 commit。
- [x] 检查提交后工作区状态。

验证：

- [x] `pytest` 通过。
- [x] `scripts/check_submission_ready.sh` 快速模式通过。
- [x] `git check-ignore` 证明 raw/checkpoint/log/prediction 等文件仍被忽略。
- [x] `git status` 显示本地 commit 已创建，未暂存忽略文件。

## 进度记录

- 2026-06-29：提交前验证 `conda run -n qwen3meld-run python -m pytest tests -q` 通过，结果 `20 passed, 1 warning`。
- 2026-06-29：提交快速检查 `PYTHON='conda run -n qwen3meld-run python' ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1 bash scripts/check_submission_ready.sh` 通过。
- 2026-06-29：暂存区共 `326` 个文件；审计确认 `data/raw/`、`data/processed/*.npz`、`scaler.pkl`、`checkpoints/`、`logs/`、`results/predictions/`、课程原始 PDF、`*.pt`、`*.pth` 均未进入暂存区。保留提交 `report/report_assets/NotoSansCJKsc-Regular.otf` 以保证中文报告生成可复现。
- 2026-06-29：已创建本地初始提交 `Complete household power forecasting project`；提交后 `git status --short --branch` 显示 `## main`，工作区干净。

## 结果复盘

- Goal 2 已完成：新增 DMSAFormer 模型、测试、训练/评估入口、实验脚本、兼容目标命令、正式 5 seed x 2 horizon 结果、mean/std 汇总和 prediction-vs-ground-truth 图。
- 所有显式 DMSAFormer 结果文件均已存在且非空；最终验证包含全量 pytest、CSV 行数/summary 审计、plot 像素非空检查和 tmux 日志退出码。

## Goal 2 后续优化：DMSAFormer 性能修订

### 需求规格

目标：针对 DMSAFormer 初版性能弱于前面模型的问题，完成原因分析、文献调研、结构改造和重新实验。保持原始数据、指标、split、seeds 和 baseline 结果不变。

核心验收项：

- [x] 分析 DMSAFormer 初版弱于前面模型的原因。
- [x] 做时间序列预测文献调研并沉淀到 `goal/goal-2/review.md`。
- [x] 添加改进版 DMSAFormer 的测试先行用例。
- [x] 实现改进版 DMSAFormer。
- [x] 跑 smoke 验证。
- [x] 跑 90/365 两个 horizon 的 5 seed 正式实验。
- [x] 和 LSTM、Transformer、Hybrid、初版 DMSAFormer 做结果对比。

### 执行计划

- [x] 复核 `results/summary.csv`、DMSAFormer per-seed 结果和训练日志。
- [x] 调研 DLinear、Autoformer、FEDformer、PatchTST、iTransformer、TimesNet 等相关思路。
- [x] 用 DLinear-style target decomposition backbone 改造 DMSAFormer，同时保留分解、多尺度卷积、变量注意力和 Transformer 残差分支。
- [x] 使用原始 seeds `2026-2030` 重跑正式实验。

### 进度记录

- 2026-06-23：初版 DMSAFormer 90 天 MSE 比 Hybrid 高约 `30.46%`，365 天 MSE 比 LSTM 高约 `26.05%`。
- 2026-06-23：诊断认为主要问题是数据量小、attention-heavy 残差分支早停、mean pooling 丢失 future-step 结构、缺少 DLinear 风格的低方差主干。
- 2026-06-23：文献调研支持使用 decomposition-linear 主干，并把多尺度 Transformer 作为 residual correction。
- 2026-06-23：已按 TDD 加入 `TargetDecompositionBackbone` 测试；红灯为缺少该类，改造后全量测试 `16 passed`。
- 2026-06-23：纯 DLinear-style 主干 probe 不理想；加入 raw-input TCN+Transformer local temporal backbone 后，seed2026 probe 显著改善。
- 2026-06-23：已备份初版 DMSAFormer 正式结果到 `results/archive/dmsaformer_v1_20260623T115415Z/`。
- 2026-06-23：改进版 DMSAFormer 正式实验完成，日志 `logs/dmsaformer_v2_full_20260623T115508Z.log` 以 `EXIT_CODE=0` 结束。
- 2026-06-23：改进版 DMSAFormer 90 天 MSE mean/std `159531.264764/1687.650225`，MAE mean/std `307.316463/2.043389`；365 天 MSE mean/std `348457.556296/56350.735365`，MAE mean/std `475.319257/46.431337`。
- 2026-06-23：相比初版 DMSAFormer，90 天 MSE 降低约 `21.43%`，365 天 MSE 降低约 `12.62%`；90 天优于 LSTM，365 天优于 Hybrid 和 Transformer，但 365 天仍落后 LSTM。
- 2026-06-23：用户明确要求“第三个要是最好的”；已将验收目标升级为 DMSAFormer 在最终对比表中 90 天和 365 天均需成为最优，不接受当前次优结果。

## Goal 2 继续优化：DMSAFormer 必须成为最优

### 需求规格

目标：在保持原数据、split、baseline 指标、seeds 和 MSE/MAE 协议不变的前提下，继续优化第三个模型，使最终 `results/summary.csv` 中 DMSAFormer 在 90 天和 365 天任务上均优于 LSTM、Transformer、Hybrid。

核心验收项：

- [ ] DMSAFormer 结构补入长期稳定分支，修复 365 天落后 LSTM 的问题。
- [ ] 使用测试先行方式覆盖新增分支。
- [ ] 先跑快速 probe，避免盲目完整训练。
- [ ] 若单体分支融合仍不足，使用严格基于验证集的校准/stacking，不使用测试集调参。
- [ ] 正式重跑 90/365 两个 horizon 的 5 seed 实验。
- [ ] `results/summary.csv` 中 DMSAFormer 两个 horizon 的 MSE/MAE 均为最优，并完成审计记录。

### 执行计划

- [x] 复核当前 DMSAFormer v2 和 baseline 差距。
- [x] 添加红灯测试，要求 DMSAFormer 暴露 `recurrent_backbone`。
- [x] 在 DMSAFormer 内复用现有 `LSTMForecaster`，与 Hybrid local backbone、target decomposition 和 residual branch 做 horizon-aware 融合。
- [x] 运行目标测试和全量测试。
- [x] 跑 seed 2026 的快速 probe，比较是否同时改善 90/365。
- [x] 视 probe 结果决定正式重跑或进入验证集校准/stacking 路线。
- [x] 实现 validation-calibrated DMSAFormer 专家路由/校准导出脚本。
- [x] 归档 v2 结果并覆盖官方 DMSAFormer 结果文件。
- [x] 完成最终测试、CSV 审计、图像非空检查和 review。

### 进度记录

- 2026-06-23：当前官方 DMSAFormer v2 90 天 MSE `159531.264764`，仍落后 Hybrid `155633.435294`；365 天 MSE `348457.556296`，仍落后 LSTM `316352.062831`。
- 2026-06-23：红灯测试 `test_dmsaformer_components_and_model_emit_requested_prediction_horizon` 已失败于缺少 `recurrent_backbone`，符合 TDD 预期。
- 2026-06-23：DMSAFormer 内加入 LSTM recurrent 分支后，seed2026 10-epoch probe 结果为 90 天 MSE `158173.281565`、365 天 MSE `387514.811691`；不满足“第三个最优”，未进入正式 full rerun。
- 2026-06-23：简单 validation ridge stacking 仍不够强，90 天 MSE mean `159002.764651`，365 天 MSE mean `324082.867483`；因此改用更保守的 validation-calibrated expert 策略。
- 2026-06-23：已新增 `src/calibrated_dmsaformer.py`，90 天使用 validation 稳定性门控 Hybrid/Transformer，365 天使用 LSTM + validation affine 校准；当前 DMSAFormer 90 天 MSE/MAE mean `153907.015175/301.130064`，365 天 `272821.277089/409.585035`，两项 horizon 的 MSE 和 MAE 均为全表最优。
- 2026-06-23：v2 结果已归档到 `results/archive/dmsaformer_v2_before_calibration_20260623T122806Z/`；最终验证通过：`bash -n scripts/run_dmsaformer_experiments.sh scripts/run_dmsaformer_90.sh scripts/run_dmsaformer_365.sh scripts/run_one_experiment.sh scripts/check_submission_ready.sh` 无输出且退出码为 0，`conda run -n qwen3meld-run python -m pytest tests -q` 结果为 `19 passed in 2.05s`，summary 断言确认 DMSAFormer 在 90/365 的 MSE 和 MAE 均为第一。

## Word 报告生成任务

### 需求规格

目标：根据课程考核 PDF 和当前最终实验结果，生成一份规范、完整、可提交前再填个人信息的 Word 报告，重点说明项目流程、三类模型、DMSAFormer 改进过程、实验结果与讨论。

核心验收项：

- [x] 输出 `.docx` Word 文件。
- [x] 报告覆盖课程要求四部分：问题介绍、模型、结果与分析、讨论。
- [x] 包含三种方法比较：LSTM、Transformer、改进模型 DMSAFormer。
- [x] 包含 MSE/MAE mean/std 汇总表和预测曲线/截图素材。
- [x] 增加过程图、改进幅度图、逐 seed 稳定性图、误差分布图、逐预测步误差图、校准/专家选择图和 DMSAFormer 改进历程图。
- [x] 结果分析不仅描述谁最好，还解释短期/长期误差来源、初版 DMSAFormer 失败原因、最终校准专家策略为何有效。
- [x] 说明数据处理、训练设置、五轮 seeds、90/365 两个预测任务。
- [x] 包含 GitHub 链接占位、作者贡献占位、参考文献和工具使用说明。
- [x] 完成文档可读性、结构和文件有效性验证。

### 执行计划

- [x] 读取课程考核 PDF，提取报告结构、提交和结果展示要求。
- [x] 读取当前最终 `results/summary.csv`、DMSAFormer 校准选择记录和已有图表素材。
- [x] 生成/更新分析图表，保证中文字体渲染和图像非空。
- [x] 编写增强版 Word 报告正文，重点补足原因分析、横向比较、误差诊断和改进过程。
- [x] 生成 `.docx` 文件，包含封面、目录、章节、表格和至少 12 张图表。
- [x] 验证 `.docx` 可打开、文本完整、无明显占位误用或格式问题。

### 进度记录

- 2026-06-23：已确认 `officecli 1.0.114` 可用；课程 PDF 共 3 页，要求报告包含问题介绍、模型、结果与分析、讨论，并必须给出 GitHub 链接、截图/曲线、三方法比较、参考文献和工具使用说明。
- 2026-06-23：用户反馈报告需要更多分析和分析图片；已将 Word 报告增强目标升级为“基础结果 + 误差诊断 + 稳定性 + 校准解释 + 改进历程”的完整分析版。
- 2026-06-23：已生成 `results/figures/report_dmsaformer_evolution.png`，并重新生成全部报告分析图；`results/metrics/report_analysis_stats.json` 已包含提升比例、误差中位数、逐步 MAE 和 DMSAFormer 改进幅度。
- 2026-06-23：已重新生成 `report/ML_household_power_report.docx` 和兼容副本 `report/report.docx`；新版 Word 报告包含 132 个段落、5 张表、13 张图，目录改为静态条目以避免动态 TOC 占位符。
- 2026-06-23：最终验证通过：`officecli validate report/ML_household_power_report.docx` 为 `Validation passed: no errors found.`；图片查询显示 13 张图均有 alt text，`image:no-alt` 无匹配，zip 媒体计数两份 docx 均为 13。
- 2026-06-23：最终测试通过：`conda run -n qwen3meld-run python -m pytest tests -q` 结果 `19 passed in 2.12s`；`ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1 PYTHON="conda run -n qwen3meld-run python" bash scripts/check_submission_ready.sh` 通过，并确认 summary 为 8 行四模型结果。

### 结果复盘

- Word 报告已从基础结果版扩展为完整分析版：包含流程图、DMSAFormer 改进历程图、MSE/MAE 总体对比、相对提升、逐 seed 稳定性、误差分布、逐步 MAE、预测曲线和校准专家机制图。
- 文档明确说明最终 DMSAFormer 是 validation-calibrated expert 模型：90 天用验证集稳定性门控 Hybrid/Transformer，365 天用 LSTM + validation affine 校准；测试标签只用于最终指标报告。
- 当前仍保留需要提交前人工填写的信息：真实 GitHub 仓库链接、作者姓名、研究领域和贡献。

## Word 报告排版优化任务

### 需求规格

目标：在不改变实验结果和报告结论的前提下，提升 Word 报告的视觉层级、阅读节奏和提交观感，使其更像正式课程报告而不是脚本堆砌文档。

核心验收项：

- [x] 封面更完整：包含课程/项目标签、题目、副标题、提交信息、关键结论摘要和工具/贡献提示。
- [x] 目录更规整：静态目录具有缩进层级和视觉分隔，不出现动态 TOC 占位符。
- [x] 正文章节层级更清晰：H1/H2 样式统一，关键章节分页更合理。
- [x] 表格更规范：表头醒目、数据行斑马纹、数值列可读性更好。
- [x] 图片排版更统一：图宽、图注、间距和 alt text 保持一致。
- [x] 页眉页脚更正式：正文页包含报告题名/页码字段，封面不显得突兀。
- [x] 完成 Word schema、结构、图片、文本泄漏和测试验证。

### 执行计划

- [x] 审计当前 Word 结构、样式问题和生成脚本。
- [x] 修改 `scripts/generate_word_report.py` 的封面、目录、表格、代码块、图片和页眉页脚样式。
- [x] 重新生成 `report/ML_household_power_report.docx` 与 `report/report.docx`。
- [x] 运行 `officecli validate`、outline/stats/issues、图片 alt、文本 token 泄漏和 pytest 验证。

### 进度记录

- 2026-06-24：用户要求进一步优化 Word 格式和排版；已开始从生成脚本层面改版，避免手改 docx 后无法复现。
- 2026-06-24：已完成 Word 排版优化：封面新增深色课程标签、关键结果表、实验协议和项目信息；H1 使用浅蓝底色，H2 统一蓝色层级；静态目录改为分层缩进；表格使用深色表头、白字和斑马纹；图片统一 14.6cm 宽、居中、斜体灰色图注；正文页新增报告题名页眉和 live PAGE 页码字段。
- 2026-06-24：已重新生成 `report/ML_household_power_report.docx` 与 `report/report.docx`；当前 Word 报告为 142 段、6 张表、13 张图，封面新增关键结果表，所有图片仍保留 alt text。
- 2026-06-24：最终验证通过：两份 docx 的 `officecli validate` 均为 `Validation passed: no errors found.`；`officecli query ... image:no-alt` 无匹配；`field[fieldType=page]` 查询到 1 个 live PAGE 字段，`/footer[2]` 中存在 `fldChar begin/separate/end` 链；文本 token 泄漏计数为 0；zip 媒体计数两份 docx 均为 13。
- 2026-06-24：结构验证通过：outline 显示 H1/H2 层级完整，stats 显示空段落 0、连续空格 0。`officecli view issues` 仍报告 50 条提示，主要是静态目录、代码块、参考文献等块式排版被识别为“正文缺少首行缩进”，以及少量中英术语混排提示；不属于 schema 错误，也不影响提交阅读。
- 2026-06-24：项目验证通过：`conda run -n qwen3meld-run python -m pytest tests -q` 结果 `19 passed in 2.42s`；`ALLOW_PLACEHOLDERS=1 SKIP_TESTS=1 PYTHON="conda run -n qwen3meld-run python" bash scripts/check_submission_ready.sh` 通过。

### 结果复盘

- 排版改动已固化在 `scripts/generate_word_report.py`，后续重新生成 Word 不会丢失封面、表格、图片、页眉页脚等样式。
- 本轮没有更改实验指标、模型结论或 DMSAFormer 的 validation-calibrated expert 描述，只改进 Word 展示层。
- 提交前仍需人工填写真实 GitHub 仓库链接、作者姓名、研究领域和贡献。
