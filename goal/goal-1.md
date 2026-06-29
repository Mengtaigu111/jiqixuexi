/goal 完成 2026 年专硕机器学习课程项目：家庭电力消耗多变量时间序列预测。请实现完整可运行代码、实验结果、图表截图、PDF 报告草稿，并准备可提交到 GitHub 的项目仓库。

项目主题：

基于 LSTM、Transformer 与改进混合模型的家庭电力消耗多变量时间序列预测

课程要求：

1. 使用家庭电力消耗数据进行多变量时间序列预测。
2. 根据过去 90 天数据预测未来：

   * 90 天变化曲线，短期预测；
   * 365 天变化曲线，长期预测。
3. 短期预测和长期预测需要分别训练，长期预测模型参数不能复用短期预测模型。
4. 必须实现三类方法：

   * LSTM 模型；
   * Transformer 模型；
   * 自己提出的改进模型。
5. 使用两种评价指标：

   * MSE；
   * MAE。
6. 每个模型、每个预测长度至少进行 5 轮实验。
7. 报告平均值 mean 和标准差 std。
8. 绘制 power 预测值与 Ground Truth 曲线对比图。
9. 报告结果需以截图或图片形式贴入 PDF。
10. 最终需要提交完整可运行代码到 GitHub，并在 PDF 报告中附 GitHub 链接。
11. PDF 报告由四部分构成：

    * 问题介绍；
    * 模型，可以包含少量伪代码；
    * 结果与分析；
    * 讨论。

第一阶段：项目结构整理

创建或整理如下目录结构：

* data/

  * raw/
  * processed/
* src/

  * data_preprocess.py
  * dataset.py
  * models/

    * lstm.py
    * transformer.py
    * hybrid_model.py
  * train.py
  * evaluate.py
  * utils.py
* scripts/

  * run_lstm_90.sh
  * run_lstm_365.sh
  * run_transformer_90.sh
  * run_transformer_365.sh
  * run_hybrid_90.sh
  * run_hybrid_365.sh
  * run_all_experiments.sh
* results/

  * metrics/
  * figures/
  * screenshots/
  * predictions/
* report/

  * report_draft.md
  * report.pdf
  * report_assets/
* README.md
* requirements.txt
* .gitignore

要求：

* 所有脚本都可以通过命令行运行。
* 不要把大数据文件和模型权重提交到 GitHub。
* README.md 中要写清楚环境安装、数据准备、训练、测试、复现实验和生成报告的方法。

第二阶段：数据检查与预处理

检查当前项目中是否已有：

* train.csv
* test.csv
* tes.csv
* weather 数据
* household power consumption 原始数据

如果文件名是 tes.csv，请兼容；如果实际是 test.csv，也要兼容。

实现 src/data_preprocess.py。

数据处理要求：

1. 读取原始分钟级家庭电力数据。
2. 处理缺失值：

   * 将 ?、空字符串、非法数值转为 NaN；
   * 时间列正确解析；
   * 数值列转为 float；
   * 缺失值可使用前向填充、后向填充或按天插值，但需要在 README 和报告中说明。
3. 按天聚合：

   * global_active_power 按天求和；
   * global_reactive_power 按天求和；
   * sub_metering_1 按天求和；
   * sub_metering_2 按天求和；
   * sub_metering_3 按天求和，如果存在；
   * voltage 按天求平均；
   * global_intensity 按天求平均；
   * RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU 取当天任意一个值，或按天取第一个非空值。
4. 如果 sub_metering_remainder 不存在，则按公式计算：
   sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)
5. 构造时间特征：

   * day_of_week
   * month
   * day_of_year
   * is_weekend
   * 可选 sin/cos 周期编码。
6. 目标变量：

   * global_active_power，即每日总有功功率。
7. 标准化：

   * 只能在 train 集上 fit scaler；
   * valid/test 只能 transform；
   * target 的反标准化要可用，以便画真实单位下的预测曲线。
8. 构造滑动窗口样本：

   * input_len = 90；
   * output_len = 90 或 365；
   * stride 可配置，默认 1；
   * 每个样本形状为：
     X: [90, num_features]
     y: [output_len]
9. 保存处理后的数据：

   * data/processed/daily_power.csv
   * data/processed/train_90.npz
   * data/processed/test_90.npz
   * data/processed/train_365.npz
   * data/processed/test_365.npz
   * data/processed/scaler.pkl

第三阶段：实现 Dataset 与 DataLoader

实现 src/dataset.py。

要求：

* 支持 output_len=90 和 output_len=365。
* 返回 X、y、date_index。
* 支持 train/valid/test。
* 支持 batch_size 参数。
* 确保测试集预测曲线可以按日期还原和绘图。

第四阶段：实现三个模型

1. LSTM 模型

文件：src/models/lstm.py

结构建议：

* 输入：[batch, 90, num_features]
* 多层 LSTM 编码历史序列。
* 取最后 hidden state。
* 经过 MLP 输出未来 output_len 天预测。
* 支持参数：

  * hidden_size
  * num_layers
  * dropout
  * output_len

2. Transformer 模型

文件：src/models/transformer.py

结构建议：

* 线性层将输入特征映射到 d_model。
* 加入 positional encoding。
* 使用 TransformerEncoder。
* pooling 或取最后 token 表示。
* MLP 输出未来 output_len 天预测。
* 支持参数：

  * d_model
  * nhead
  * num_layers
  * dim_feedforward
  * dropout
  * output_len

3. 改进模型：TCN-CNN-Transformer Hybrid

文件：src/models/hybrid_model.py

模型名称建议：

HybridTCNTransformer

结构建议：

* 输入：[batch, 90, num_features]
* 先用 1D CNN / TCN 提取局部时间模式：

  * Conv1d
  * dilation
  * residual connection
  * dropout
* 再接 TransformerEncoder 建模长期依赖。
* 加入时间特征 embedding 或周期特征。
* 最后用 MLP 多步预测头输出未来 output_len 天。
* 说明该模型的设计动机：

  * LSTM 擅长顺序建模，但长距离依赖能力有限；
  * Transformer 擅长全局依赖，但对局部波动和短期模式不一定敏感；
  * TCN/CNN 能提取局部周期、突增突降和短期波动；
  * 因此用 TCN/CNN + Transformer 兼顾局部特征和长期依赖。
* 这是本项目的开放题改进模型。

第五阶段：训练与评估脚本

实现 src/train.py。

命令行参数至少包括：

* --model lstm / transformer / hybrid
* --output_len 90 / 365
* --seed
* --epochs
* --batch_size
* --learning_rate
* --hidden_size
* --d_model
* --num_layers
* --dropout
* --data_dir
* --save_dir
* --device
* --early_stop_patience

训练要求：

* 使用 MSELoss 作为训练损失。
* 同时计算 MAE。
* 使用 Adam 或 AdamW。
* 支持 early stopping。
* 保存 best checkpoint。
* 保存训练日志为 CSV：

  * epoch
  * train_loss
  * valid_mse
  * valid_mae
* 每轮实验保存到：
  results/metrics/{model}_{output_len}*seed{seed}.csv
  results/predictions/{model}*{output_len}*seed{seed}.csv
  checkpoints/{model}*{output_len}_seed{seed}.pt

实现 src/evaluate.py。

要求：

* 读取 checkpoint。
* 在 test 集上计算：

  * MSE
  * MAE
* 保存预测结果。
* 反标准化后绘制预测曲线和 Ground Truth 对比图。
* 至少生成以下图：

  * results/figures/{model}_{output_len}_seed{seed}_curve.png
  * results/figures/{model}_{output_len}_seed{seed}_error.png
* 曲线图必须清楚标注：

  * Ground Truth
  * Prediction
  * model name
  * output_len
  * seed
  * MSE / MAE

第六阶段：五轮实验

使用 seeds：

* 2026
* 2027
* 2028
* 2029
* 2030

每个组合都运行 5 次：

* LSTM, output_len=90
* LSTM, output_len=365
* Transformer, output_len=90
* Transformer, output_len=365
* HybridTCNTransformer, output_len=90
* HybridTCNTransformer, output_len=365

总共 3 × 2 × 5 = 30 次实验。

如果完整实验耗时过长，可以先提供 quick 模式：

* epochs=3
* small subset
* 用于 smoke test

但最终报告必须使用 full 模式结果。如果 full 模式没有跑完，需要在 report_todo.md 中明确说明。

第七阶段：结果汇总

实现 scripts/run_all_experiments.sh。

实现 src/summarize_results.py。

汇总所有 seed 的结果，生成：

* results/metrics/summary.csv
* results/metrics/summary.md

summary 表格字段：

* Model
* Horizon
* MSE mean
* MSE std
* MAE mean
* MAE std

分别统计 90 天预测和 365 天预测。

生成对比图：

* results/figures/metric_bar_mse.png
* results/figures/metric_bar_mae.png
* results/figures/prediction_comparison_90.png
* results/figures/prediction_comparison_365.png

要求：

* 对三种方法进行比较。
* 90 天和 365 天分别画图。
* 图像分辨率适合放入 PDF。
* 同时将关键图片复制到 results/screenshots/，用于报告截图。

第八阶段：报告撰写

生成 report/report_draft.md。

报告必须由以下四部分组成：

1. 问题介绍
2. 模型
3. 结果与分析
4. 讨论

报告题目：

基于深度学习的家庭电力消耗多变量时间序列预测

报告内容要求：

第一部分：问题介绍

需要写：

* 智能家居和智能电网背景。
* 家庭电力消耗预测的实际意义。
* 数据集说明。
* 预测任务定义：

  * 用过去 90 天预测未来 90 天；
  * 用过去 90 天预测未来 365 天。
* 输入变量说明：

  * global_active_power
  * global_reactive_power
  * voltage
  * global_intensity
  * sub_metering_1
  * sub_metering_2
  * sub_metering_3
  * sub_metering_remainder
  * 天气变量
  * 时间特征
* 评价指标：

  * MSE
  * MAE

第二部分：模型

需要写三个模型：

1. LSTM

写清楚：

* 序列输入方式。
* 隐状态建模历史电力变化。
* 多步预测头输出未来曲线。
* 可包含简短伪代码。

2. Transformer

写清楚：

* 输入嵌入。
* 位置编码。
* 自注意力机制。
* TransformerEncoder。
* 多步预测头。

3. 改进模型 HybridTCNTransformer

写清楚：

* 局部卷积/TCN 提取短期波动和局部周期。
* Transformer 捕捉长期依赖。
* 残差结构提高稳定性。
* 为什么这个模型有新意。
* 可包含少量伪代码。

第三部分：结果与分析

需要写：

* 数据预处理结果。
* 实验设置：

  * train/test 划分；
  * input_len=90；
  * output_len=90/365；
  * seeds=5；
  * batch size；
  * optimizer；
  * epoch；
  * device。
* 结果汇总表：

  * 三种模型；
  * 两个预测长度；
  * MSE mean/std；
  * MAE mean/std。
* 插入截图：

  * summary 表截图；
  * MSE/MAE 对比图；
  * 90 天预测曲线图；
  * 365 天预测曲线图。
* 分析：

  * 90 天预测通常比 365 天更容易。
  * LSTM、Transformer、Hybrid 模型各自优缺点。
  * 如果 Hybrid 性能最好，分析原因。
  * 如果 Hybrid 性能不如 baseline，重点分析原因和新颖性。
  * 分析长期预测误差累积、季节变化、异常波动等现象。

第四部分：讨论

需要写：

* 本项目的主要发现。
* 数据缺失和真实数据噪声影响。
* 多变量特征和天气变量的作用。
* 365 天长期预测的困难。
* 改进模型的局限。
* 后续改进方向：

  * 引入更细粒度天气；
  * 加入节假日特征；
  * 使用概率预测；
  * 尝试 Informer、Autoformer、PatchTST 等时间序列模型；
  * 加入异常检测和不确定性估计。
* 明确说明是否使用了 ChatGPT 或其他工具辅助报告写作。
* 必须附上 GitHub 链接占位符：
  GitHub: https://github.com/你的用户名/你的仓库名

第九阶段：生成 PDF 报告

将 report/report_draft.md 转换为 PDF。

输出：

* report/ML_household_power_report.pdf

要求：

* PDF 中包含 GitHub 链接。
* PDF 中包含截图或图片形式的实验结果。
* PDF 中包含 power prediction vs Ground Truth 曲线。
* PDF 四大部分标题必须明确：

  1. 问题介绍
  2. 模型
  3. 结果与分析
  4. 讨论
* 参考文献必须列出。
* 报告中不得编造结果，所有结果必须来自 results/metrics/summary.csv 和实验图。

如果无法直接生成 PDF，则至少生成：

* report/report_draft.md
* report/report_draft.docx
* report/pdf_todo.md

并说明如何手工导出 PDF。

第十阶段：GitHub 提交准备

生成 README.md。

README.md 必须包含：

1. 项目简介
2. 数据来源
3. 环境安装
4. 数据预处理命令
5. 训练命令
6. 测试命令
7. 复现全部实验命令
8. 结果说明
9. 报告文件路径
10. 作者贡献说明占位符
11. 参考文献

生成 requirements.txt。

生成 .gitignore，至少排除：

* data/raw/
* data/processed/*.npz
* checkpoints/
* **pycache**/
* *.pt
* *.pth
* .ipynb_checkpoints/

准备 git 提交：

* 检查不应提交的大文件。
* 生成 github_release_checklist.md。

第十一阶段：自检

生成 project_checklist.md，逐项检查：

* 是否完成 LSTM。
* 是否完成 Transformer。
* 是否完成改进模型。
* 是否完成 90 天预测。
* 是否完成 365 天预测。
* 是否每组至少 5 个 seed。
* 是否计算 MSE 和 MAE。
* 是否计算 mean 和 std。
* 是否生成预测曲线和 Ground Truth 对比图。
* 是否生成结果截图。
* 是否生成 PDF 报告。
* PDF 是否包含 GitHub 链接。
* PDF 是否包含四部分：问题介绍、模型、结果与分析、讨论。
* README 是否可指导他人运行代码。
* GitHub 是否准备提交。

最终交付文件：

必须生成：

* src/data_preprocess.py
* src/dataset.py
* src/models/lstm.py
* src/models/transformer.py
* src/models/hybrid_model.py
* src/train.py
* src/evaluate.py
* src/summarize_results.py
* scripts/run_all_experiments.sh
* results/metrics/summary.csv
* results/metrics/summary.md
* results/figures/metric_bar_mse.png
* results/figures/metric_bar_mae.png
* results/figures/prediction_comparison_90.png
* results/figures/prediction_comparison_365.png
* report/report_draft.md
* report/ML_household_power_report.pdf
* README.md
* requirements.txt
* github_release_checklist.md
* project_checklist.md

验收标准：

1. 三种模型都能运行。
2. 90 天和 365 天任务分别训练。
3. 每种模型和预测长度至少 5 轮实验。
4. summary.csv 中包含 MSE/MAE 的 mean 和 std。
5. 报告中包含三种方法比较。
6. 报告中包含预测曲线和 Ground Truth 对比图。
7. 报告中包含 GitHub 链接。
8. 代码可被他人根据 README 复现。
