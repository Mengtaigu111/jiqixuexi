# 基于深度学习的家庭电力消耗多变量时间序列预测

GitHub: https://github.com/Mengtaigu111/jiqixuexi

作者贡献与研究领域：待填写。最多 2 人组队时，请分别列明姓名、所属研究领域和具体贡献。

工具使用说明：本报告草稿允许使用 ChatGPT/DeepSeek 等工具辅助撰写，最终提交前需由作者核对实验结果、参考文献和表述准确性。

实验结果说明：本报告结果来自 UCI Individual household electric power consumption 原始分钟级数据，已完成 LSTM、Transformer、HybridTCNTransformer 三种模型在 90 天和 365 天预测任务上的 5 个 seed 实验，共 30 次训练与评估。

## 1. 问题介绍

随着智能家居、物联网和智能电网技术的发展，家庭电力消耗预测对节能减排、居民用电行为分析、电力负荷调度和分布式能源管理具有实际意义。家庭用电受季节、天气、家庭活动模式、设备使用和异常事件影响，表现出明显的多变量时间序列特征。

本项目采用 Individual household electric power consumption 数据集。原始数据以分钟为粒度，包含 `global_active_power`、`global_reactive_power`、`voltage`、`global_intensity`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 等变量；同时可加入 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 等天气变量。预处理时将分钟级数据按天聚合，构造时间特征，并以过去 90 天的多变量序列作为输入。

预测任务包括两类：

1. 用过去 90 天预测未来 90 天总有功功率曲线。
2. 用过去 90 天预测未来 365 天总有功功率曲线。

两个任务分别训练模型，长期预测模型参数不复用短期预测模型。评价指标为均方误差 MSE 和平均绝对误差 MAE。

## 2. 模型

### 2.1 LSTM

LSTM 模型将输入序列表示为 `[batch, 90, num_features]`，通过多层 LSTM 编码历史电力变化。模型取最后一层的最终隐状态作为历史窗口表示，再经过多层感知机输出未来 `output_len` 天的预测曲线。

简要伪代码：

```text
h, c = LSTM(X)
z = last_hidden_state(h)
y_hat = MLP(z)
```

LSTM 的优势是顺序建模能力强，适合捕捉连续时间变化；局限是对非常长的依赖和复杂周期模式建模能力有限。

### 2.2 Transformer

Transformer 模型先将多变量输入映射到 `d_model` 维表示，加入正弦位置编码，然后使用 `TransformerEncoder` 通过自注意力机制建模不同日期之间的全局依赖。最后对编码后的序列做平均池化，并通过预测头输出未来曲线。

简要伪代码：

```text
E = Linear(X) + PositionalEncoding
H = TransformerEncoder(E)
z = MeanPool(H)
y_hat = MLP(z)
```

Transformer 的优势是并行建模全局依赖；局限是对局部突增突降和短周期模式不一定敏感。

### 2.3 改进模型 HybridTCNTransformer

HybridTCNTransformer 先使用 1D CNN/TCN 残差块提取局部时间模式，包括短期波动、局部周期和突增突降；再接 TransformerEncoder 捕捉更长范围的依赖。该结构希望结合 TCN/CNN 的局部感知能力与 Transformer 的全局建模能力。

简要伪代码：

```text
Z = Conv1dProjection(X)
Z = TCNResidualBlocks(Z)
H = TransformerEncoder(PositionalEncoding(Z))
y_hat = MLP(MeanPool(H))
```

该模型的新意在于：将局部模式提取放在全局依赖建模之前，使 Transformer 处理的表示已经包含更稳定的局部用电模式。

## 3. 结果与分析

### 3.1 数据预处理

预处理脚本完成以下步骤：

- 将 `?`、空字符串和非法数值转为缺失值。
- 时间列按日解析并排序。
- 数值列转为浮点数。
- 核心电力传感器列采用前向填充和后向填充；若仍存在不可恢复缺失则报错，不将缺失功率直接补 0。
- 可选天气变量缺失时仅作为占位字段补 0。
- `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按天求和。
- `voltage`、`global_intensity` 按天取平均。
- 天气变量取当天第一个非空值。
- 计算 `sub_metering_remainder = global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3`。
- 增加星期、月份、年内日、周末标记和年周期 sin/cos 编码。
- 只在 train 集上拟合 scaler，valid/test 只 transform。

### 3.2 实验设置

正式实验设置为：

- `input_len=90`
- `output_len=90/365`
- 模型：LSTM、Transformer、HybridTCNTransformer
- seeds：2026、2027、2028、2029、2030
- epoch：30
- batch size：64
- 优化器：AdamW
- 损失函数：MSELoss
- 评价指标：MSE、MAE
- device：CUDA GPU
- 数据划分：按日期连续划分为 train/valid/test。当前窗口样本数为 `train_90/valid_90/test_90 = 353/366/366`，`train_365/valid_365/test_365 = 78/91/91`。

### 3.3 结果汇总表

| Model | Horizon | MSE mean | MSE std | MAE mean | MAE std | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | 90 | 155633.435294 | 3504.548138 | 302.205511 | 2.725820 | 5 |
| lstm | 90 | 163266.742215 | 1593.567804 | 312.448075 | 2.165346 | 5 |
| transformer | 90 | 156632.342114 | 3657.475827 | 305.301225 | 4.984317 | 5 |
| hybrid | 365 | 368574.752477 | 43798.405207 | 491.580361 | 35.248571 | 5 |
| lstm | 365 | 316352.062831 | 16271.974417 | 446.398379 | 12.259544 | 5 |
| transformer | 365 | 442238.940370 | 45530.658164 | 545.818405 | 32.605792 | 5 |

### 3.4 图表

MSE 对比图：`results/figures/metric_bar_mse.png`

MAE 对比图：`results/figures/metric_bar_mae.png`

90 天预测对比图：`results/figures/prediction_comparison_90.png`

365 天预测对比图：`results/figures/prediction_comparison_365.png`

90 天预测中，HybridTCNTransformer 的 MSE 和 MAE 均最低，说明 TCN/CNN 对局部波动和短周期模式的提取对短期预测有帮助；Transformer 与 Hybrid 的指标接近，均优于 LSTM。365 天预测中，LSTM 取得最低 MSE 和 MAE，Hybrid 位于第二，Transformer 误差最高。这说明在当前数据划分和训练规模下，长期预测对模型稳定性要求更高，复杂模型不一定直接带来更低误差；Hybrid 的新颖性主要体现在局部模式提取与全局依赖建模的组合，但长期预测仍受样本量、季节变化和误差累积影响。

## 4. 讨论

本项目已经完成可运行代码链路和正式 30 次实验，包括预处理、Dataset、三类模型、训练、评估、图表和汇总。真实数据中的缺失值和用电噪声会影响模型稳定性，因此本项目将 `?` 等非法标记转为 NaN 后，对核心电力传感器列采用时间序列前向/后向填充；不可恢复的核心缺失会触发报错，避免将缺测误解释为真实零用电。标准化器严格只在 train 集上拟合。

多变量输入和天气变量能够提供用电行为之外的外部因素；但月度天气变量较粗，可能无法捕捉短期温度、降水和节假日造成的日级波动。365 天长期预测难度更高，模型需要同时处理趋势、季节性和异常扰动。

后续改进方向包括：

- 引入更细粒度天气数据。
- 加入节假日和工作日特征。
- 尝试概率预测，给出不确定性区间。
- 尝试 Informer、Autoformer、PatchTST 等时间序列模型。
- 加入异常检测，降低异常用电对训练的影响。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption Data Set. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

[3] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.

[4] PyTorch Documentation. https://pytorch.org/docs/stable/nn.html
