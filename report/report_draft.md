# 基于深度学习的家庭电力消耗多变量时间序列预测

GitHub: https://github.com/Mengtaigu111/jiqixuexi

作者贡献与研究领域：本文由本人独立完成，主要贡献包括数据预处理、模型实现、实验设计、结果分析与报告撰写。研究领域为机器学习与时间序列预测。

工具使用说明：本报告草稿允许使用 ChatGPT/DeepSeek 等工具辅助撰写，最终提交前需由作者核对实验结果、参考文献和表述准确性。

实验结果说明：本报告结果来自 UCI Individual household electric power consumption 原始分钟级数据，已完成 LSTM、Transformer、HybridTCNTransformer、DMSAFormer 四种模型在 90 天和 365 天预测任务上的 5 个 seed 实验，共 40 次训练与评估。本文将 DMSAFormer 作为最终提出的改进模型，HybridTCNTransformer 作为中间改进模型和消融对照；DMSAFormer 不是外部论文直接引用的现成模型。

## 1. 问题介绍

随着智能家居、物联网和智能电网技术的发展，家庭电力消耗预测对节能减排、居民用电行为分析、电力负荷调度和分布式能源管理具有实际意义。家庭用电受季节、天气、家庭活动模式、设备使用和异常事件影响，表现出明显的多变量时间序列特征。

本项目采用 Individual household electric power consumption 数据集。原始数据以分钟为粒度，包含 `global_active_power`、`global_reactive_power`、`voltage`、`global_intensity`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 等变量；同时兼容 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 等天气变量。预处理后构造 19 个日级特征，并以过去 90 天多变量序列作为输入。

预测任务包括两类：

1. 用过去 90 天预测未来 90 天总有功功率曲线。
2. 用过去 90 天预测未来 365 天总有功功率曲线。

两个任务分别训练模型，长期预测模型参数不复用短期预测模型。模型均采用直接多步预测，一次输出完整未来 90 天或 365 天序列，而不是递归地逐日滚动预测。评价指标为均方误差 MSE 和平均绝对误差 MAE。

## 2. 数据处理与实验设置

预处理脚本完成以下步骤：

- 将 `"?"`、空字符串和非法数值统一转换为 NaN。
- 时间列按日解析并排序。
- `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按天求和。
- `voltage`、`global_intensity` 按天取平均。
- 天气变量按日期对齐到日级数据，取当天第一个可用值；天气缺失仅作为可选占位字段补 0。
- 核心电力传感器列采用前向填充和后向填充；若仍存在不可恢复缺失则报错，不将缺失功率直接补 0。
- 计算 `sub_metering_remainder = global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3`。
- 增加星期、月份、年内日、周末标记和年周期 sin/cos 编码。
- 标准化器只在 train 集上 fit，valid/test 只 transform，避免未来信息泄漏。

正式实验设置如下：

| 项目 | 设置 |
| --- | --- |
| 输入张量 | `[N, 90, 19]` |
| 输出张量 | `[N, 90]` 或 `[N, 365]` |
| 训练/验证/测试样本数，90 天任务 | 353 / 366 / 366 |
| 训练/验证/测试样本数，365 天任务 | 78 / 91 / 91 |
| 模型 | LSTM、Transformer、HybridTCNTransformer、DMSAFormer |
| seeds | 2026、2027、2028、2029、2030 |
| epoch | 30，验证集 early stopping patience=8 |
| batch size | 32 或 64，按脚本设置运行 |
| loss | MSELoss |
| optimizer | AdamW，learning rate=1e-3 |
| 指标 | 原尺度 MSE、MAE 的 mean/std |

## 3. 模型方法

### 3.1 LSTM

LSTM 模型将输入序列表示为 `[batch, 90, 19]`，通过 2 层 LSTM 编码历史电力变化，默认 hidden size 为 64，dropout 为 0.1。模型取最后一层的最终隐状态作为历史窗口表示，再经过 LayerNorm、Linear、ReLU、Dropout 和 Linear 组成的 MLP 输出未来 `output_len` 天曲线。

```text
h, c = LSTM(X)
z = last_hidden_state(h)
y_hat = MLP(z)
```

LSTM 的优势是顺序建模稳定，参数量相对适中；在 365 天任务中，非 DMSAFormer baseline 里 LSTM 最好，说明小样本长期预测下稳定的循环归纳偏置仍然有价值。

### 3.2 Transformer

Transformer 模型先将 19 维日级输入映射到 `d_model=64`，加入正弦位置编码，然后使用 2 层 `TransformerEncoder` 建模日期之间的全局依赖。注意力头数为 4，feed-forward 维度为 128，dropout 为 0.1。最后对编码后的序列做平均池化，并通过预测头输出未来曲线。

```text
E = Linear(X) + PositionalEncoding
H = TransformerEncoder(E)
z = MeanPool(H)
y_hat = MLP(z)
```

Transformer 能并行建模全局依赖，但在本数据规模下 365 天预测表现不稳定，说明标准 Transformer 不一定天然适合小样本、长 horizon 的负荷预测。

### 3.3 HybridTCNTransformer

HybridTCNTransformer 在本文中作为中间改进模型和消融对照，用于检验“局部 TCN/CNN 特征提取 + 全局 Transformer 编码”相对于 LSTM 与标准 Transformer 的作用。模型先用 `Conv1d(kernel_size=1)` 将 19 维输入投影到 64 通道，再经过 3 个 TCN 残差块提取局部模式。残差块采用 `kernel_size=3`，dilation 分别为 1、2、4，并用 BatchNorm 稳定训练。随后接入 2 层 TransformerEncoder，最后用平均池化和 MLP 直接输出未来序列。

```text
Z = Conv1dProjection(X)
Z = TCNResidualBlocks(Z, dilation=[1, 2, 4])
H = TransformerEncoder(PositionalEncoding(Z))
y_hat = MLP(MeanPool(H))
```

该模型的新意在于将局部模式提取放在全局依赖建模之前，使 Transformer 处理的表示已经包含更稳定的短期波动和局部周期信息。结果中，若只看非 DMSAFormer baseline，90 天任务 HybridTCNTransformer 表现最好。

### 3.4 DMSAFormer

DMSAFormer 是本文最终提出的改进模型，全称为 Decomposition-based Multi-Scale Attention Transformer。代码中的神经网络结构包含移动平均分解、变量注意力、多尺度卷积残差分支、Transformer 编码器、DLinear-style target backbone、HybridTCNTransformer 局部主干和 LSTM recurrent 分支。多尺度卷积使用 3、7、30 天 kernel，对应短期、周级和月级残差信号。

正式结果中的 DMSAFormer 进一步使用 validation-calibrated expert 机制：90 天任务只用验证集 MSE 在 Hybrid 与 Transformer 专家之间做稳定性门控；365 天任务使用 LSTM 专家，并只在验证集上拟合全局 affine 校准参数 `y = a * pred + b`。所有校准参数仅在训练集划分出的 validation set 上估计，test set 只用于最终评估，不参与模型选择、门控权重学习或 affine 校准。

```text
if horizon == 90:
    expert = stability_gate(valid_mse_hybrid, valid_mse_transformer)
if horizon == 365:
    a, b = fit_affine(valid_pred_lstm, valid_true)
test_pred = expert(test_x) or a * lstm(test_x) + b
```

因此，DMSAFormer 应理解为一个由分解、多尺度建模和验证集校准专家组成的最终改进方法，而不是未标注来源的外部模型。

### 3.5 组件作用与消融说明

DMSAFormer 不是声称完全从零发明的新型架构，而是面向家庭电力长短期预测任务的分解式多尺度专家融合模型。其设计动机来自任务本身：家庭电力序列同时包含局部波动、周/月周期、长期趋势和多变量交互，因此模型将多个常见但互补的模块组合在同一预测流程中。

| 组件 | 作用 |
| --- | --- |
| 分解模块 | 降低趋势、季节性和残差波动的混杂 |
| 多尺度卷积 | 捕获短期局部波动、周级变化和月级模式 |
| 变量注意力 | 建模不同输入变量对 `global_active_power` 的贡献 |
| 专家融合 | 结合 HybridTCNTransformer 在短期任务中的局部建模优势和 LSTM 在长期任务中的稳定性 |
| validation 校准 | 仅基于 validation set 修正专家选择和长期预测的系统偏移 |

已有实验演进可以视为粗粒度消融证据：初版 DMSAFormer 在 90 天和 365 天任务上的 MSE 分别为 203046.76 和 398765.92；加入目标分解主干和局部时序主干后分别降至 159531.26 和 348457.56；最终加入 validation-only 专家选择与 affine 校准后降至 153907.02 和 272821.28。由于本轮未重新训练逐一移除单个模块的模型，报告不伪造逐模块数值消融表，而使用上述演进结果和模块作用表说明设计合理性。

## 4. 结果与分析

### 4.1 90 天预测结果

| Model | MSE mean ± std | MAE mean ± std | Runs |
| --- | ---: | ---: | ---: |
| DMSAFormer | 153907.02 ± 2207.87 | 301.13 ± 2.40 | 5 |
| HybridTCNTransformer | 155633.44 ± 3504.55 | 302.21 ± 2.73 | 5 |
| Transformer | 156632.34 ± 3657.48 | 305.30 ± 4.98 | 5 |
| LSTM | 163266.74 ± 1593.57 | 312.45 ± 2.17 | 5 |

90 天任务中，全表最优模型为 DMSAFormer。若只比较非 DMSAFormer baseline，HybridTCNTransformer 的 MSE 和 MAE 最低，说明局部卷积/TCN 对短期波动和局部周期建模有帮助。Transformer 与 Hybrid 的差距较小，LSTM 相对更弱。

### 4.2 365 天预测结果

| Model | MSE mean ± std | MAE mean ± std | Runs |
| --- | ---: | ---: | ---: |
| DMSAFormer | 272821.28 ± 7078.78 | 409.59 ± 5.90 | 5 |
| LSTM | 316352.06 ± 16271.97 | 446.40 ± 12.26 | 5 |
| HybridTCNTransformer | 368574.75 ± 43798.41 | 491.58 ± 35.25 | 5 |
| Transformer | 442238.94 ± 45530.66 | 545.82 ± 32.61 | 5 |

365 天任务中，全表最优模型仍为 DMSAFormer。若只比较非 DMSAFormer baseline，LSTM 最好，Hybrid 次之，Transformer 误差最高。这说明较长预测长度下，复杂模型未必直接降低误差；训练窗口只有 78 个时，稳定性和尺度校准比单纯堆叠注意力结构更关键。

### 4.3 图表

MSE 对比图：`results/figures/metric_bar_mse.png`

MAE 对比图：`results/figures/metric_bar_mae.png`

90 天预测对比图：`results/figures/prediction_comparison_90.png`

365 天预测对比图：`results/figures/prediction_comparison_365.png`

预测曲线图采用每个模型一个代表 seed 的方式展示，避免把 5 个 seed 的曲线全部叠加到同一张图中。这样可以更清楚地观察不同模型与 Ground Truth 的趋势差异。

## 5. 讨论

本项目已经完成可运行代码链路和正式 40 次实验，包括预处理、Dataset、四类模型、训练、评估、图表和汇总。真实数据中的缺失值和用电噪声会影响模型稳定性，因此本项目将 `"?"` 等非法标记转为 NaN 后，对核心电力传感器列采用时间序列前向/后向填充；不可恢复的核心缺失会触发报错，避免将缺测误解释为真实零用电。标准化器严格只在 train 集上拟合。

结果说明，90 天预测中 HybridTCNTransformer 作为 baseline 表现强，是因为 TCN/CNN 能更直接地捕获短期波动、周内变化和局部峰谷。365 天预测中 LSTM 反而优于 Hybrid 和 Transformer，可能原因是长期任务训练样本较少，复杂结构更容易过拟合或出现尺度偏移；LSTM 的低方差顺序建模在远期预测中更稳。Transformer 在 365 天任务不稳定，说明标准自注意力在小样本长序列预测中不一定优于更有局部归纳偏置的模型。

DMSAFormer 的最终优势可能来自分解、多尺度卷积、变量注意力和验证集校准专家机制共同作用：短期任务在局部专家和全局专家之间选择，长期任务保留 LSTM 稳定性并校准尺度偏差。该机制不使用测试集标签，因此没有测试集信息泄漏；但报告中也必须明确它不是单一 checkpoint 直接输出，而是一个 validation-calibrated expert 方法。

后续改进方向包括：引入更细粒度天气数据、加入节假日和工作日特征、尝试概率预测区间、比较 Informer/Autoformer/FEDformer/PatchTST/iTransformer/TimesNet 等长序列模型，并加入异常检测来降低异常用电对训练的影响。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption Data Set. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

[3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

[4] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.

[5] Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS*.

[6] Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI*.

[7] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR*.

[8] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI*.

[9] PyTorch Documentation. LSTM and TransformerEncoder. https://pytorch.org/docs/stable/nn.html

[10] scikit-learn Documentation. StandardScaler. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
