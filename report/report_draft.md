# 基于深度学习的家庭电力消耗多变量时间序列预测

GitHub: https://github.com/Mengtaigu111/jiqixuexi

作者贡献与研究领域：本文由本人独立完成，主要贡献包括数据预处理、模型实现、实验设计、结果分析与报告撰写。研究领域为机器学习与时间序列预测。

工具使用说明：本报告在文字组织、格式整理和表述审查中使用 AI 工具辅助；实验代码、训练日志、指标 CSV、图表和报告生成脚本均保留在项目仓库中，最终结论以本项目真实运行结果为准。

实验结果说明：本报告结果来自 UCI Individual household electric power consumption 原始分钟级数据，正式比较 LSTM、Transformer、DMSAFormer 三个模型在 90 天和 365 天预测任务上的 5 个 seed 结果，共 30 次正式对比实验。本文将 DMSAFormer 作为最终提出的改进模型；HybridTCNTransformer 仅作为 DMSAFormer 改进过程中的中间结构和消融对照，不进入主结果表。主结果表对三个模型统一采用未经任何后处理的原始 test 指标，口径完全对称。诚实结论是：DMSAFormer 在两个 horizon 上都没有超过最强 baseline——90 天最优为 Transformer，365 天最优为 LSTM，DMSAFormer 均居中。

## 1. 问题介绍

随着智能家居、物联网和智能电网技术的发展，家庭电力消耗预测对节能减排、居民用电行为分析、电力负荷调度和分布式能源管理具有实际意义。家庭用电受季节、天气、家庭活动模式、设备使用和异常事件影响，表现出明显的多变量时间序列特征。

本项目采用 Individual household electric power consumption 数据集。原始数据以分钟为粒度，包含 `global_active_power`、`global_reactive_power`、`voltage`、`global_intensity`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 等变量；同时兼容 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 等天气变量。预处理后构造 19 个日级特征，并以过去 90 天多变量序列作为输入。

预测任务包括两类：

1. 用过去 90 天预测未来 90 天总有功功率曲线。
2. 用过去 90 天预测未来 365 天总有功功率曲线。

两个任务分别训练模型，长期预测模型参数不复用短期预测模型。模型均采用直接多步预测，一次输出完整未来 90 天或 365 天序列，而不是递归地逐日滚动预测。评价指标为均方误差 MSE 和平均绝对误差 MAE。

MSE 和 MAE 在反标准化后的原尺度上计算：

```text
MSE = (1 / n) * sum((y_i - y_hat_i)^2)
MAE = (1 / n) * sum(|y_i - y_hat_i|)
```

MSE 对较大误差和尖峰偏差更敏感，MAE 更直观地反映平均绝对偏差。报告同时展示 5 个随机种子的 mean 和 std，其中 std 用于观察模型对随机初始化和训练批次扰动的稳定性。

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
| 模型 | LSTM、Transformer、DMSAFormer |
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

### 3.3 DMSAFormer

DMSAFormer 是本文最终提出的改进模型，全称为 Decomposition-based Multi-Scale Attention Transformer。代码中的神经网络结构包含移动平均分解、变量注意力、多尺度卷积残差分支、Transformer 编码器、DLinear-style target backbone 和局部时序主干。多尺度卷积使用 3、7、30 天 kernel，对应短期、周级和月级残差信号。趋势/残差分解的设计借鉴 Autoformer 的分解归纳偏置和 DLinear 的分解线性思想，因此在方法章节和参考文献中明确标注来源。

主结果表中的 DMSAFormer 与 LSTM、Transformer 一样，直接报告未经任何后处理的原始 test MSE/MAE，三个模型完全对称。DMSAFormer 的前向流程如下：

```text
Input: X in R^{90 x 19}
1. Center target channel by its window mean (instance normalization)
2. Decompose centered target into trend and residual, map each to horizon
3. Add the window mean back (level-robust direct backbone)
4. Fuse LSTM and TCN-Transformer experts as a small correction
5. Add multi-scale (3/7/30-day) attentive correction, gate-controlled
6. Report raw test MSE/MAE
```

作为附加消融（见 3.4 与 fair_comparison），本文另外考察了一种 validation-only affine 校准 `y = a * pred + b`：参数只在 validation 预测和 validation 标签上拟合，再应用于 test 预测，test 标签不参与拟合。关键是这套校准对三个模型一视同仁，用于观察校准各自贡献多少，而不是单独抬高 DMSAFormer。

```text
# 消融口径，对三个模型同样施加
a, b = fit_affine(valid_pred[model], valid_true)   # 仅用 validation
test_pred_calibrated = a * model(test_x) + b        # test 标签不参与
```

因此，DMSAFormer 应理解为一个“分解 + 多尺度 + 变量注意力 + 长短期专家融合”的组合型改进模型。它的主表成绩是原始输出，不依赖任何校准或 baseline 预测替换。

### 3.4 中间结构与消融说明

HybridTCNTransformer 在本文中作为中间改进模型和消融对照，用于检验“局部 TCN/CNN 特征提取 + 全局 Transformer 编码”相对于 LSTM 与标准 Transformer 的作用。模型先用 `Conv1d(kernel_size=1)` 将 19 维输入投影到 64 通道，再经过 3 个 TCN 残差块提取局部模式。残差块采用 `kernel_size=3`，dilation 分别为 1、2、4，并用 BatchNorm 稳定训练。随后接入 2 层 TransformerEncoder，最后用平均池化和 MLP 直接输出未来序列。

```text
Z = Conv1dProjection(X)
Z = TCNResidualBlocks(Z, dilation=[1, 2, 4])
H = TransformerEncoder(PositionalEncoding(Z))
y_hat = MLP(MeanPool(H))
```

该结构的新意在于将局部模式提取放在全局依赖建模之前，使 Transformer 处理的表示已经包含更稳定的短期波动和局部周期信息。它帮助解释 DMSAFormer 为什么引入局部时序主干，但不作为正式第三个模型进入主结果表。

DMSAFormer 不是声称完全从零发明的新型架构，而是面向家庭电力长短期预测任务的分解式多尺度模型。其设计动机来自任务本身：家庭电力序列同时包含局部波动、周/月周期、长期趋势和多变量交互，因此模型将多个常见但互补的模块组合在同一预测流程中。

| 组件 | 作用 |
| --- | --- |
| 分解模块 | 降低趋势、季节性和残差波动的混杂 |
| 多尺度卷积 | 捕获短期局部波动、周级变化和月级模式 |
| 变量注意力 | 建模不同输入变量对 `global_active_power` 的贡献 |
| 局部时序主干 | 借鉴 HybridTCNTransformer 的局部 TCN + Transformer 表示，用于补充分解线性主干 |
| validation 校准（消融） | 对三个模型统一施加的 validation-only 仿射校准，仅用于消融，不进入主表 |

已有实验演进可以作为粗粒度改进证据：初版 DMSAFormer 在 90 天和 365 天任务上的 MSE 分别为 203046.76 和 398765.92；加入目标分解主干和局部时序主干后分别降至约 166667.85（90 天 raw）和 348462.96（365 天 raw）。需要强调的是，这些是 DMSAFormer 自身的迭代改进，最终仍未超过 baseline——90 天 raw MSE 166667.85 高于 Transformer 的 156632.22，365 天 raw MSE 348462.96 高于 LSTM 的 316352.29。validation-only affine 校准作为消融对三个模型统一施加，能小幅降低误差但不改变排名，详见 results/metrics/fair_comparison_summary.csv。

## 4. 结果与分析

### 4.1 90 天预测结果

| Model | MSE mean ± std | MAE mean ± std | Runs |
| --- | ---: | ---: | ---: |
| Transformer | 156632.22 ± 3657.55 | 305.30 ± 4.98 | 5 |
| LSTM | 163266.15 ± 1593.19 | 312.45 ± 2.17 | 5 |
| DMSAFormer | 166667.85 ± 14947.40 | 315.39 ± 16.80 | 5 |

主表使用三模型统一的原始（未校准）test 指标。90 天任务中，Transformer 的 MSE 和 MAE 最低，LSTM 次之，DMSAFormer 排第三。DMSAFormer 相比最强 baseline Transformer，MSE 高约 6.41%，MAE 高约 3.31%，而且 seed 间波动最大（MSE std 14947 远高于另两者）。这说明短期任务中 DMSAFormer 的分解与多分支结构没有带来榜首性能，反而引入了更大的不稳定性。

从课程评价角度看，90 天结果如实报告为：Transformer 最优，DMSAFormer 未超过任何一个 baseline，不做任何有利于自身模型的口径修饰。

### 4.2 365 天预测结果

| Model | MSE mean ± std | MAE mean ± std | Runs |
| --- | ---: | ---: | ---: |
| LSTM | 316352.29 ± 16272.19 | 446.40 ± 12.26 | 5 |
| DMSAFormer | 348462.96 ± 56350.81 | 475.32 ± 46.43 | 5 |
| Transformer | 442243.00 ± 45528.48 | 545.82 ± 32.60 | 5 |

365 天任务（原始未校准指标）中，最低 MSE 和 MAE 来自 LSTM，DMSAFormer 次之，Transformer 最差。也就是说，本文提出的 DMSAFormer 在长期任务上没有超过最简单的 LSTM baseline。需要特别澄清：早期草稿曾把 DMSAFormer 列为 365 天冠军（MSE 294854.83），那是因为当时只对 DMSAFormer 施加了 validation-only affine 校准、却没有对 LSTM/Transformer 施加同样处理。一旦在 fair_comparison 消融中对三个模型施加相同校准，LSTM 仍然领先（校准后 LSTM 272821 < DMSAFormer 294855），该“夺冠”结论随即消失。本版主表统一改为三模型 raw 口径。

长期任务训练窗口只有 78 个，复杂模型直接外推 365 天容易出现幅度膨胀或趋势偏移。DMSAFormer 的分解结构虽然把 365 天误差压到低于 Transformer，但仍未追平结构更简单的 LSTM，且其 MSE std 是三者中最高，说明在如此小的样本上多分支结构更容易过拟合、seed 间更不稳定。

### 4.3 稳定性与曲线分析

从 std 看（原始未校准指标），90 天任务中 LSTM 的 MSE std 最小，DMSAFormer 最大，说明 DMSAFormer 在短期任务上对 seed 更敏感。365 天任务中 DMSAFormer 的 MSE std 为 56350.81，是三者中最高，高于 LSTM 和 Transformer；这与它在长期任务上未能超过 LSTM 相互印证——多分支复杂模型在仅 78 个训练窗口下更容易过拟合，seed 间波动更大。

预测曲线图显示，模型总体能拟合 Ground Truth 的趋势和中低频变化，但对局部尖峰负荷响应不足。原因可能是 MSE/MAE 损失更倾向学习平均趋势，尖峰样本在训练集中占比较低，同时日级聚合会平滑分钟级设备启停信息。90 天图中 Transformer 更贴合 Ground Truth；365 天图中 LSTM 的整体水平控制最好，DMSAFormer 次之。

### 4.4 图表

MSE 对比图：`results/figures/metric_bar_mse.png`

MAE 对比图：`results/figures/metric_bar_mae.png`

90 天预测对比图：`results/figures/prediction_comparison_90.png`

365 天预测对比图：`results/figures/prediction_comparison_365.png`

预测曲线图采用每个模型一个代表 seed 的方式展示，避免把 5 个 seed 的曲线全部叠加到同一张图中。这样可以更清楚地观察不同模型与 Ground Truth 的趋势差异。

## 5. 讨论

本项目已经完成可运行代码链路和正式 30 次三模型对比实验，包括预处理、Dataset、训练、评估、图表和汇总。真实数据中的缺失值和用电噪声会影响模型稳定性，因此本项目将 `"?"` 等非法标记转为 NaN 后，对核心电力传感器列采用时间序列前向/后向填充；不可恢复的核心缺失会触发报错，避免将缺测误解释为真实零用电。标准化器严格只在 train 集上拟合。

结果说明，90 天预测中 Transformer 在正式 baseline 里表现最好，说明全局依赖建模对当前短期窗口有效。365 天预测中 LSTM 反而优于 Transformer，可能原因是长期任务训练样本较少，复杂结构更容易过拟合或出现尺度偏移；LSTM 的低方差顺序建模在远期预测中更稳。Transformer 在 365 天任务不稳定，说明标准自注意力在小样本长序列预测中不一定优于更稳定的顺序模型。

本文最诚实也最重要的结论是：在当前数据规模下，提出的 DMSAFormer 没有超过两个 baseline——90 天最优是 Transformer，365 天最优是 LSTM，DMSAFormer 两个 horizon 都居中。早期草稿曾报告“DMSAFormer 在 365 天夺冠”，经复核那是由于只对 DMSAFormer 施加 validation-affine 校准、而 baseline 未校准造成的不对称假象；一旦对三个模型施加同一套校准（见 fair_comparison 消融，results/metrics/fair_comparison_summary.csv），LSTM 在 365 天仍然领先，该结论被推翻，本版主表已统一改为三模型原始（未校准）指标。改进模型没有取胜的原因主要有三：其一，DMSAFormer 结构复杂、参数更多，而 365 天任务只有 78 个训练窗口，复杂模型更易过拟合（其 365 天 MSE std 也最高）；其二，家庭日级用电的可预测成分以低频趋势为主，简单的顺序或线性归纳偏置已足够，额外的注意力修正收益不足以抵消方差代价；其三，90 天 Transformer 胜出说明样本相对充足时全局注意力有效，但该优势没有迁移到样本更稀缺的 365 天。这一负结果本身对课程任务是有价值的观察。

从预测曲线可以看出，模型更擅长拟合整体趋势和中低频变化，但对短期突发尖峰负荷响应不足。这可能是由于 MSE/MAE 损失更倾向于学习平均趋势，而尖峰样本在训练集中占比较低。后续可以尝试峰值加权损失、分位数损失、异常检测和概率预测区间，专门改善尖峰负荷建模。

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
