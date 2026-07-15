from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager

from src.utils import ensure_dir

A4 = (8.27, 11.69)


@dataclass(frozen=True)
class ReportTextPage:
    title: str
    body: str


def _font(font_path: Path, size: int = 11) -> FontProperties:
    if font_path.exists():
        fontManager.addfont(str(font_path))
        return FontProperties(fname=str(font_path), size=size)
    return FontProperties(size=size)


def _display_width(text: str) -> int:
    return sum(2 if re.match(r"[\u4e00-\u9fff]", char) else 1 for char in text)


def _tokens_for_wrap(line: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char.isspace():
            tokens.append(" ")
            index += 1
        elif ord(char) < 128:
            start = index
            while index < len(line) and ord(line[index]) < 128 and not line[index].isspace():
                index += 1
            tokens.append(line[start:index])
        else:
            tokens.append(char)
            index += 1
    return tokens


def _split_long_token(token: str, width: int) -> list[str]:
    parts: list[str] = []
    current = ""
    current_width = 0
    for char in token:
        char_width = _display_width(char)
        if current and current_width + char_width > width:
            parts.append(current)
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width
    if current:
        parts.append(current)
    return parts


def _wrap_line(line: str, width: int = 64) -> list[str]:
    if not line.strip():
        return [""]
    lines: list[str] = []
    current = ""
    current_width = 0
    for token in _tokens_for_wrap(line):
        token_width = _display_width(token)
        if token_width > width:
            if current.strip():
                lines.append(current.rstrip())
                current = ""
                current_width = 0
            lines.extend(_split_long_token(token, width))
            continue
        if current.strip() and current_width + token_width > width:
            lines.append(current.rstrip())
            current = token.lstrip()
            current_width = _display_width(current)
        else:
            current += token
            current_width += token_width
    if current.strip():
        lines.append(current.rstrip())
    return lines or [line]


def _wrap_paragraphs(text: str, width: int = 64) -> list[str]:
    lines: list[str] = []
    for raw in text.strip().splitlines():
        lines.extend(_wrap_line(raw, width=width))
        lines.append("")
    return lines


def add_text_page(pdf: PdfPages, title: str, body: str, font: FontProperties) -> None:
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    title_font = font.copy()
    title_font.set_size(18)
    ax.text(0.08, 0.93, title, fontproperties=title_font, weight="bold", va="top")
    y = 0.86
    body_font = font.copy()
    body_font.set_size(10.2)
    for line in _wrap_paragraphs(body, width=68):
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=A4)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.93
        ax.text(0.08, y, line, fontproperties=body_font, va="top")
        y -= 0.022 if line else 0.014
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, font: FontProperties) -> None:
    fig = plt.figure(figsize=A4)
    ax_title = fig.add_axes([0.08, 0.91, 0.84, 0.06])
    ax_title.axis("off")
    title_font = font.copy()
    title_font.set_size(16)
    ax_title.text(0, 0.7, title, fontproperties=title_font, weight="bold", va="top")
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    ax.axis("off")
    if image_path.exists():
        img = mpimg.imread(image_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Missing image: {image_path}", ha="center", va="center", fontproperties=font)
    pdf.savefig(fig)
    plt.close(fig)


def add_two_images_page(
    pdf: PdfPages,
    title: str,
    first_image_path: Path,
    first_caption: str,
    second_image_path: Path,
    second_caption: str,
    font: FontProperties,
) -> None:
    fig = plt.figure(figsize=A4)
    ax_title = fig.add_axes([0.08, 0.93, 0.84, 0.04])
    ax_title.axis("off")
    title_font = font.copy()
    title_font.set_size(16)
    ax_title.text(0, 0.8, title, fontproperties=title_font, weight="bold", va="top")
    for index, (image_path, caption) in enumerate(
        [(first_image_path, first_caption), (second_image_path, second_caption)]
    ):
        top = 0.51 if index == 0 else 0.10
        ax = fig.add_axes([0.08, top, 0.84, 0.34])
        ax.axis("off")
        if image_path.exists():
            img = mpimg.imread(image_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing image: {image_path}", ha="center", va="center", fontproperties=font)
        caption_ax = fig.add_axes([0.08, top - 0.03, 0.84, 0.025])
        caption_ax.axis("off")
        caption_ax.text(0.5, 0.5, caption, ha="center", va="center", fontproperties=font)
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_page(pdf: PdfPages, summary_path: Path, font: FontProperties) -> None:
    fig = plt.figure(figsize=A4)
    title_font = font.copy()
    title_font.set_size(16)
    fig.text(0.08, 0.94, "结果汇总表", fontproperties=title_font, weight="bold", va="top")
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        name_map = {
            "dmsaformer": "DMSAFormer",
            "hybrid": "HybridTCNTransformer",
            "lstm": "LSTM",
            "transformer": "Transformer",
        }
        for index, horizon in enumerate([90, 365]):
            ax = fig.add_axes([0.06, 0.52 - index * 0.42, 0.88, 0.32])
            ax.axis("off")
            subset = summary[summary["Horizon"].astype(int) == horizon].copy()
            subset = subset.sort_values("MSE mean")
            rows = [
                [
                    name_map.get(str(row["Model"]), str(row["Model"])),
                    f"{float(row['MSE mean']):.2f} ± {float(row['MSE std']):.2f}",
                    f"{float(row['MAE mean']):.2f} ± {float(row['MAE std']):.2f}",
                    str(int(row["Runs"])),
                ]
                for _, row in subset.iterrows()
            ]
            ax.text(0, 1.02, f"{horizon} 天预测结果", fontproperties=font, weight="bold", va="bottom")
            if rows:
                table = ax.table(
                    cellText=rows,
                    colLabels=["Model", "MSE mean ± std", "MAE mean ± std", "Runs"],
                    loc="center",
                    cellLoc="center",
                    colWidths=[0.30, 0.30, 0.30, 0.10],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.2)
                table.scale(1, 1.6)
            else:
                ax.text(0.5, 0.5, f"No {horizon}-day results", ha="center", va="center", fontproperties=font)
    else:
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
        ax.axis("off")
        ax.text(0.5, 0.5, f"Missing summary: {summary_path}", ha="center", va="center", fontproperties=font)
    pdf.savefig(fig)
    plt.close(fig)


def _read_summary(summary_path: Path) -> pd.DataFrame:
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame(columns=["Model", "Horizon", "MSE mean", "MSE std", "MAE mean", "MAE std", "Runs"])


def _display_model(name: str) -> str:
    return {
        "dmsaformer": "DMSAFormer",
        "hybrid": "HybridTCNTransformer",
        "lstm": "LSTM",
        "transformer": "Transformer",
    }.get(str(name), str(name))


def _best_rows(summary: pd.DataFrame, horizon: int) -> tuple[pd.Series | None, pd.Series | None]:
    """Return (DMSAFormer row, best-overall row) for a horizon, or (None, None)."""
    if summary.empty:
        return None, None
    subset = summary[summary["Horizon"].astype(int) == horizon]
    if subset.empty:
        return None, None
    dmsa_rows = subset[subset["Model"].astype(str) == "dmsaformer"]
    dmsa = dmsa_rows.iloc[0] if not dmsa_rows.empty else None
    best = subset.sort_values("MSE mean").iloc[0]
    return dmsa, best


def _pct_reduction(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return (old_value - new_value) / old_value * 100.0


def _rank_of(summary: pd.DataFrame, horizon: int, model: str) -> tuple[int, int]:
    """1-indexed rank of ``model`` by MSE mean at ``horizon`` and the field size."""
    subset = summary[summary["Horizon"].astype(int) == horizon].sort_values("MSE mean")
    ordered = [str(m) for m in subset["Model"].tolist()]
    total = len(ordered)
    rank = ordered.index(model) + 1 if model in ordered else 0
    return rank, total


def _improvement_sentence(summary: pd.DataFrame, horizon: int) -> str:
    """Honest ranking sentence: state exactly where DMSAFormer lands, no spin."""
    dmsa, best = _best_rows(summary, horizon)
    if dmsa is None or best is None:
        return f"{horizon} 天任务的相对排名需以 summary.csv 为准。"
    best_name = _display_model(best["Model"])
    rank, total = _rank_of(summary, horizon, "dmsaformer")
    rank_zh = {1: "第一", 2: "第二", 3: "第三"}.get(rank, f"第 {rank}")
    if str(best["Model"]) == "dmsaformer":
        return (
            f"{horizon} 天任务中，DMSAFormer 取得三模型中最低的 MSE 与 MAE。"
        )
    mse_gap = _pct_reduction(float(best["MSE mean"]), float(dmsa["MSE mean"]))
    mae_gap = _pct_reduction(float(best["MAE mean"]), float(dmsa["MAE mean"]))
    return (
        f"{horizon} 天任务中最强模型是 {best_name}；DMSAFormer 在三模型中排名{rank_zh}"
        f"（共 {total} 个），未超过最强 baseline，其 MSE 比 {best_name} 高 {abs(mse_gap):.2f}%，"
        f"MAE 高 {abs(mae_gap):.2f}%。"
    )


def _stability_sentence(summary: pd.DataFrame, horizon: int) -> str:
    if summary.empty:
        return f"{horizon} 天任务的 seed 稳定性需以 summary.csv 的 std 列为准。"
    subset = summary[summary["Horizon"].astype(int) == horizon].copy()
    if subset.empty:
        return f"{horizon} 天任务的 seed 稳定性需以 summary.csv 的 std 列为准。"
    most_stable = subset.sort_values("MSE std").iloc[0]
    least_stable = subset.sort_values("MSE std").iloc[-1]
    return (
        f"{horizon} 天任务中，MSE std 最小的是 {_display_model(most_stable['Model'])} "
        f"({float(most_stable['MSE std']):.2f})，最大的是 {_display_model(least_stable['Model'])} "
        f"({float(least_stable['MSE std']):.2f})。std 反映不同随机种子下结果波动，"
        "长期预测中复杂模型的 std 更大通常意味着训练样本不足、预测跨度长和尺度偏移会放大不稳定性。"
    )


def build_report_text_pages(summary_path: str | Path) -> list[ReportTextPage]:
    summary = _read_summary(Path(summary_path))
    improvement_90 = _improvement_sentence(summary, 90)
    improvement_365 = _improvement_sentence(summary, 365)
    stability_90 = _stability_sentence(summary, 90)
    stability_365 = _stability_sentence(summary, 365)

    return [
        ReportTextPage(
            "基于深度学习的家庭电力消耗多变量时间序列预测",
            "课程项目 PDF 报告\n\n"
            "GitHub: https://github.com/Mengtaigu111/jiqixuexi\n\n"
            "作者贡献与研究领域：本文由本人独立完成，主要贡献包括数据预处理、模型实现、实验设计、结果分析与报告撰写。研究领域为机器学习与时间序列预测。\n\n"
            "工具使用说明：报告撰写、格式整理与审查使用 AI 工具辅助；实验结果来自本项目真实代码运行、CSV 指标汇总和图表导出。\n\n"
            "实验概况：项目使用 UCI Individual household electric power consumption 原始分钟级数据，正式比较 LSTM、Transformer 和 DMSAFormer 三个模型在 90 天和 365 天预测任务上的 5 个 seed 结果，共 30 次正式对比实验。主结果表统一采用未做任何后处理校准的原始 test 指标，三个模型完全同一口径对比。DMSAFormer 是本文提出的分解式多尺度改进模型；HybridTCNTransformer 只作为 DMSAFormer 改进过程中的中间结构和消融对照，不进入主结果表。诚实结论：在公平对比下 DMSAFormer 两个 horizon 均未超过最强 baseline（90 天最优为 Transformer，365 天最优为 LSTM），本文如实报告这一结果并分析原因。",
        ),
        ReportTextPage(
            "1. 问题介绍",
            "家庭电力消耗预测可服务于智能家居能耗管理、居民用电行为分析、需求侧响应和智能电网调度。与普通表格回归不同，家庭用电序列同时受季节、天气、工作日/周末、家庭活动模式、设备启停和异常事件影响，表现出多变量、非平稳、强噪声和局部尖峰并存的时间序列特征。\n\n"
            "本项目使用 UCI Individual household electric power consumption 数据集。原始数据为分钟级记录，核心目标变量是 global_active_power，即家庭总有功功率。为了完成课程要求中的长期负荷预测，预处理阶段将分钟级数据聚合为日级序列，再用过去 90 天的多变量历史窗口预测未来 90 天或未来 365 天的 global_active_power 曲线。输入张量为 [N, 90, 19]，输出张量为 [N, 90] 或 [N, 365]。\n\n"
            "设置 90 天和 365 天两个 horizon 是为了同时考察短期/中期波动建模和长期趋势外推能力。90 天任务更关注局部周期和近期负荷形态，365 天任务更容易受到季节漂移、样本数量减少和误差累积影响，因此难度更高。模型均采用直接多步预测，一次输出完整未来序列，避免递归预测把前一步误差继续输入下一步。\n\n"
            "评价指标为 MSE 和 MAE。MSE = (1/n) * sum((y_i - y_hat_i)^2)，对较大误差和尖峰偏差更敏感；MAE = (1/n) * sum(|y_i - y_hat_i|)，更直观地反映平均绝对偏差。两者均在反标准化后的原尺度上计算，并对 5 个随机种子汇总 mean 和 std。",
        ),
        ReportTextPage(
            "1.1 数据预处理与样本构造",
            "原始分钟级数据首先解析 Date/Time 字段并按时间排序，\"?\"、空字符串、非法数字等标记统一转为 NaN。核心电力传感器列包括 global_active_power、global_reactive_power、voltage、global_intensity、sub_metering_1、sub_metering_2 和 sub_metering_3。对这些连续传感器列，脚本采用时间序列前向填充和后向填充；如果仍存在不可恢复缺失，则直接报错，而不是把缺测功率解释为真实 0。\n\n"
            "日级聚合遵循变量物理含义：global_active_power、global_reactive_power 和三个 sub_metering 分表能耗按天求和；voltage 和 global_intensity 按天取平均；可选天气变量 RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU 按日期对齐并取当天第一个可用值，天气字段缺失时只作为可选占位字段补 0。脚本还计算 sub_metering_remainder = global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3，用于表示未被三个分表覆盖的剩余用电量。\n\n"
            "特征工程加入 day_of_week、month、day_of_year、is_weekend，以及 day_of_year 的 sin/cos 周期编码。最终每一天包含 19 个输入特征。标准化器只在 train 划分上 fit，valid/test 只 transform，避免未来信息泄漏。之后使用滑动窗口构造样本：每个样本取连续 90 天特征作为 X，紧随其后的 90 天或 365 天 global_active_power 作为 y。90 天和 365 天任务分别构建 Dataset，模型也分别训练，长期模型不复用短期模型参数。",
        ),
        ReportTextPage(
            "1.2 实验设置与泄漏控制",
            "正式数据来自 UCI 原始分钟级文件，聚合后覆盖 2006-12-16 到 2010-11-26。窗口构造后，90 天任务的 train/valid/test 样本数为 353/366/366，365 天任务为 78/91/91。365 天任务样本更少，是长期预测更困难的重要原因之一。\n\n"
            "三个正式模型使用相同数据划分、相同输入长度和相同评价脚本。训练设置为 seeds=2026、2027、2028、2029、2030，epoch=30，early stopping patience=8，loss=MSELoss，optimizer=AdamW，learning rate=1e-3。LSTM、Transformer 和 DMSAFormer 都在 90 天与 365 天两个输出长度上分别训练和评估。\n\n"
            "数据泄漏控制主要体现在三点：第一，StandardScaler 只在 train 上拟合；第二，valid 只用于 early stopping 与可选的 affine 校准参数估计；第三，test set 只在最终评估阶段使用，不参与权重训练、参数拟合或模型选择。主结果表中的三个模型都直接报告未经任何后处理的原始 test MSE/MAE，因此对比是完全对称的。此外，本文在消融部分给出一个额外的公平校准实验：对三个模型施加完全相同的 validation-only affine 校准 y = a * pred + b，其中 a、b 只在 valid 预测上拟合、只应用于 test 预测，绝不接触 test 标签。该实验用于说明校准对每个模型的独立影响，而不是用来单独抬高某一个模型。",
        ),
        ReportTextPage(
            "2. 基线模型方法",
            "LSTM 模型使用两层 LSTM 编码过去 90 天的多变量序列，hidden size=64，dropout=0.1。模型取最后一层最终隐状态作为历史窗口表示，经 LayerNorm、Linear、ReLU、Dropout 和 Linear 组成的 MLP 直接输出未来 output_len 天曲线。LSTM 的优势是顺序归纳偏置稳定，参数量适中；在样本较少的 365 天任务中，LSTM 取得三个模型里最低的 MSE 与 MAE，说明稳定的顺序归纳偏置在小样本长期预测中反而更可靠。\n\n"
            "Transformer 模型先把 19 维输入映射到 d_model=64，加入正弦位置编码，再使用 2 层 TransformerEncoder 建模日期间全局依赖。注意力头数为 4，feed-forward 维度为 128，最后对编码序列做 mean pooling 并输出未来曲线。Transformer 能并行捕获长距离关系，但在本项目数据规模下，365 天任务表现不稳定，说明标准自注意力并不必然适合小样本长 horizon 预测。\n\n"
            "这两个 baseline 分别代表顺序递归建模和全局注意力建模。后续 DMSAFormer 的改进过程曾引入 HybridTCNTransformer-style 局部时序主干作为内部结构探索，但它不作为正式第三个对比模型进入主结果表。",
        ),
        ReportTextPage(
            "2.1 DMSAFormer 方法细节",
            "DMSAFormer 是本文最终提出的改进模型，全称为 Decomposition-based Multi-Scale Attention Transformer。它不是简单堆叠模型名，而是围绕家庭用电序列的趋势、残差、变量贡献和长短期专家稳定性设计的组合型改进方法。\n\n"
            "趋势分解模块：使用 MovingAverage 将输入序列拆分为慢变化 trend 和 residual。家庭用电序列同时包含季节性趋势与短期设备启停波动，直接用单一编码器学习两类信号容易互相干扰。分解后，趋势分支更适合低频外推，残差分支更专注局部波动。\n\n"
            "多尺度卷积模块：残差分支使用 3、7、30 天不同 kernel 的卷积，对应短期波动、周级变化和月级模式。不同尺度的局部滤波可以弥补标准 Transformer 在小样本场景下缺少局部归纳偏置的问题。\n\n"
            "变量注意力模块：输入包含功率、电压、电流、分表、天气和时间特征，不同变量对 global_active_power 的贡献并不相同。VariableAttention 从整个 lookback window 学习 feature-wise gate，对重要变量加权，使模型能够根据样本动态调整输入特征贡献。\n\n"
            "局部时序主干：神经网络内包含 target decomposition backbone、HybridTCNTransformer-style local backbone 和 residual Transformer correction branch。分解线性主干降低长期尺度方差，局部主干补充 TCN/Transformer 的短期模式提取能力。该设计借鉴 Autoformer 的分解归纳偏置和 DLinear 的分解线性思想，并在参考文献中标注来源。",
        ),
        ReportTextPage(
            "2.2 公平对比口径与校准消融",
            "正式主表采用最严格也最无争议的公平口径：LSTM、Transformer 和 DMSAFormer 三个模型全部报告未经任何后处理的原始 test MSE/MAE。这样三个模型完全站在同一起跑线上，避免只对某一个模型施加额外校准而制造不对称优势。\n\n"
            "作为附加消融，本文还考察了一种 validation-only affine 校准：每个 seed 和 horizon 先由 checkpoint 生成 validation/test 预测，只在 validation 预测与 validation 标签上拟合全局参数 y = a * pred + b（test 只用于最终评估，无泄漏），再把同一套参数应用到 test 预测。关键在于这套校准对三个模型一视同仁地施加，用来观察校准各自贡献多少，而不是只用来抬高 DMSAFormer。\n\n"
            "消融结论很明确：affine 校准对三个模型在多数设置下都能小幅降低误差（尤其 365 天任务），但它不改变模型之间的相对排名——90 天仍是 Transformer 最优，365 天仍是 LSTM 最优。因此 DMSAFormer 早期报告中\"365 天夺冠\"的现象，实为只校准 DMSAFormer、不校准 baseline 造成的假象，本版已纠正。完整的 raw 与 calibrated 对照见 results/metrics/fair_comparison_summary.csv。\n\n"
            "DMSAFormer 前向伪代码如下：\n"
            "Input: X in R^{90 x 19}\n"
            "1. Center target channel by its window mean (instance norm)\n"
            "2. Decompose centered target into trend and residual\n"
            "3. Map trend/residual to horizon by linear heads, add level back\n"
            "4. Fuse LSTM and TCN-Transformer experts as a small correction\n"
            "5. Add multi-scale (3/7/30-day) attentive correction, gate-controlled\n"
            "6. Report raw test MSE/MAE (calibration only used in the ablation)",
        ),
        ReportTextPage(
            "3. 结果与分析",
            "三模型主表全部使用原始（未校准）test 指标。90 天任务的最低 MSE/MAE 来自 Transformer，LSTM 次之，DMSAFormer 排第三；365 天任务的最低 MSE/MAE 来自 LSTM，DMSAFormer 次之，Transformer 最差。也就是说，本文提出的 DMSAFormer 在两个 horizon 上都没有超过最强 baseline，两个任务的冠军分别是 Transformer（短期）和 LSTM（长期）。我们如实报告这一结果，而不做任何有利于自身模型的口径修饰。\n\n"
            f"相对差距：{improvement_90} {improvement_365} 需要特别说明的是，早期草稿曾出现\"DMSAFormer 在 365 天夺冠\"的结论，但那是因为当时只对 DMSAFormer 施加了 validation affine 校准、却没有对 LSTM/Transformer 施加同样处理。一旦把相同校准公平地施加到三个模型（见 fair_comparison 消融），LSTM 在 365 天仍然领先，该\"夺冠\"结论随即消失。本版主表已改为三模型统一 raw 口径以杜绝此类不对称。\n\n"
            f"稳定性分析：{stability_90} {stability_365} 报告中的 std 不能只作为表格附属列，应理解为模型对随机初始化和训练批次扰动的敏感程度。365 天任务中 Transformer 与 DMSAFormer 的 std 都明显偏大，说明复杂模型在小样本长期预测中更容易出现不稳定，而 LSTM 相对更稳。\n\n"
            "Transformer 在 365 天任务中误差最高，可能原因包括训练样本只有 78 个、预测跨度长、标准自注意力缺少局部平滑约束，以及直接输出 365 个点时更容易发生整体尺度漂移。DMSAFormer 虽然引入了分解式低方差主干，365 天误差低于 Transformer，但仍未追平结构更简单的 LSTM，说明在当前数据规模下额外的多分支结构并没有转化为真正的泛化优势。",
        ),
        ReportTextPage(
            "3.1 预测曲线分析",
            "预测曲线不是只用于展示结果截图，也可以帮助解释模型误差来源。90 天预测曲线中，三个正式模型整体能跟随 Ground Truth 的中低频趋势；DMSAFormer 的 seed 间波动较大（MSE std 明显高于 LSTM），因此主指标没有超过 Transformer 甚至没有超过 LSTM。\n\n"
            "365 天预测曲线中，模型仍能抓住年度尺度的总体水平变化，但对局部尖峰和突发波动响应不足。尖峰负荷响应不足的一个原因是 MSE/MAE 损失倾向于学习平均趋势，而尖峰样本在训练集中占比较低；另一个原因是日级聚合会平滑分钟级设备启停信息，使模型更擅长预测总体水平而非短时异常。\n\n"
            "从预测曲线可以看出，模型更擅长拟合整体趋势和中低频变化，但对短期突发尖峰负荷响应不足。这也是后续可以引入异常检测、分位数损失、峰值加权损失或概率预测区间的原因。",
        ),
        ReportTextPage(
            "3.4 诊断分析一：朴素基线地板与参数量代价",
            "本文提出的 DMSAFormer 在两个 horizon 上都没有超过 baseline，但这一结果本身值得深入诊断，因为它揭示了任务本身的特性。第一步是建立朴素基线地板：持续法（用输入窗口最后一天外推）、窗口均值（用过去 90 天均值外推）和周季节 naive（按 7 天周期重复）。这些方法完全不训练，代表\"零智能\"下限。\n\n"
            "结果显示，最强深度模型相对最好的朴素基线：90 天任务中 Transformer 的 MSE 为 156632，最好的朴素基线（窗口均值）为 354862，深度模型降低约 55.9%；365 天任务中 LSTM 的 MSE 为 316352，最好的朴素基线（周季节 naive）为 396030，深度模型只降低约 20.1%。这个对比很关键：深度模型确实学到了真实结构，但 365 天任务里深度模型相对朴素地板的优势（20%）远小于 90 天（56%），说明长期任务的可预测成分本就稀薄，留给复杂模型拉开差距的空间很小。这直接解释了为什么 365 天任务中三个模型挤在一起、结构差异难以转化为显著性能差异。\n\n"
            "第二个诊断是参数量与收益。DMSAFormer 的可训练参数量为 90 天 393794、365 天 532669，而 Transformer 仅 88282、LSTM 仅 83053。也就是说 DMSAFormer 用了 4 到 6 倍的参数量，90 天仍比 Transformer 高 6.4% MSE，365 天仍比 LSTM 高 10.2% MSE。\"更大的模型换来更差的结果\"是过拟合的典型信号，尤其当 365 天训练窗口只有 78 个样本时，高容量模型难以被如此少的数据充分约束。",
        ),
        ReportTextPage(
            "3.5 诊断分析二：过拟合缺口与逐模块消融",
            "第三个诊断直接测量过拟合：用每个模型训练日志里最终 train loss 与最优 valid MSE 的比值（均在标准化尺度上，比值越大越过拟合）。365 天任务中，DMSAFormer 的 valid/train 比值为 2.53，是三者中最高；它把 train loss 压到 0.297（最低，说明拟合训练集最狠），valid MSE 却停在 0.751（最差之一）。对比之下 LSTM 比值 1.65、Transformer 1.90。DMSAFormer 在极少样本上把训练损失学得最低、泛化却最差，与参数量诊断形成闭环：高容量在小样本长期任务上转化为过拟合而非泛化。\n\n"
            "第四个诊断是逐模块消融：以完整 DMSAFormer 为基准，每次关闭一个模块（实例归一化、趋势分解、变量门控、多尺度注意力修正），用当前架构从头重训、5 个 seed 汇总原尺度 test MSE。需要说明，消融为保证一致性统一使用当前 DMSAFormer 类从零重训，其\"完整模型\"数值（90 天 192194、365 天 428501）与主表中基于已保存 checkpoint 的 DMSAFormer 不完全相同，因此消融只用于模块间的相对比较，不替代主表数字。\n\n"
            "消融揭示模块价值高度依赖 horizon。365 天任务中，完整模型是所有变体里最好的（428501），去掉任一模块都变差，其中去掉趋势分解最严重（跳到 501846），说明趋势/残差分解是长期预测最关键的归纳偏置。90 天任务则相反：去掉实例归一化反而最好（169213 优于完整模型的 192194），说明短期预测中窗口均值中心化会抹掉有用的绝对水平信息，而多尺度注意力修正在短期是有帮助的（去掉它误差升到 195328）。综合来看，DMSAFormer 的问题不是某个模块\"坏\"，而是它把一组彼此价值随 horizon 变化、甚至相互冲突的模块固定组合在一起，导致在任一 horizon 上都不是最优配置——这正是组合型模型在小样本任务上的典型陷阱。",
        ),
        ReportTextPage(
            "4. 讨论与后续工作",
            "本方法的主要优点是保持了完整可复现链路：从原始数据清洗、日级聚合、滑动窗口 Dataset、三模型正式训练与评估、原尺度 mean/std 汇总到 PDF/Word 报告生成都由脚本完成，且主表三个模型使用完全对称的 raw 评价口径，不存在只对某一模型施加后处理的问题。\n\n"
            "本项目最重要也最诚实的发现是：在当前数据规模下，本文提出的 DMSAFormer 并没有超过两个 baseline——90 天最优是 Transformer，365 天最优是 LSTM，DMSAFormer 在两个 horizon 上都居中。早期草稿曾出现\"DMSAFormer 在 365 天夺冠\"的结论，经复核那是由于只对 DMSAFormer 施加 validation-affine 校准、而 baseline 未校准造成的不对称假象；在对三个模型施加同一套校准后（见 fair_comparison 消融），排名不变，该结论被推翻并在本版中纠正。\n\n"
            "为什么改进模型没有赢，是更值得讨论的问题。其一，DMSAFormer 是工程组合型模型（分解主干 + LSTM/TCN-Transformer 专家 + 多尺度注意力修正），参数量和结构复杂度都更高，而 365 天任务训练样本只有 78 个，复杂模型在如此小的样本上更容易过拟合、seed 间方差更大（其 365 天 MSE std 是三者中最高）。其二，家庭日级用电的可预测成分以低频趋势为主，简单的顺序或线性归纳偏置（LSTM、乃至 DLinear 式分解）已足以捕捉，额外的注意力修正带来的收益不足以抵消其方差代价。其三，90 天任务中 Transformer 胜出说明在样本相对充足时全局注意力有效，但这一优势没有迁移到样本更稀缺的 365 天。\n\n"
            "365 天比 90 天更难，原因包括输出维度更高、训练样本更少、季节性和生活习惯漂移更强、远期误差更容易形成整体尺度偏移。需要特别指出的一个方法学局限：test_365 只有 91 个高度重叠的窗口（相邻窗口 365 天中有 364 天相同），本质上接近单条长轨迹，因此 365 天的 std 主要反映初始化随机性而非测试采样不确定性，长期结论应谨慎对待。\n\n"
            "后续可以从几个方向改进：为改进模型引入更强的正则化或更小的容量以匹配小样本长期任务；引入更细粒度天气和节假日特征；比较 Informer、Autoformer、PatchTST、TimesNet 等长序列模型；加入峰值加权损失或分位数损失；做显式异常检测和缺失 mask；并通过更长的数据或滚动 origin 评估来获得更可靠的长期预测置信区间。",
        ),
        ReportTextPage(
            "参考文献",
            "[1] UCI Machine Learning Repository. Individual household electric power consumption Data Set. https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption\n"
            "[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.\n"
            "[3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.\n"
            "[4] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.\n"
            "[5] Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS.\n"
            "[6] Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.\n"
            "[7] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR.\n"
            "[8] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? AAAI.\n"
            "[9] PyTorch Documentation. LSTM and TransformerEncoder. https://pytorch.org/docs/stable/nn.html\n"
            "[10] scikit-learn Documentation. StandardScaler. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
        ),
    ]


def generate_pdf(report_dir: str | Path = "report", results_dir: str | Path = "results") -> Path:
    report_dir = Path(report_dir)
    results_dir = Path(results_dir)
    ensure_dir(report_dir)
    font_path = report_dir / "report_assets" / "NotoSansCJKsc-Regular.otf"
    if not font_path.exists():
        font_path = Path(__file__).resolve().parents[1] / "report" / "report_assets" / "NotoSansCJKsc-Regular.otf"
    font = _font(font_path)
    out_path = report_dir / "ML_household_power_report.pdf"
    summary_path = results_dir / "metrics" / "summary.csv"
    figures_dir = results_dir / "figures"

    with PdfPages(out_path) as pdf:
        text_pages = build_report_text_pages(summary_path)
        # Pages 0..8: cover + intro through 3.1 prediction-curve analysis.
        for page in text_pages[:9]:
            add_text_page(pdf, page.title, page.body, font)
        add_summary_page(pdf, summary_path, font)
        add_two_images_page(
            pdf,
            "MSE 与 MAE 指标对比",
            figures_dir / "metric_bar_mse.png",
            "MSE 越低表示大误差和尖峰偏差越小",
            figures_dir / "metric_bar_mae.png",
            "MAE 越低表示平均绝对偏差越小",
            font,
        )
        add_image_page(pdf, "90 天预测曲线对比", figures_dir / "prediction_comparison_90.png", font)
        add_image_page(pdf, "365 天预测曲线对比", figures_dir / "prediction_comparison_365.png", font)
        # Pages 9..10: the two diagnostic text pages (3.2 / 3.3).
        for page in text_pages[9:11]:
            add_text_page(pdf, page.title, page.body, font)
        # Diagnostic figures backing the 3.2 / 3.3 analysis.
        add_two_images_page(
            pdf,
            "诊断图 1-2：朴素基线地板与参数量代价",
            figures_dir / "diag_naive_floor.png",
            "深度模型显著优于朴素地板，但 365 天优势远小于 90 天",
            figures_dir / "diag_capacity_vs_error.png",
            "DMSAFormer 参数量最大却非最优",
            font,
        )
        add_two_images_page(
            pdf,
            "诊断图 3-4：过拟合缺口与逐模块消融",
            figures_dir / "diag_overfitting_gap.png",
            "365 天 DMSAFormer 验证/训练比最高，过拟合最严重",
            figures_dir / "diag_ablation.png",
            "当前架构重训下各模块对 test MSE 的相对贡献",
            font,
        )
        # Remaining pages: discussion + references.
        for page in text_pages[11:]:
            add_text_page(pdf, page.title, page.body, font)
    shutil.copyfile(out_path, report_dir / "report.pdf")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF draft report from summary and figures.")
    parser.add_argument("--report_dir", default="report")
    parser.add_argument("--results_dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = generate_pdf(args.report_dir, args.results_dir)
    print(f"Saved PDF report: {out_path}")


if __name__ == "__main__":
    main()
