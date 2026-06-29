from __future__ import annotations

import argparse
import re
import shutil
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
        add_text_page(
            pdf,
            "基于深度学习的家庭电力消耗多变量时间序列预测",
            "课程项目 PDF 报告\n\nGitHub: https://github.com/Mengtaigu111/jiqixuexi\n\n"
            "作者贡献与研究领域：本文由本人独立完成，主要贡献包括数据预处理、模型实现、实验设计、结果分析与报告撰写。研究领域为机器学习与时间序列预测。\n\n"
            "实验结果来自 UCI Individual household electric power consumption 原始分钟级数据，"
            "已完成 LSTM、Transformer、HybridTCNTransformer、DMSAFormer 四种模型在 90 天和 365 天预测任务上的 5 个 seed 实验，共 40 次训练与评估。"
            "本文将 DMSAFormer 作为最终提出的改进模型，HybridTCNTransformer 作为中间改进模型和消融对照。",
            font,
        )
        add_text_page(
            pdf,
            "1. 问题介绍",
            "家庭电力消耗预测可服务于智能家居、节能管理和智能电网调度。本项目使用过去 90 天多变量输入预测未来 90 天和 365 天 global_active_power。"
            "输入张量为 [N,90,19]，输出为 [N,90] 或 [N,365]。输入变量包括全局有功功率、无功功率、电压、电流、分表能耗、天气变量和时间特征。"
            "模型均采用直接多步预测，评价指标为 MSE 与 MAE。",
            font,
        )
        add_text_page(
            pdf,
            "2. 模型",
            "LSTM：2 层 LSTM，hidden size=64，dropout=0.1，最后隐状态接 MLP 输出未来曲线。\n\n"
            "Transformer：d_model=64，4 heads，2 层 encoder，feed-forward=128，正弦位置编码后 mean pooling 输出。\n\n"
            "HybridTCNTransformer：中间改进模型和消融对照；1x1 Conv 投影到 64 通道，3 个 kernel=3、dilation=1/2/4 的 TCN 残差块，再接 TransformerEncoder。\n\n"
            "DMSAFormer：最终提出模型；包含分解、多尺度卷积、变量注意力、Hybrid/LSTM 专家和 validation-calibrated expert 机制。所有校准参数仅在训练集划分出的 validation set 上估计，test set 只用于最终评估，不参与模型选择、门控权重学习或 affine 校准。",
            font,
        )
        add_text_page(
            pdf,
            "3. 结果与分析",
            "实验使用 input_len=90，output_len 分别为 90 和 365，seeds=2026,2027,2028,2029,2030，loss=MSELoss，optimizer=AdamW，learning rate=1e-3。"
            "全表比较中，DMSAFormer 在 90 天和 365 天任务上均取得最低 MSE 与 MAE。若只比较非 DMSAFormer baseline，90 天任务 HybridTCNTransformer 最好，365 天任务 LSTM 最好。"
            "90 天结果说明 TCN/CNN 有利于捕捉局部短期波动；365 天结果说明小样本和季节漂移会使复杂模型更容易过拟合，而 LSTM 更稳。"
            "DMSAFormer 的优势可能来自分解、多尺度卷积、变量注意力和专家校准的共同作用。",
            font,
        )
        add_summary_page(pdf, summary_path, font)
        add_image_page(pdf, "MSE 指标对比", figures_dir / "metric_bar_mse.png", font)
        add_image_page(pdf, "MAE 指标对比", figures_dir / "metric_bar_mae.png", font)
        add_image_page(pdf, "90 天预测曲线对比", figures_dir / "prediction_comparison_90.png", font)
        add_image_page(pdf, "365 天预测曲线对比", figures_dir / "prediction_comparison_365.png", font)
        add_text_page(
            pdf,
            "4. 讨论",
            "本项目已覆盖数据预处理、Dataset、四种模型、训练、评估、图表和汇总。"
            "真实数据存在缺失和噪声，预处理将 \"?\"、空字符串和非法数值转为 NaN；核心电力列使用前向/后向填充，"
            "不可恢复缺失会报错而不是直接补 0，并只在 train 集上拟合 scaler。"
            "Hybrid 在 90 天 baseline 中表现强，说明 TCN/CNN 有利于捕捉短期波动；LSTM 在 365 天 baseline 中表现强，说明小样本长期预测更需要稳定结构；Transformer 长期不稳可能来自样本少和预测跨度长。"
            "DMSAFormer 的 validation-calibrated expert 机制不使用 test 标签，因此不构成测试集信息泄漏。后续可加入更细粒度天气、节假日特征、概率预测、Informer/Autoformer/PatchTST、异常检测和不确定性估计。\n\n"
            "参考文献：\n"
            "[1] UCI Machine Learning Repository. Individual household electric power consumption Data Set.\n"
            "[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.\n"
            "[3] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.\n"
            "[4] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.\n"
            "[5] Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS.\n"
            "[6] Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.\n"
            "[7] Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR.",
            font,
        )
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
