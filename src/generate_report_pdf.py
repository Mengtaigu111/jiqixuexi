from __future__ import annotations

import argparse
import re
import shutil
import textwrap
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


def _wrap_line(line: str, width: int = 42) -> list[str]:
    if not line.strip():
        return [""]
    if re.search(r"[\u4e00-\u9fff]", line):
        return [line[i : i + width] for i in range(0, len(line), width)]
    return textwrap.wrap(line, width=72) or [line]


def _wrap_paragraphs(text: str, width: int = 42) -> list[str]:
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
    body_font.set_size(10.5)
    for line in _wrap_paragraphs(body):
        if y < 0.07:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=A4)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.93
        ax.text(0.08, y, line, fontproperties=body_font, va="top")
        y -= 0.026 if line else 0.018
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
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
    ax.axis("off")
    title_font = font.copy()
    title_font.set_size(16)
    fig.text(0.08, 0.94, "结果汇总表", fontproperties=title_font, weight="bold", va="top")
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        table = ax.table(cellText=summary.values, colLabels=summary.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.45)
    else:
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
            "课程项目 PDF 报告草稿\n\nGitHub: https://github.com/Mengtaigu111/jiqixuexi\n\n"
            "作者贡献与研究领域：待填写。\n\n"
            "实验结果来自 UCI Individual household electric power consumption 原始分钟级数据，"
            "已完成 LSTM、Transformer、HybridTCNTransformer 三种模型在 90 天和 365 天预测任务上的 5 个 seed 实验，共 30 次训练与评估。",
            font,
        )
        add_text_page(
            pdf,
            "1. 问题介绍",
            "家庭电力消耗预测可服务于智能家居、节能管理和智能电网调度。本项目使用过去 90 天多变量输入预测未来 90 天和 365 天 global_active_power。"
            "输入变量包括全局有功功率、无功功率、电压、电流、分表能耗、天气变量和时间特征。评价指标为 MSE 与 MAE。",
            font,
        )
        add_text_page(
            pdf,
            "2. 模型",
            "LSTM 使用多层循环结构编码历史序列，并通过 MLP 多步预测头输出未来曲线。\n\n"
            "Transformer 将输入投影到 d_model，加入位置编码后用 TransformerEncoder 建模全局依赖。\n\n"
            "HybridTCNTransformer 先通过 TCN/CNN 残差块提取局部波动和周期模式，再用 Transformer 捕捉长期依赖，兼顾局部特征和全局关系。",
            font,
        )
        add_text_page(
            pdf,
            "3. 结果与分析",
            "实验使用 input_len=90，output_len 分别为 90 和 365，seeds=2026,2027,2028,2029,2030。"
            "90 天预测中 HybridTCNTransformer 的 MSE 和 MAE 最低，说明局部卷积/TCN 特征对短期波动建模有效。"
            "365 天预测中 LSTM 指标最好，Hybrid 次之，说明长期预测更受样本量、季节变化和误差累积影响，复杂模型不一定直接带来最低误差。",
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
            "本项目已覆盖数据预处理、Dataset、LSTM、Transformer、HybridTCNTransformer、训练、评估、图表和汇总。"
            "真实数据存在缺失和噪声，预处理将 ? 等非法标记转为 NaN；核心电力列使用前向/后向填充，"
            "不可恢复缺失会报错而不是直接补 0，并只在 train 集上拟合 scaler。"
            "后续可加入更细粒度天气、节假日特征、概率预测、Informer/Autoformer/PatchTST、异常检测和不确定性估计。\n\n"
            "参考文献：UCI Individual household electric power consumption；Hochreiter and Schmidhuber, 1997；Vaswani et al., 2017；PyTorch Documentation。",
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
