"""Render diagnostic figures for the report's "why the improved model does not win" analysis.

All figures are built from the confirmed on-disk CSV artifacts produced by
``src.diagnostics`` and ``src.ablation_dmsaformer``:

* ``naive_floor_summary.csv``       - persistence / window-mean / seasonal-naive floors.
* ``capacity_summary.csv``          - trainable params vs test MSE for the three models.
* ``overfitting_gap_summary.csv``   - final train loss, best valid MSE, valid/train ratio.
* ``ablation_dmsaformer_summary.csv`` - per-module ablation of the current DMSAFormer.

Outputs (into ``results/figures``, also mirrored into ``results/screenshots``):

* ``diag_naive_floor.png``
* ``diag_capacity_vs_error.png``
* ``diag_overfitting_gap.png``
* ``diag_ablation.png``
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "report" / "report_assets" / "NotoSansCJKsc-Regular.otf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    matplotlib.rcParams["font.family"] = "Noto Sans CJK SC"
matplotlib.rcParams["axes.unicode_minus"] = False

MODEL_DISPLAY = {"dmsaformer": "DMSAFormer", "lstm": "LSTM", "transformer": "Transformer"}
MODEL_COLOR = {"dmsaformer": "#1f77b4", "lstm": "#2ca02c", "transformer": "#9467bd"}
NAIVE_DISPLAY = {
    "persistence": "持续法",
    "window_mean": "窗口均值",
    "seasonal_naive_7d": "周季节naive",
}


def _read(metrics_dir: Path, name: str) -> pd.DataFrame:
    path = metrics_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_naive_floor(metrics_dir: Path, figures_dir: Path) -> Path:
    naive = _read(metrics_dir, "naive_floor_summary.csv")
    capacity = _read(metrics_dir, "capacity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        labels, values, colors = [], [], []
        for _, r in capacity[capacity["horizon"] == horizon].sort_values("test_mse_mean").iterrows():
            labels.append(MODEL_DISPLAY[r["model"]])
            values.append(float(r["test_mse_mean"]))
            colors.append(MODEL_COLOR[r["model"]])
        for _, r in naive[naive["horizon"] == horizon].sort_values("test_mse").iterrows():
            labels.append(NAIVE_DISPLAY.get(r["method"], r["method"]))
            values.append(float(r["test_mse"]))
            colors.append("#b0b0b0")
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_title(f"{horizon} 天：深度模型 vs 朴素基线地板")
        ax.set_ylabel("test MSE (原尺度)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "diag_naive_floor.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_capacity(metrics_dir: Path, figures_dir: Path) -> Path:
    capacity = _read(metrics_dir, "capacity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        subset = capacity[capacity["horizon"] == horizon]
        for _, r in subset.iterrows():
            ax.scatter(
                float(r["trainable_params"]) / 1000.0,
                float(r["test_mse_mean"]),
                s=140,
                color=MODEL_COLOR[r["model"]],
                zorder=3,
            )
            ax.annotate(
                MODEL_DISPLAY[r["model"]],
                (float(r["trainable_params"]) / 1000.0, float(r["test_mse_mean"])),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
            )
        ax.set_title(f"{horizon} 天：参数量 vs 测试误差")
        ax.set_xlabel("可训练参数量 (千)")
        ax.set_ylabel("test MSE (原尺度)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "diag_capacity_vs_error.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_overfitting_gap(metrics_dir: Path, figures_dir: Path) -> Path:
    gap = _read(metrics_dir, "overfitting_gap_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    horizons = [90, 365]
    models = ["lstm", "transformer", "dmsaformer"]
    width = 0.25
    for idx, model in enumerate(models):
        vals = []
        for horizon in horizons:
            row = gap[(gap["model"] == model) & (gap["horizon"] == horizon)]
            vals.append(float(row["valid_over_train_ratio"].iloc[0]) if not row.empty else 0.0)
        xs = [i + (idx - 1) * width for i in range(len(horizons))]
        bars = ax.bar(xs, vals, width=width, color=MODEL_COLOR[model], label=MODEL_DISPLAY[model])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--", label="valid=train")
    ax.set_xticks(range(len(horizons)), [f"{h} 天" for h in horizons])
    ax.set_ylabel("best valid MSE / final train loss")
    ax.set_title("过拟合诊断：验证/训练损失比 (越高越过拟合)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figures_dir / "diag_overfitting_gap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_ablation(metrics_dir: Path, figures_dir: Path) -> Path:
    ablation = _read(metrics_dir, "ablation_dmsaformer_summary.csv")
    display = {
        "full": "完整模型",
        "no_correction": "去注意力修正",
        "no_decomposition": "去趋势分解",
        "no_window_norm": "去实例归一化",
        "no_variable_attention": "去变量门控",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        subset = ablation[ablation["horizon"] == horizon].sort_values("MSE mean")
        labels = [display.get(v, v) for v in subset["variant"]]
        values = subset["MSE mean"].to_numpy(dtype=float)
        colors = ["#1f77b4" if v == "full" else "#9ecae1" for v in subset["variant"]]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        ax.set_title(f"{horizon} 天：DMSAFormer 逐模块消融")
        ax.set_ylabel("test MSE (原尺度, 当前架构重训)")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = figures_dir / "diag_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_diagnostic_figures(
    metrics_dir: str | Path = "results/metrics",
    figures_dir: str | Path = "results/figures",
    screenshots_dir: str | Path = "results/screenshots",
) -> list[Path]:
    metrics_path = Path(metrics_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    screenshots_path = Path(screenshots_dir)
    screenshots_path.mkdir(parents=True, exist_ok=True)

    written = [
        plot_naive_floor(metrics_path, figures_path),
        plot_capacity(metrics_path, figures_path),
        plot_overfitting_gap(metrics_path, figures_path),
        plot_ablation(metrics_path, figures_path),
    ]
    for path in written:
        shutil.copy2(path, screenshots_path / path.name)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render diagnostic figures for the report analysis section.")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--screenshots_dir", default="results/screenshots")
    return parser.parse_args()


def main() -> None:
    for path in generate_diagnostic_figures(**vars(parse_args())):
        print(path)


if __name__ == "__main__":
    main()
