from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics"
PREDICTIONS = ROOT / "results" / "predictions"
MODELS = ["dmsaformer", "lstm", "transformer"]
DISPLAY = {
    "dmsaformer": "DMSAFormer",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
COLORS = {
    "dmsaformer": "#1f77b4",
    "lstm": "#2ca02c",
    "transformer": "#9467bd",
}
FONT_PATH = ROOT / "report" / "report_assets" / "NotoSansCJKsc-Regular.otf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    matplotlib.rcParams["font.family"] = "Noto Sans CJK SC"
matplotlib.rcParams["axes.unicode_minus"] = False

EVOLUTION_SUMMARIES = {
    "初版 DMSAFormer": ROOT / "results" / "archive" / "dmsaformer_v1_20260623T115415Z" / "summary.csv",
    "结构改造版": ROOT
    / "results"
    / "archive"
    / "dmsaformer_v2_before_calibration_20260623T122806Z"
    / "results_exports"
    / "summary.csv",
    "自身校准版": ROOT / "results" / "summary.csv",
}


def read_metric_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS.glob("*_test_metrics.csv")):
        frame = pd.read_csv(path)
        if {"model", "output_len", "seed", "test_mse", "test_mae"}.issubset(frame.columns):
            rows.append(frame[["model", "output_len", "seed", "test_mse", "test_mae"]])
    if not rows:
        raise FileNotFoundError("No test metric CSV files found.")
    return pd.concat(rows, ignore_index=True)


def read_prediction_errors() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for horizon in [90, 365]:
            for path in sorted(PREDICTIONS.glob(f"{model}_{horizon}_seed*.csv")):
                frame = pd.read_csv(path, usecols=["model", "output_len", "seed", "step", "y_true", "y_pred"])
                frame["abs_error"] = (frame["y_pred"] - frame["y_true"]).abs()
                rows.append(frame[["model", "output_len", "seed", "step", "abs_error"]])
    if not rows:
        raise FileNotFoundError("No prediction CSV files found.")
    return pd.concat(rows, ignore_index=True)


def plot_per_seed_mse(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160, sharey=False)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        subset = metrics[metrics["output_len"] == horizon]
        for model in MODELS:
            model_rows = subset[subset["model"] == model].sort_values("seed")
            ax.plot(
                model_rows["seed"],
                model_rows["test_mse"],
                marker="o",
                linewidth=2,
                color=COLORS[model],
                label=DISPLAY[model],
            )
        ax.set_title(f"{horizon}-day per-seed MSE")
        ax.set_xlabel("Seed")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "report_per_seed_mse.png")
    plt.close(fig)


def plot_improvement(summary: pd.DataFrame) -> dict[str, float]:
    rows = []
    stats: dict[str, float] = {}
    for horizon in [90, 365]:
        subset = summary[summary["Horizon"] == horizon]
        dmsa = subset[subset["Model"] == "dmsaformer"].iloc[0]
        baselines = subset[subset["Model"] != "dmsaformer"]
        for metric in ["MSE", "MAE"]:
            col = f"{metric} mean"
            best = baselines[col].min()
            improvement = (best - dmsa[col]) / best * 100.0
            rows.append({"horizon": str(horizon), "metric": metric, "improvement": improvement})
            stats[f"{horizon}_{metric.lower()}_improvement_pct"] = float(improvement)
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    x = np.arange(2)
    width = 0.34
    for idx, metric in enumerate(["MSE", "MAE"]):
        vals = frame[frame["metric"] == metric].sort_values("horizon")["improvement"].to_numpy()
        bars = ax.bar(x + (idx - 0.5) * width, vals, width=width, label=metric)
        ax.bar_label(bars, fmt="%.2f%%", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, ["90-day", "365-day"])
    ax.set_ylabel("Reduction vs. strongest baseline (%)")
    ax.set_title("DMSAFormer improvement over the best non-DMSAFormer baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "report_dmsaformer_improvement.png")
    plt.close(fig)
    return stats


def plot_error_distribution(errors: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160, sharey=False)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        data = []
        labels = []
        subset = errors[errors["output_len"] == horizon]
        for model in MODELS:
            values = subset[subset["model"] == model]["abs_error"].to_numpy(dtype=float)
            stats[f"{horizon}_{model}_median_abs_error"] = float(np.median(values))
            if values.size > 30000:
                rng = np.random.default_rng(2026)
                values = rng.choice(values, size=30000, replace=False)
            data.append(values)
            labels.append(DISPLAY[model])
        box = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, model in zip(box["boxes"], MODELS, strict=True):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.7)
        ax.set_title(f"{horizon}-day absolute error distribution")
        ax.set_ylabel("|Prediction - Ground Truth|")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "report_error_distribution.png")
    plt.close(fig)
    return stats


def plot_step_mae(errors: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160, sharey=False)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        subset = errors[(errors["output_len"] == horizon) & (errors["model"] == "dmsaformer")]
        step_mae = subset.groupby("step", as_index=False)["abs_error"].mean()
        window = 7 if horizon == 90 else 30
        step_mae["rolling"] = step_mae["abs_error"].rolling(window=window, min_periods=1, center=True).mean()
        ax.plot(step_mae["step"], step_mae["abs_error"], color="#9ecae1", linewidth=1.2, label="step MAE")
        ax.plot(step_mae["step"], step_mae["rolling"], color="#08519c", linewidth=2.2, label=f"{window}-step rolling mean")
        ax.set_title(f"DMSAFormer error by forecast step ({horizon}-day)")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        stats[f"{horizon}_dmsa_step_mae_start"] = float(step_mae["abs_error"].head(max(1, horizon // 10)).mean())
        stats[f"{horizon}_dmsa_step_mae_end"] = float(step_mae["abs_error"].tail(max(1, horizon // 10)).mean())
    fig.tight_layout()
    fig.savefig(FIGURES / "report_dmsaformer_step_mae.png")
    plt.close(fig)
    return stats


def plot_calibration_choices() -> None:
    choices = pd.read_csv(METRICS / "dmsaformer_calibration_choices.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), dpi=160)
    for ax, horizon in zip(axes, [90, 365], strict=True):
        subset = choices[choices["horizon"] == horizon].sort_values("seed")
        ax.plot(subset["seed"].astype(str), subset["calibration_scale"], marker="o", label="scale a", color="#08519c")
        ax.set_ylabel("scale a")
        ax.grid(alpha=0.25)
        twin = ax.twinx()
        twin.plot(subset["seed"].astype(str), subset["calibration_bias"], marker="s", label="bias b", color="#b23a48")
        twin.set_ylabel("bias b")
        ax.set_title(f"{horizon}-day DMSAFormer self affine calibration fitted on validation predictions")
        ax.set_xlabel("Seed")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "report_dmsaformer_calibration.png")
    plt.close(fig)


def plot_dmsaformer_evolution(current_summary: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    stage_frames = {stage: pd.read_csv(path) for stage, path in EVOLUTION_SUMMARIES.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    stage_labels = list(EVOLUTION_SUMMARIES)
    bar_colors = ["#9ecae1", "#6baed6", "#08519c"]
    for ax, horizon in zip(axes, [90, 365], strict=True):
        values = []
        for stage in stage_labels:
            frame = stage_frames[stage]
            row = frame[(frame["Model"] == "dmsaformer") & (frame["Horizon"] == horizon)].iloc[0]
            values.append(float(row["MSE mean"]))
        best_baseline = float(
            current_summary[
                (current_summary["Horizon"] == horizon) & (current_summary["Model"] != "dmsaformer")
            ]["MSE mean"].min()
        )
        values_with_baseline = values + [best_baseline]
        labels = stage_labels + ["最强 baseline"]
        colors = bar_colors + ["#b23a48"]
        bars = ax.bar(labels, values_with_baseline, color=colors, alpha=0.9)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
        ax.set_title(f"{horizon} 天任务：DMSAFormer 改进历程")
        ax.set_ylabel("MSE mean")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        stats[f"{horizon}_dmsa_v1_to_final_mse_reduction_pct"] = float((values[0] - values[2]) / values[0] * 100.0)
        stats[f"{horizon}_dmsa_v2_to_final_mse_reduction_pct"] = float((values[1] - values[2]) / values[1] * 100.0)
        stats[f"{horizon}_dmsa_final_vs_best_baseline_mse_reduction_pct"] = float(
            (best_baseline - values[2]) / best_baseline * 100.0
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "report_dmsaformer_evolution.png")
    plt.close(fig)
    return stats


def plot_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.axis("off")
    boxes = [
        ("原始分钟级电力数据\n+ 月度天气变量", 0.05, 0.62),
        ("缺失处理与日级聚合\n求和/均值/派生余量", 0.26, 0.62),
        ("90天输入窗口\n90/365天输出窗口", 0.47, 0.62),
        ("正式三模型\nLSTM / Transformer / DMSAFormer", 0.68, 0.62),
        ("DMSA自身验证集校准\n只用valid不碰test", 0.47, 0.22),
        ("测试集评估\nMSE/MAE mean±std", 0.68, 0.22),
    ]
    for text, x, y in boxes:
        rect = plt.Rectangle((x, y), 0.17, 0.18, facecolor="#D9EAF7", edgecolor="#1F4E79", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.085, y + 0.09, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.22, 0.71), (0.26, 0.71)),
        ((0.43, 0.71), (0.47, 0.71)),
        ((0.64, 0.71), (0.68, 0.71)),
        ((0.555, 0.62), (0.555, 0.40)),
        ((0.64, 0.31), (0.68, 0.31)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.5, 0.94, "实验流程与数据泄漏控制", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "report_pipeline_flow.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = read_metric_rows()
    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    errors = read_prediction_errors()

    stats: dict[str, float] = {}
    plot_per_seed_mse(metrics)
    stats.update(plot_improvement(summary))
    stats.update(plot_error_distribution(errors))
    stats.update(plot_step_mae(errors))
    stats.update(plot_dmsaformer_evolution(summary))
    plot_calibration_choices()
    plot_pipeline_flow()

    out = METRICS / "report_analysis_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    for path in [
        "report_pipeline_flow.png",
        "report_dmsaformer_improvement.png",
        "report_per_seed_mse.png",
        "report_error_distribution.png",
        "report_dmsaformer_step_mae.png",
        "report_dmsaformer_calibration.png",
        "report_dmsaformer_evolution.png",
    ]:
        print(FIGURES / path)


if __name__ == "__main__":
    main()
