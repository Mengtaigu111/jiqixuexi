from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def _read_metric_rows(metrics_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(metrics_dir).glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"model", "output_len", "seed", "test_mse", "test_mae"}.issubset(frame.columns):
            rows.append(frame[["model", "output_len", "seed", "test_mse", "test_mae"]])
    if not rows:
        return pd.DataFrame(columns=["model", "output_len", "seed", "test_mse", "test_mae"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["model", "output_len", "seed"])


def _normalize_model_names(model_names: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if not model_names:
        return None
    normalized = {name.strip() for name in model_names if name.strip()}
    return normalized or None


def summarize_metrics(metrics_dir: str | Path, model_names: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    metrics = _read_metric_rows(metrics_dir)
    allowed_models = _normalize_model_names(model_names)
    if allowed_models is not None:
        metrics = metrics[metrics["model"].isin(allowed_models)]
    if metrics.empty:
        return pd.DataFrame(columns=["Model", "Horizon", "MSE mean", "MSE std", "MAE mean", "MAE std", "Runs"])
    grouped = (
        metrics.groupby(["model", "output_len"], as_index=False)
        .agg(
            **{
                "MSE mean": ("test_mse", "mean"),
                "MSE std": ("test_mse", "std"),
                "MAE mean": ("test_mae", "mean"),
                "MAE std": ("test_mae", "std"),
                "Runs": ("seed", "nunique"),
            }
        )
        .rename(columns={"model": "Model", "output_len": "Horizon"})
    )
    for col in ["MSE std", "MAE std"]:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped.sort_values(["Horizon", "Model"]).reset_index(drop=True)


def _format_markdown_value(value) -> str:
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_outputs(summary: pd.DataFrame, metrics_dir: str | Path) -> tuple[Path, Path]:
    metrics_path = ensure_dir(metrics_dir)
    csv_path = metrics_path / "summary.csv"
    md_path = metrics_path / "summary.md"
    summary.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(summary) if not summary.empty else "No test metrics found.\n")
        f.write("\n")
    return csv_path, md_path


def _plot_metric_bar(summary: pd.DataFrame, metric_prefix: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    if summary.empty:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
    else:
        labels = [f"{row.Model}-{int(row.Horizon)}" for row in summary.itertuples()]
        means = summary[f"{metric_prefix} mean"].to_numpy(dtype=float)
        stds = summary[f"{metric_prefix} std"].to_numpy(dtype=float)
        ax.bar(labels, means, yerr=stds, capsize=4, color="#4c78a8")
        ax.set_ylabel(metric_prefix)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_prediction_comparison(
    predictions_dir: Path,
    output_len: int,
    out_path: Path,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> None:
    ensure_dir(out_path.parent)
    allowed_models = _normalize_model_names(model_names)
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    plotted_truth = False
    for path in sorted(predictions_dir.glob(f"*_{output_len}_seed*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        first_sample = frame[frame["sample_index"] == frame["sample_index"].min()]
        dates = pd.to_datetime(first_sample["date"])
        model = str(first_sample["model"].iloc[0])
        if allowed_models is not None and model not in allowed_models:
            continue
        if not plotted_truth:
            ax.plot(dates, first_sample["y_true"], label="Ground Truth", linewidth=2.2, color="#333333")
            plotted_truth = True
        ax.plot(dates, first_sample["y_pred"], label=model, linewidth=1.8)
    if not plotted_truth:
        ax.text(0.5, 0.5, f"No {output_len}-day predictions", ha="center", va="center")
    ax.set_title(f"Prediction comparison horizon={output_len}")
    ax.set_xlabel("Date")
    ax.set_ylabel("power")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def copy_screenshots(figures_dir: str | Path, screenshots_dir: str | Path) -> list[Path]:
    src = Path(figures_dir)
    dst = ensure_dir(screenshots_dir)
    copied = []
    for path in src.glob("*.png"):
        target = dst / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def summarize_project(args: argparse.Namespace) -> pd.DataFrame:
    model_names = args.models if getattr(args, "models", None) else None
    summary = summarize_metrics(args.metrics_dir, model_names=model_names)
    write_summary_outputs(summary, args.metrics_dir)
    figures_dir = ensure_dir(args.figures_dir)
    _plot_metric_bar(summary, "MSE", figures_dir / "metric_bar_mse.png")
    _plot_metric_bar(summary, "MAE", figures_dir / "metric_bar_mae.png")
    predictions_dir = Path(args.predictions_dir)
    _plot_prediction_comparison(predictions_dir, 90, figures_dir / "prediction_comparison_90.png", model_names=model_names)
    _plot_prediction_comparison(predictions_dir, 365, figures_dir / "prediction_comparison_365.png", model_names=model_names)
    copy_screenshots(figures_dir, args.screenshots_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed forecasting experiment results.")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--screenshots_dir", default="results/screenshots")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model names to include in summary and comparison plots.")
    return parser.parse_args()


def main() -> None:
    summary = summarize_project(parse_args())
    print(summary.to_string(index=False) if not summary.empty else "No test metrics found.")


if __name__ == "__main__":
    main()
