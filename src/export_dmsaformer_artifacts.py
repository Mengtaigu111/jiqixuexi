from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def _write_horizon_results(metrics_dir: Path, results_dir: Path, horizon: int) -> Path:
    rows = []
    for path in sorted(metrics_dir.glob(f"dmsaformer_{horizon}_seed*_test_metrics.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No DMSAFormer {horizon}-day metric files found in {metrics_dir}")
    combined = pd.concat(rows, ignore_index=True).sort_values(["output_len", "seed"])
    out_path = results_dir / f"dmsaformer_{horizon}_results.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def _write_prediction_plot(predictions_dir: Path, root_figures_dir: Path, horizon: int) -> Path:
    candidates = sorted(predictions_dir.glob(f"dmsaformer_{horizon}_seed*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No DMSAFormer {horizon}-day prediction CSV found in {predictions_dir}")
    frame = pd.read_csv(candidates[0])
    if frame.empty:
        raise ValueError(f"Prediction CSV is empty: {candidates[0]}")
    sample = frame[frame["sample_index"] == frame["sample_index"].min()].sort_values("step")
    out_path = root_figures_dir / f"dmsaformer_{horizon}_prediction.png"
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    ax.plot(sample["step"], sample["y_true"], label="Ground Truth", linewidth=2)
    ax.plot(sample["step"], sample["y_pred"], label="Predicted", linewidth=2)
    ax.set_title(f"DMSAFormer {horizon}-day prediction")
    ax.set_xlabel("Future day index")
    ax.set_ylabel("global_active_power")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def export_dmsaformer_artifacts(
    results_dir: str | Path = "results",
    metrics_dir: str | Path = "results/metrics",
    figures_dir: str | Path = "results/figures",
    predictions_dir: str | Path = "results/predictions",
    root_figures_dir: str | Path = "figures",
) -> list[Path]:
    results_path = ensure_dir(results_dir)
    metrics_path = Path(metrics_dir)
    Path(figures_dir)
    predictions_path = Path(predictions_dir)
    root_figures_path = ensure_dir(root_figures_dir)

    written = [
        _write_horizon_results(metrics_path, results_path, 90),
        _write_horizon_results(metrics_path, results_path, 365),
    ]

    summary_src = metrics_path / "summary.csv"
    if not summary_src.exists():
        raise FileNotFoundError(summary_src)
    summary_dst = results_path / "summary.csv"
    import shutil

    shutil.copy2(summary_src, summary_dst)
    written.append(summary_dst)

    written.append(_write_prediction_plot(predictions_path, root_figures_path, 90))
    written.append(_write_prediction_plot(predictions_path, root_figures_path, 365))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DMSAFormer artifacts using required goal filenames.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--root_figures_dir", default="figures")
    return parser.parse_args()


def main() -> None:
    for path in export_dmsaformer_artifacts(**vars(parse_args())):
        print(path)


if __name__ == "__main__":
    main()
