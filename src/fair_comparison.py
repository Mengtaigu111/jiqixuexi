"""Fair, apples-to-apples comparison of LSTM / Transformer / DMSAFormer.

The formal main table historically compared *validation-calibrated* DMSAFormer
against *uncalibrated* LSTM/Transformer, which gives DMSAFormer an asymmetric
advantage. This module removes that asymmetry by reporting, for every one of the
three formal models, two numbers per (horizon, seed):

* ``raw``        - test MSE/MAE with no calibration at all.
* ``calibrated`` - test MSE/MAE after fitting ``y = a * pred + b`` on the
  *validation* split only and applying it to the test predictions.

The affine parameters are always fit on validation and never on test, so the
calibrated numbers contain no leakage. Because the *same* calibration recipe is
applied to all three models, the comparison is fair: the reader can see both
whether uncalibrated DMSAFormer already beats the baselines and how much the
calibration step contributes to each model.

Outputs (under ``results/metrics`` by default):

* ``fair_comparison_per_seed.csv`` - one row per (model, horizon, seed) with raw
  and calibrated MSE/MAE plus the fitted (scale, bias).
* ``fair_comparison_summary.csv`` / ``.md`` - mean/std across seeds for both the
  raw and calibrated variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibrated_dmsaformer import (
    fit_affine_calibration,
    predict_checkpoint,
    _original_scale_metrics,
)
from src.utils import ensure_dir

FORMAL_MODELS = ("lstm", "transformer", "dmsaformer")
HORIZONS = (90, 365)
SEEDS = (2026, 2027, 2028, 2029, 2030)
CALIBRATION_METHOD = "validation_affine"


def collect_fair_metrics(
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    models: tuple[str, ...] = FORMAL_MODELS,
    horizons: tuple[int, ...] = HORIZONS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Return one row per (model, horizon, seed) with raw & calibrated metrics."""
    scaler_bundle = joblib.load(scaler_path)
    rows: list[dict[str, object]] = []
    for model in models:
        for horizon in horizons:
            for seed in seeds:
                ckpt = Path(checkpoint_dir) / f"{model}_{horizon}_seed{seed}.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)

                y_valid, pred_valid, _ = predict_checkpoint(model, horizon, seed, "valid", checkpoint_dir, data_dir)
                y_test, pred_test, _ = predict_checkpoint(model, horizon, seed, "test", checkpoint_dir, data_dir)

                # Calibration params fit on validation ONLY (no test leakage),
                # identical recipe for every model.
                scale, bias = fit_affine_calibration(pred_valid, y_valid)
                calibrated_test = scale * pred_test + bias

                raw_metrics = _original_scale_metrics(y_test, pred_test, scaler_bundle)
                cal_metrics = _original_scale_metrics(y_test, calibrated_test, scaler_bundle)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "seed": seed,
                        "calibration_method": CALIBRATION_METHOD,
                        "calibration_scale": scale,
                        "calibration_bias": bias,
                        "raw_test_mse": raw_metrics["mse"],
                        "raw_test_mae": raw_metrics["mae"],
                        "calibrated_test_mse": cal_metrics["mse"],
                        "calibrated_test_mae": cal_metrics["mae"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_fair_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-seed table into mean/std per (model, horizon)."""
    grouped = (
        per_seed.groupby(["model", "horizon"], as_index=False)
        .agg(
            **{
                "Raw MSE mean": ("raw_test_mse", "mean"),
                "Raw MSE std": ("raw_test_mse", "std"),
                "Raw MAE mean": ("raw_test_mae", "mean"),
                "Raw MAE std": ("raw_test_mae", "std"),
                "Cal MSE mean": ("calibrated_test_mse", "mean"),
                "Cal MSE std": ("calibrated_test_mse", "std"),
                "Cal MAE mean": ("calibrated_test_mae", "mean"),
                "Cal MAE std": ("calibrated_test_mae", "std"),
                "Runs": ("seed", "nunique"),
            }
        )
        .rename(columns={"model": "Model", "horizon": "Horizon"})
    )
    for col in ["Raw MSE std", "Raw MAE std", "Cal MSE std", "Cal MAE std"]:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped.sort_values(["Horizon", "Model"]).reset_index(drop=True)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(round(v, 4)) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_raw_test_metrics(per_seed: pd.DataFrame, metrics_dir: str | Path) -> list[Path]:
    """Overwrite each ``{model}_{horizon}_seed{seed}_test_metrics.csv`` with the
    raw (uncalibrated) numbers, in the exact schema ``summarize_results`` reads.

    This makes the formal main table an apples-to-apples RAW comparison for all
    three models, instead of calibrated-DMSAFormer versus raw baselines.
    """
    metrics_path = ensure_dir(metrics_dir)
    written: list[Path] = []
    for row in per_seed.itertuples(index=False):
        pred_path = f"results/predictions/{row.model}_{row.horizon}_seed{row.seed}.csv"
        metric_path = metrics_path / f"{row.model}_{row.horizon}_seed{row.seed}_test_metrics.csv"
        pd.DataFrame(
            [
                {
                    "model": row.model,
                    "output_len": row.horizon,
                    "seed": row.seed,
                    "test_mse": row.raw_test_mse,
                    "test_mae": row.raw_test_mae,
                    "predictions": pred_path,
                }
            ]
        ).to_csv(metric_path, index=False)
        written.append(metric_path)
    return written


def export_fair_comparison(
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    metrics_dir: str | Path = "results/metrics",
    write_raw_metrics: bool = True,
) -> pd.DataFrame:
    metrics_path = ensure_dir(metrics_dir)
    per_seed = collect_fair_metrics(checkpoint_dir, data_dir, scaler_path)
    per_seed.to_csv(metrics_path / "fair_comparison_per_seed.csv", index=False)
    if write_raw_metrics:
        write_raw_test_metrics(per_seed, metrics_path)

    summary = summarize_fair_metrics(per_seed)
    summary.to_csv(metrics_path / "fair_comparison_summary.csv", index=False)
    with (metrics_path / "fair_comparison_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Fair comparison (same validation-affine calibration for all models)\n\n")
        f.write("Raw = no calibration. Cal = y = a*pred + b fit on validation only.\n\n")
        f.write(_dataframe_to_markdown(summary))
        f.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fair raw-vs-calibrated comparison for the three formal models.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--metrics_dir", default="results/metrics")
    return parser.parse_args()


def main() -> None:
    summary = export_fair_comparison(**vars(parse_args()))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
