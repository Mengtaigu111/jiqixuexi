from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.dataset import make_dataloader
from src.evaluate import plot_error_curve, plot_prediction_curve
from src.train import build_model
from src.utils import compute_metrics, ensure_dir, inverse_transform_target


SEEDS = (2026, 2027, 2028, 2029, 2030)
SHORT_HORIZON_EXPERTS = ("hybrid", "transformer")
LONG_HORIZON_EXPERT = "lstm"


def fit_affine_calibration(prediction: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> tuple[float, float]:
    matrix = np.stack([prediction.reshape(-1), np.ones(prediction.size, dtype=prediction.dtype)], axis=1)
    target_vector = target.reshape(-1)
    penalty = np.diag([ridge, 0.0]).astype(matrix.dtype)
    scale, bias = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target_vector)
    return float(scale), float(bias)


def select_short_horizon_experts(valid_metrics: pd.DataFrame, threshold_multiplier: float = 2.0) -> dict[int, str]:
    required = {"seed", "model", "valid_mse"}
    if not required.issubset(valid_metrics.columns):
        raise ValueError(f"valid_metrics must include {sorted(required)}")
    hybrid_metrics = valid_metrics[valid_metrics["model"] == "hybrid"]["valid_mse"].to_numpy(dtype=float)
    if hybrid_metrics.size == 0:
        raise ValueError("hybrid validation metrics are required")
    threshold = float(np.std(hybrid_metrics, ddof=1) * threshold_multiplier) if hybrid_metrics.size > 1 else 0.0
    choices: dict[int, str] = {}
    for seed, group in valid_metrics.groupby("seed"):
        by_model = {str(row.model): float(row.valid_mse) for row in group.itertuples(index=False)}
        missing = set(SHORT_HORIZON_EXPERTS) - set(by_model)
        if missing:
            raise ValueError(f"Missing validation metrics for seed {seed}: {sorted(missing)}")
        improvement = by_model["hybrid"] - by_model["transformer"]
        choices[int(seed)] = "transformer" if improvement > threshold else "hybrid"
    return choices


@torch.no_grad()
def predict_checkpoint(
    model_name: str,
    horizon: int,
    seed: int,
    split: str,
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}_{horizon}_seed{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kwargs = checkpoint.get("model_kwargs", {})
    model = build_model(
        checkpoint.get("model_name", model_name),
        checkpoint["num_features"],
        checkpoint["output_len"],
        hidden_size=kwargs.get("hidden_size", 64),
        d_model=kwargs.get("d_model", 64),
        nhead=kwargs.get("nhead", 4),
        num_layers=kwargs.get("num_layers", 2),
        dim_feedforward=kwargs.get("dim_feedforward", 128),
        dropout=kwargs.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = make_dataloader(Path(data_dir) / f"{split}_{horizon}.npz", batch_size=batch_size, shuffle=False)
    y_true, y_pred, dates = [], [], []
    for X, y, batch_dates in loader:
        y_true.append(y.numpy())
        y_pred.append(model(X).numpy())
        dates.append(batch_dates)
    if not y_true:
        raise ValueError(f"No samples available in {split}_{horizon}.npz")
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(dates)


def _original_scale_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_bundle) -> dict[str, float]:
    return compute_metrics(
        inverse_transform_target(y_true_scaled, scaler_bundle),
        inverse_transform_target(y_pred_scaled, scaler_bundle),
    )


def _write_prediction_outputs(
    horizon: int,
    seed: int,
    source_model: str,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_dates: np.ndarray,
    scaler_bundle,
    predictions_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    calibration_scale: float | None,
    calibration_bias: float | None,
) -> dict[str, object]:
    y_true = inverse_transform_target(y_true_scaled, scaler_bundle)
    y_pred = inverse_transform_target(y_pred_scaled, scaler_bundle)
    metrics = compute_metrics(y_true, y_pred)
    pred_path = predictions_dir / f"dmsaformer_{horizon}_seed{seed}.csv"
    metric_path = metrics_dir / f"dmsaformer_{horizon}_seed{seed}_test_metrics.csv"

    rows = []
    for sample_idx in range(y_true.shape[0]):
        for step in range(y_true.shape[1]):
            rows.append(
                {
                    "model": "dmsaformer",
                    "source_model": source_model,
                    "output_len": horizon,
                    "seed": seed,
                    "sample_index": sample_idx,
                    "step": step + 1,
                    "date": str(target_dates[sample_idx, step])[:10],
                    "y_true": float(y_true[sample_idx, step]),
                    "y_pred": float(y_pred[sample_idx, step]),
                    "calibration_scale": calibration_scale,
                    "calibration_bias": calibration_bias,
                }
            )
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "dmsaformer",
                "output_len": horizon,
                "seed": seed,
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "predictions": str(pred_path),
            }
        ]
    ).to_csv(metric_path, index=False)

    first_dates = pd.to_datetime(target_dates[0])
    first_metrics = compute_metrics(y_true[0], y_pred[0])
    plot_prediction_curve(
        first_dates,
        y_true[0],
        y_pred[0],
        "dmsaformer",
        horizon,
        seed,
        first_metrics["mse"],
        first_metrics["mae"],
        figures_dir / f"dmsaformer_{horizon}_seed{seed}_curve.png",
    )
    plot_error_curve(
        first_dates,
        y_pred[0] - y_true[0],
        "dmsaformer",
        horizon,
        seed,
        figures_dir / f"dmsaformer_{horizon}_seed{seed}_error.png",
    )
    return {
        "horizon": horizon,
        "seed": seed,
        "source_model": source_model,
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
    }


def export_calibrated_dmsaformer(
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    predictions_dir: str | Path = "results/predictions",
    metrics_dir: str | Path = "results/metrics",
    figures_dir: str | Path = "results/figures",
    threshold_multiplier: float = 2.0,
) -> pd.DataFrame:
    scaler_bundle = joblib.load(scaler_path)
    predictions_path = ensure_dir(predictions_dir)
    metrics_path = ensure_dir(metrics_dir)
    figures_path = ensure_dir(figures_dir)

    valid_rows = []
    for seed in SEEDS:
        for model_name in SHORT_HORIZON_EXPERTS:
            y_valid, pred_valid, _ = predict_checkpoint(model_name, 90, seed, "valid", checkpoint_dir, data_dir)
            valid_metrics = _original_scale_metrics(y_valid, pred_valid, scaler_bundle)
            valid_rows.append({"seed": seed, "model": model_name, "valid_mse": valid_metrics["mse"]})
    valid_frame = pd.DataFrame(valid_rows)
    short_choices = select_short_horizon_experts(valid_frame, threshold_multiplier=threshold_multiplier)

    choice_rows = []
    for seed in SEEDS:
        source_model = short_choices[seed]
        y_test, pred_test, dates = predict_checkpoint(source_model, 90, seed, "test", checkpoint_dir, data_dir)
        choice_rows.append(
            _write_prediction_outputs(
                90,
                seed,
                source_model,
                y_test,
                pred_test,
                dates,
                scaler_bundle,
                predictions_path,
                metrics_path,
                figures_path,
                calibration_scale=None,
                calibration_bias=None,
            )
        )

    for seed in SEEDS:
        y_valid, pred_valid, _ = predict_checkpoint(LONG_HORIZON_EXPERT, 365, seed, "valid", checkpoint_dir, data_dir)
        scale, bias = fit_affine_calibration(pred_valid, y_valid)
        y_test, pred_test, dates = predict_checkpoint(LONG_HORIZON_EXPERT, 365, seed, "test", checkpoint_dir, data_dir)
        calibrated = scale * pred_test + bias
        choice_rows.append(
            _write_prediction_outputs(
                365,
                seed,
                LONG_HORIZON_EXPERT,
                y_test,
                calibrated,
                dates,
                scaler_bundle,
                predictions_path,
                metrics_path,
                figures_path,
                calibration_scale=scale,
                calibration_bias=bias,
            )
        )

    choices = pd.DataFrame(choice_rows)
    choices.to_csv(metrics_path / "dmsaformer_calibration_choices.csv", index=False)
    valid_frame.to_csv(metrics_path / "dmsaformer_short_horizon_validation_metrics.csv", index=False)
    return choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validation-calibrated DMSAFormer expert predictions.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--threshold_multiplier", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    choices = export_calibrated_dmsaformer(**vars(parse_args()))
    print(choices.to_string(index=False))


if __name__ == "__main__":
    main()
