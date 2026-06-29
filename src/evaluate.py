from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import make_dataloader
from src.train import build_model
from src.utils import compute_metrics, ensure_dir, inverse_transform_target, resolve_device


def plot_prediction_curve(
    dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_len: int,
    seed: int,
    mse: float,
    mae: float,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    ax.plot(dates, y_true, label="Ground Truth", linewidth=2)
    ax.plot(dates, y_pred, label="Prediction", linewidth=2)
    ax.set_title(f"{model_name} horizon={output_len} seed={seed} | MSE={mse:.4f} MAE={mae:.4f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("power")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_curve(dates, errors: np.ndarray, model_name: str, output_len: int, seed: int, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ax.plot(dates, errors, color="#b23a48", linewidth=1.8)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title(f"{model_name} error horizon={output_len} seed={seed}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction - Ground Truth")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred, dates = [], [], []
    for X, y, batch_dates in loader:
        X = X.to(device)
        pred = model(X).detach().cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
        dates.append(batch_dates)
    if not y_true:
        raise ValueError("No samples available for evaluation.")
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(dates)


def evaluate_model(args: argparse.Namespace) -> Path:
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", args.model)
    output_len = int(checkpoint.get("output_len", args.output_len))
    seed = int(checkpoint.get("seed", args.seed))
    num_features = int(checkpoint["num_features"])
    kwargs = checkpoint.get("model_kwargs", {})

    device = resolve_device(args.device)
    model = build_model(
        model_name=model_name,
        num_features=num_features,
        output_len=output_len,
        hidden_size=kwargs.get("hidden_size", args.hidden_size),
        d_model=kwargs.get("d_model", args.d_model),
        nhead=kwargs.get("nhead", args.nhead),
        num_layers=kwargs.get("num_layers", args.num_layers),
        dim_feedforward=kwargs.get("dim_feedforward", args.dim_feedforward),
        dropout=kwargs.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_dir = Path(args.data_dir)
    test_path = data_dir / f"test_{output_len}.npz"
    loader = make_dataloader(test_path, batch_size=args.batch_size, shuffle=False)
    y_true_scaled, y_pred_scaled, target_dates = predict(model, loader, device)

    scaler_path = Path(args.scaler_path)
    scaler_bundle = joblib.load(scaler_path) if scaler_path.exists() else None
    y_true = inverse_transform_target(y_true_scaled, scaler_bundle)
    y_pred = inverse_transform_target(y_pred_scaled, scaler_bundle)
    metrics = compute_metrics(y_true, y_pred)

    pred_dir = ensure_dir(args.predictions_dir)
    metrics_dir = ensure_dir(args.metrics_dir)
    figures_dir = ensure_dir(args.figures_dir)
    pred_path = pred_dir / f"{model_name}_{output_len}_seed{seed}.csv"
    metric_path = metrics_dir / f"{model_name}_{output_len}_seed{seed}_test_metrics.csv"

    rows = []
    for sample_idx in range(y_true.shape[0]):
        for step in range(y_true.shape[1]):
            rows.append(
                {
                    "model": model_name,
                    "output_len": output_len,
                    "seed": seed,
                    "sample_index": sample_idx,
                    "step": step + 1,
                    "date": str(target_dates[sample_idx, step])[:10],
                    "y_true": float(y_true[sample_idx, step]),
                    "y_pred": float(y_pred[sample_idx, step]),
                }
            )
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "output_len": output_len,
                "seed": seed,
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "predictions": str(pred_path),
            }
        ]
    ).to_csv(metric_path, index=False)

    first_dates = pd.to_datetime(target_dates[0])
    curve_path = figures_dir / f"{model_name}_{output_len}_seed{seed}_curve.png"
    error_path = figures_dir / f"{model_name}_{output_len}_seed{seed}_error.png"
    first_metrics = compute_metrics(y_true[0], y_pred[0])
    plot_prediction_curve(
        first_dates,
        y_true[0],
        y_pred[0],
        model_name,
        output_len,
        seed,
        first_metrics["mse"],
        first_metrics["mae"],
        curve_path,
    )
    plot_error_curve(first_dates, y_pred[0] - y_true[0], model_name, output_len, seed, error_path)
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metric_path}")
    print(f"Test MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f}")
    return metric_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained forecasting checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--output_len", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    evaluate_model(parse_args())


if __name__ == "__main__":
    main()
