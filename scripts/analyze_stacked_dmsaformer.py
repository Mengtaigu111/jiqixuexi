from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import make_dataloader
from src.train import build_model
from src.utils import compute_metrics, inverse_transform_target


MODELS = ["lstm", "transformer", "hybrid", "dmsaformer"]
SEEDS = [2026, 2027, 2028, 2029, 2030]


@torch.no_grad()
def predict_checkpoint(model_name: str, horizon: int, seed: int, split: str):
    checkpoint_path = Path("checkpoints") / f"{model_name}_{horizon}_seed{seed}.pt"
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

    loader = make_dataloader(f"data/processed/{split}_{horizon}.npz", batch_size=256, shuffle=False)
    y_true, y_pred, dates = [], [], []
    for X, y, batch_dates in loader:
        y_pred.append(model(X).numpy())
        y_true.append(y.numpy())
        dates.append(batch_dates)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(dates)


def fit_ridge_stack(predictions: np.ndarray, target: np.ndarray, ridge: float = 1e-2):
    design = np.concatenate([predictions, np.ones((predictions.shape[0], 1), dtype=predictions.dtype)], axis=1)
    penalty = np.eye(design.shape[1], dtype=predictions.dtype) * ridge
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return beta[:-1], float(beta[-1])


def fit_stepwise_stack(valid_predictions: list[np.ndarray], target: np.ndarray, ridge: float = 1e-2):
    output_len = target.shape[1]
    weights = []
    biases = []
    for step in range(output_len):
        step_matrix = np.stack([pred[:, step] for pred in valid_predictions], axis=1)
        step_weights, step_bias = fit_ridge_stack(step_matrix, target[:, step], ridge=ridge)
        weights.append(step_weights)
        biases.append(step_bias)
    return np.stack(weights), np.asarray(biases)


def apply_stepwise_stack(test_predictions: list[np.ndarray], weights: np.ndarray, biases: np.ndarray):
    output = np.zeros_like(test_predictions[0])
    for step in range(output.shape[1]):
        step_matrix = np.stack([pred[:, step] for pred in test_predictions], axis=1)
        output[:, step] = step_matrix @ weights[step] + biases[step]
    return output


def main() -> None:
    scaler_bundle = joblib.load("data/processed/scaler.pkl")
    rows = []
    for horizon in [90, 365]:
        for seed in SEEDS:
            valid_preds, test_preds = [], []
            y_valid = y_test = target_dates = None
            for model_name in MODELS:
                current_valid, pred_valid, _ = predict_checkpoint(model_name, horizon, seed, "valid")
                current_test, pred_test, current_dates = predict_checkpoint(model_name, horizon, seed, "test")
                if y_valid is None:
                    y_valid = current_valid
                    y_test = current_test
                    target_dates = current_dates
                valid_preds.append(pred_valid.reshape(-1))
                test_preds.append(pred_test.reshape(-1))

            valid_arrays = [pred.reshape(y_valid.shape) for pred in valid_preds]
            test_arrays = [pred.reshape(y_test.shape) for pred in test_preds]
            weights, biases = fit_stepwise_stack(valid_arrays, y_valid)
            pred_scaled = apply_stepwise_stack(test_arrays, weights, biases)
            y_true = inverse_transform_target(y_test, scaler_bundle)
            y_pred = inverse_transform_target(pred_scaled, scaler_bundle)
            metrics = compute_metrics(y_true, y_pred)

            row = {
                "horizon": horizon,
                "seed": seed,
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "bias": float(np.mean(biases)),
            }
            row.update({f"weight_{name}": float(np.mean(weights[:, idx])) for idx, name in enumerate(MODELS)})
            rows.append(row)
            print(row)

    frame = pd.DataFrame(rows)
    print("\nSummary")
    print(
        frame.groupby("horizon")
        .agg(
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", "std"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
