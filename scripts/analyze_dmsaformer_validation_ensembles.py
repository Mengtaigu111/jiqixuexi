from __future__ import annotations

from itertools import combinations
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


MODELS = ["lstm", "transformer", "hybrid"]
SEEDS = [2026, 2027, 2028, 2029, 2030]


@torch.no_grad()
def predict_checkpoint(model_name: str, horizon: int, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
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
    y_true, y_pred = [], []
    for X, y, _ in loader:
        y_pred.append(model(X).numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def fit_ridge(predictions: list[np.ndarray], target: np.ndarray, ridge: float = 1e-2) -> tuple[np.ndarray, float]:
    matrix = np.stack([pred.reshape(-1) for pred in predictions], axis=1)
    target_vector = target.reshape(-1)
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1), dtype=matrix.dtype)], axis=1)
    penalty = np.eye(design.shape[1], dtype=matrix.dtype) * ridge
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ target_vector)
    return beta[:-1], float(beta[-1])


def apply_linear(predictions: list[np.ndarray], weights: np.ndarray, bias: float) -> np.ndarray:
    output = np.zeros_like(predictions[0])
    for weight, pred in zip(weights, predictions, strict=True):
        output = output + float(weight) * pred
    return output + bias


def subset_names() -> list[tuple[str, ...]]:
    names = []
    for size in range(1, len(MODELS) + 1):
        names.extend(combinations(MODELS, size))
    return names


def main() -> None:
    scaler_bundle = joblib.load("data/processed/scaler.pkl")
    rows = []
    for horizon in [90, 365]:
        for seed in SEEDS:
            valid_true = test_true = None
            valid_by_model = {}
            test_by_model = {}
            for model_name in MODELS:
                current_valid_true, valid_pred = predict_checkpoint(model_name, horizon, seed, "valid")
                current_test_true, test_pred = predict_checkpoint(model_name, horizon, seed, "test")
                if valid_true is None:
                    valid_true = current_valid_true
                    test_true = current_test_true
                valid_by_model[model_name] = valid_pred
                test_by_model[model_name] = test_pred

            candidates = []
            for names in subset_names():
                valid_preds = [valid_by_model[name] for name in names]
                weights, bias = fit_ridge(valid_preds, valid_true)
                valid_pred = apply_linear(valid_preds, weights, bias)
                valid_metrics = compute_metrics(
                    inverse_transform_target(valid_true, scaler_bundle),
                    inverse_transform_target(valid_pred, scaler_bundle),
                )
                candidates.append((valid_metrics["mse"], names, weights, bias))

            valid_mse, names, weights, bias = min(candidates, key=lambda item: item[0])
            test_preds = [test_by_model[name] for name in names]
            test_pred = apply_linear(test_preds, weights, bias)
            test_metrics = compute_metrics(
                inverse_transform_target(test_true, scaler_bundle),
                inverse_transform_target(test_pred, scaler_bundle),
            )
            rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "method": "+".join(names),
                    "valid_mse": valid_mse,
                    "test_mse": test_metrics["mse"],
                    "test_mae": test_metrics["mae"],
                    "weights": ",".join(f"{weight:.6f}" for weight in weights),
                    "bias": bias,
                }
            )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
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
