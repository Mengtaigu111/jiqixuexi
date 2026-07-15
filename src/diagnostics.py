"""Diagnostic analyses that explain *why* the proposed DMSAFormer, despite its
extra machinery, does not beat the simple baselines on this dataset.

Three independent, training-free diagnostics are produced (all metrics in the
original power scale, identical footing to the main table):

1. Naive floors (``naive_floor_summary.csv``). Persistence, look-back mean and
   weekly-seasonal-naive forecasts. These require *no* learning at all. If the
   deep models only barely clear (or fail to clear) these floors, that is strong
   evidence the task has little learnable structure beyond a stable level, which
   is exactly why a high-capacity model cannot pull ahead.

2. Capacity vs. payoff (``capacity_summary.csv``). Trainable-parameter count of
   every model against its test MSE. Shows DMSAFormer spends far more parameters
   for no accuracy gain -- the classic small-sample overparameterisation story.

3. Overfitting gap (``overfitting_gap_summary.csv``). Final train-loss vs.
   best-validation-MSE (both scaled) read from the training logs. A widening gap
   for the heavier models on the 365-day task quantifies overfitting on only
   ~78 training windows.

Run: ``python -m src.diagnostics``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.train import build_model
from src.utils import compute_metrics, ensure_dir, inverse_transform_target

FORMAL_MODELS = ("lstm", "transformer", "dmsaformer")
HORIZONS = (90, 365)
SEEDS = (2026, 2027, 2028, 2029, 2030)
TARGET_INDEX = 0
WEEK = 7


# --------------------------------------------------------------------------- #
# 1. Naive floors                                                             #
# --------------------------------------------------------------------------- #
def _persistence(target_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed day for the whole horizon."""
    last = target_hist[:, -1:]
    return np.repeat(last, horizon, axis=1)


def _window_mean(target_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the look-back window mean for the whole horizon."""
    mean = target_hist.mean(axis=1, keepdims=True)
    return np.repeat(mean, horizon, axis=1)


def _seasonal_naive(target_hist: np.ndarray, horizon: int, period: int = WEEK) -> np.ndarray:
    """Tile the last ``period`` observed days across the horizon (weekly cycle)."""
    season = target_hist[:, -period:]
    reps = int(np.ceil(horizon / period))
    tiled = np.tile(season, (1, reps))
    return tiled[:, :horizon]


def naive_floor_metrics(
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    scaler_bundle = joblib.load(scaler_path)
    feature_scaler = scaler_bundle.get("feature_scaler")
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        data = np.load(Path(data_dir) / f"test_{horizon}.npz", allow_pickle=True)
        X = data["X"].astype(np.float64)
        y_scaled = data["y"].astype(np.float64)

        # Recover the target history in original units (input channel is scaled
        # by feature_scaler, target y by target_scaler; both share the same
        # train-fitted mean/std for global_active_power, but we invert each with
        # its own scaler to stay exact).
        hist_scaled = X[:, :, TARGET_INDEX]
        if feature_scaler is not None:
            mean0 = float(feature_scaler.mean_[TARGET_INDEX])
            scale0 = float(feature_scaler.scale_[TARGET_INDEX])
            target_hist = hist_scaled * scale0 + mean0
        else:
            target_hist = hist_scaled
        y_true = inverse_transform_target(y_scaled, scaler_bundle)

        methods = {
            "persistence": _persistence(target_hist, horizon),
            "window_mean": _window_mean(target_hist, horizon),
            "seasonal_naive_7d": _seasonal_naive(target_hist, horizon, WEEK),
        }
        for name, pred in methods.items():
            metrics = compute_metrics(y_true, pred)
            rows.append(
                {
                    "method": name,
                    "horizon": horizon,
                    "test_mse": metrics["mse"],
                    "test_mae": metrics["mae"],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Capacity vs. payoff                                                       #
# --------------------------------------------------------------------------- #
def _count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def capacity_metrics(
    checkpoint_dir: str | Path = "checkpoints",
    metrics_dir: str | Path = "results/metrics",
    models: tuple[str, ...] = FORMAL_MODELS,
    horizons: tuple[int, ...] = HORIZONS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Param count (from a rebuilt model) alongside mean test MSE per model/horizon."""
    rows: list[dict[str, object]] = []
    for model_name in models:
        for horizon in horizons:
            # Parameter count is seed-independent; rebuild from the first checkpoint.
            ckpt_path = Path(checkpoint_dir) / f"{model_name}_{horizon}_seed{seeds[0]}.pt"
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            kwargs = checkpoint.get("model_kwargs", {})
            model = build_model(
                model_name,
                checkpoint["num_features"],
                checkpoint["output_len"],
                hidden_size=kwargs.get("hidden_size", 64),
                d_model=kwargs.get("d_model", 64),
                nhead=kwargs.get("nhead", 4),
                num_layers=kwargs.get("num_layers", 2),
                dim_feedforward=kwargs.get("dim_feedforward", 128),
                dropout=kwargs.get("dropout", 0.1),
            )
            n_params = _count_parameters(model)

            # Mean test MSE across seeds from the raw metric files.
            mses, maes = [], []
            for seed in seeds:
                metric_path = Path(metrics_dir) / f"{model_name}_{horizon}_seed{seed}_test_metrics.csv"
                if not metric_path.exists():
                    continue
                frame = pd.read_csv(metric_path)
                mses.append(float(frame["test_mse"].iloc[0]))
                maes.append(float(frame["test_mae"].iloc[0]))
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "trainable_params": n_params,
                    "test_mse_mean": float(np.mean(mses)) if mses else float("nan"),
                    "test_mae_mean": float(np.mean(maes)) if maes else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Overfitting gap                                                           #
# --------------------------------------------------------------------------- #
def overfitting_gap_metrics(
    metrics_dir: str | Path = "results/metrics",
    models: tuple[str, ...] = FORMAL_MODELS,
    horizons: tuple[int, ...] = HORIZONS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Final train loss vs. best valid MSE (both scaled) from the training logs.

    A large ``valid/train`` ratio means the model fits the training windows far
    better than it generalises -- the signature of small-sample overfitting.
    """
    rows: list[dict[str, object]] = []
    for model_name in models:
        for horizon in horizons:
            final_train, best_valid = [], []
            for seed in seeds:
                log_path = Path(metrics_dir) / f"{model_name}_{horizon}_seed{seed}.csv"
                if not log_path.exists():
                    continue
                log = pd.read_csv(log_path)
                if log.empty:
                    continue
                final_train.append(float(log["train_loss"].iloc[-1]))
                best_valid.append(float(log["valid_mse"].min()))
            if not final_train:
                continue
            ft = float(np.mean(final_train))
            bv = float(np.mean(best_valid))
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "final_train_loss": ft,
                    "best_valid_mse": bv,
                    "valid_over_train_ratio": bv / ft if ft > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def export_diagnostics(
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    metrics_dir: str | Path = "results/metrics",
) -> dict[str, pd.DataFrame]:
    metrics_path = ensure_dir(metrics_dir)
    naive = naive_floor_metrics(data_dir, scaler_path)
    capacity = capacity_metrics(checkpoint_dir, metrics_dir)
    gap = overfitting_gap_metrics(metrics_dir)

    naive.to_csv(metrics_path / "naive_floor_summary.csv", index=False)
    capacity.to_csv(metrics_path / "capacity_summary.csv", index=False)
    gap.to_csv(metrics_path / "overfitting_gap_summary.csv", index=False)
    return {"naive": naive, "capacity": capacity, "gap": gap}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training-free diagnostic analyses.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--metrics_dir", default="results/metrics")
    return parser.parse_args()


def main() -> None:
    tables = export_diagnostics(**vars(parse_args()))
    for name, frame in tables.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
