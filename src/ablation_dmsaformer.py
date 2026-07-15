"""Module-level ablation of DMSAFormer.

The main table shows DMSAFormer does not beat the LSTM/Transformer baselines.
To turn that into a *diagnosed* result, we ablate the model one module at a
time and measure the raw (uncalibrated) test MSE/MAE across seeds. This tells us
which components actually help, which are inert, and which actively hurt on the
small-sample / long-horizon regime of this dataset.

Variants (each disables exactly one switch relative to ``full``):

* ``full``                 - all modules on (the reported DMSAFormer).
* ``no_correction``        - drop the multi-scale attentive correction branch,
                             leaving the decomposition backbone + LSTM/TCN experts.
* ``no_decomposition``     - replace trend/residual decomposition with a single
                             linear map of the centered target.
* ``no_window_norm``       - remove instance-level window-mean centering.
* ``no_variable_attention``- remove the feature-wise gate.

Every variant is trained from scratch for each seed and horizon (no weight
reuse), evaluated on the test split in original units, and summarized as
mean/std over seeds. Results are written to ``results/metrics``:

* ``ablation_dmsaformer_per_seed.csv``
* ``ablation_dmsaformer_summary.csv`` / ``.md``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset import make_dataloader
from src.models.dmsaformer import DMSAFormer
from src.utils import compute_metrics, ensure_dir, inverse_transform_target, resolve_device, set_seed

HORIZONS = (90, 365)
SEEDS = (2026, 2027, 2028, 2029, 2030)

# switch overrides applied on top of the all-on defaults
VARIANTS: dict[str, dict[str, bool]] = {
    "full": {},
    "no_correction": {"use_correction": False},
    "no_decomposition": {"use_decomposition": False},
    "no_window_norm": {"use_window_norm": False},
    "no_variable_attention": {"use_variable_attention": False},
}


def _count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _train_one(
    variant_switches: dict[str, bool],
    horizon: int,
    seed: int,
    data_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> tuple[nn.Module, int]:
    set_seed(seed)
    train_loader = make_dataloader(data_dir / f"train_{horizon}.npz", batch_size=batch_size, shuffle=True)
    valid_loader = make_dataloader(data_dir / f"valid_{horizon}.npz", batch_size=batch_size, shuffle=False)
    sample_X, _, _ = train_loader.dataset[0]
    num_features = int(sample_X.shape[-1])

    model = DMSAFormer(num_features=num_features, output_len=horizon, **variant_switches).to(device)
    n_params = _count_trainable_params(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state = None
    best_valid = float("inf")
    stale = 0
    for _ in range(1, epochs + 1):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y, _ in valid_loader:
                preds.append(model(X.to(device)).cpu().numpy())
                trues.append(y.numpy())
        valid_mse = compute_metrics(np.concatenate(trues), np.concatenate(preds))["mse"]
        if valid_mse < best_valid:
            best_valid = valid_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, n_params


@torch.no_grad()
def _test_metrics(model: nn.Module, horizon: int, data_dir: Path, device: torch.device, scaler_bundle) -> dict[str, float]:
    loader = make_dataloader(data_dir / f"test_{horizon}.npz", batch_size=256, shuffle=False)
    preds, trues = [], []
    model.eval()
    for X, y, _ in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        trues.append(y.numpy())
    y_pred = inverse_transform_target(np.concatenate(preds), scaler_bundle)
    y_true = inverse_transform_target(np.concatenate(trues), scaler_bundle)
    return compute_metrics(y_true, y_pred)


def run_ablation(
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    metrics_dir: str | Path = "results/metrics",
    horizons: tuple[int, ...] = HORIZONS,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "auto",
) -> pd.DataFrame:
    data_path = Path(data_dir)
    scaler_bundle = joblib.load(scaler_path)
    dev = resolve_device(device)
    rows: list[dict[str, object]] = []
    for variant, switches in VARIANTS.items():
        for horizon in horizons:
            for seed in seeds:
                model, n_params = _train_one(
                    switches, horizon, seed, data_path, dev, epochs, batch_size, lr, patience
                )
                metrics = _test_metrics(model, horizon, data_path, dev, scaler_bundle)
                rows.append(
                    {
                        "variant": variant,
                        "horizon": horizon,
                        "seed": seed,
                        "trainable_params": n_params,
                        "test_mse": metrics["mse"],
                        "test_mae": metrics["mae"],
                    }
                )
                print(f"{variant:<22} h={horizon:<3} seed={seed} params={n_params:>7} "
                      f"mse={metrics['mse']:.2f} mae={metrics['mae']:.2f}")

    per_seed = pd.DataFrame(rows)
    metrics_path = ensure_dir(metrics_dir)
    per_seed.to_csv(metrics_path / "ablation_dmsaformer_per_seed.csv", index=False)

    summary = (
        per_seed.groupby(["variant", "horizon"], as_index=False)
        .agg(
            **{
                "trainable_params": ("trainable_params", "first"),
                "MSE mean": ("test_mse", "mean"),
                "MSE std": ("test_mse", "std"),
                "MAE mean": ("test_mae", "mean"),
                "MAE std": ("test_mae", "std"),
                "Runs": ("seed", "nunique"),
            }
        )
    )
    for col in ["MSE std", "MAE std"]:
        summary[col] = summary[col].fillna(0.0)
    summary = summary.sort_values(["horizon", "MSE mean"]).reset_index(drop=True)
    summary.to_csv(metrics_path / "ablation_dmsaformer_summary.csv", index=False)

    with (metrics_path / "ablation_dmsaformer_summary.md").open("w", encoding="utf-8") as f:
        f.write("# DMSAFormer module ablation (raw test metrics, mean/std over seeds)\n\n")
        headers = list(summary.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary.itertuples(index=False, name=None):
            cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
            f.write("| " + " | ".join(cells) + " |\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module-level ablation of DMSAFormer (raw test metrics).")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ablation(
        data_dir=args.data_dir,
        scaler_path=args.scaler_path,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )
    print("\n=== ablation summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
