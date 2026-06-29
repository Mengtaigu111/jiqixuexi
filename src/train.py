from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.dataset import make_dataloader
from src.models.dmsaformer import DMSAFormer
from src.models.hybrid_model import HybridTCNTransformer
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
from src.utils import compute_metrics, ensure_dir, resolve_device, set_seed


def build_model(
    model_name: str,
    num_features: int,
    output_len: int,
    hidden_size: int = 64,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    name = model_name.lower()
    if name == "lstm":
        return LSTMForecaster(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_len=output_len,
        )
    if name == "transformer":
        return TransformerForecaster(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_len=output_len,
        )
    if name in {"hybrid", "hybridtcntransformer"}:
        return HybridTCNTransformer(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_len=output_len,
        )
    if name == "dmsaformer":
        return DMSAFormer(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_len=output_len,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _run_train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def _evaluate_loader(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    losses, all_true, all_pred = [], [], []
    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        losses.append(float(criterion(pred, y).item()))
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
    if not all_true:
        return {"loss": float("inf"), "mse": float("inf"), "mae": float("inf")}
    metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def train_model(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_path = data_dir / f"train_{args.output_len}.npz"
    valid_path = data_dir / f"valid_{args.output_len}.npz"
    test_path = data_dir / f"test_{args.output_len}.npz"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not valid_path.exists():
        valid_path = train_path

    train_loader = make_dataloader(train_path, batch_size=args.batch_size, shuffle=True)
    valid_loader = make_dataloader(valid_path, batch_size=args.batch_size, shuffle=False)
    if len(train_loader.dataset) == 0:
        raise ValueError(f"No training samples in {train_path}")
    if len(valid_loader.dataset) == 0:
        valid_loader = make_dataloader(train_path, batch_size=args.batch_size, shuffle=False)

    sample_X, _, _ = train_loader.dataset[0]
    num_features = int(sample_X.shape[-1])
    feature_names = [str(name) for name in getattr(train_loader.dataset, "feature_names", [])]
    if feature_names:
        print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(
        f"Dataset shapes: train X={train_loader.dataset.X.shape} y={train_loader.dataset.y.shape}; "
        f"valid X={valid_loader.dataset.X.shape} y={valid_loader.dataset.y.shape}"
    )
    if test_path.exists():
        test_data = np.load(test_path, allow_pickle=True)
        print(f"Dataset shapes: test X={test_data['X'].shape} y={test_data['y'].shape}")
    device = resolve_device(args.device)
    model = build_model(
        model_name=args.model,
        num_features=num_features,
        output_len=args.output_len,
        hidden_size=args.hidden_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer_cls = torch.optim.AdamW if args.optimizer.lower() == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    metrics_dir = ensure_dir(args.metrics_dir)
    save_dir = ensure_dir(args.save_dir)
    log_path = metrics_dir / f"{args.model}_{args.output_len}_seed{args.seed}.csv"
    checkpoint_path = save_dir / f"{args.model}_{args.output_len}_seed{args.seed}.pt"

    best_mse = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_train_epoch(model, train_loader, criterion, optimizer, device)
        valid_metrics = _evaluate_loader(model, valid_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_mse": valid_metrics["mse"],
            "valid_mae": valid_metrics["mae"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, index=False)
        if valid_metrics["mse"] < best_mse:
            best_mse = valid_metrics["mse"]
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "output_len": args.output_len,
                    "num_features": num_features,
                    "seed": args.seed,
                    "model_kwargs": {
                        "hidden_size": args.hidden_size,
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                break
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train household power forecasting models.")
    parser.add_argument("--model", choices=["lstm", "transformer", "hybrid", "dmsaformer"], required=True)
    parser.add_argument("--output_len", type=int, choices=[90, 365], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    checkpoint = train_model(parse_args())
    print(f"Saved best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
