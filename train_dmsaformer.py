from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.train import train_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DMSAFormer for household power forecasting.")
    parser.add_argument("--pred_len", "--output_len", dest="output_len", type=int, choices=[90, 365], required=True)
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
    args = parser.parse_args(argv)
    args.model = "dmsaformer"
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    checkpoint = train_model(parse_args(argv))
    print(f"Saved best checkpoint: {checkpoint}")
    return checkpoint


if __name__ == "__main__":
    main()

