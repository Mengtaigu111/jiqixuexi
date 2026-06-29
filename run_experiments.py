from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-seed DMSAFormer experiments for 90 and 365 days.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "SEEDS": " ".join(str(seed) for seed in args.seeds),
            "EPOCHS": str(args.epochs),
            "BATCH_SIZE": str(args.batch_size),
            "DEVICE": args.device,
        }
    )
    subprocess.run(["bash", "scripts/run_dmsaformer_experiments.sh"], cwd=Path(__file__).parent, env=env, check=True)


if __name__ == "__main__":
    main()

