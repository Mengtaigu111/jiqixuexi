from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PowerWindowDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        data = np.load(self.path, allow_pickle=True)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.target_dates = data["target_dates"]
        self.feature_names = data.get("feature_names", np.array([]))
        if len(self.X) != len(self.y) or len(self.X) != len(self.target_dates):
            raise ValueError(f"Inconsistent sample counts in {self.path}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        X = torch.from_numpy(self.X[index])
        y = torch.from_numpy(self.y[index])
        return X, y, self.target_dates[index]


def make_dataloader(
    npz_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = PowerWindowDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_power_batch,
    )


def collate_power_batch(batch):
    xs, ys, dates = zip(*batch)
    return torch.stack(xs), torch.stack(ys), np.stack(dates)
