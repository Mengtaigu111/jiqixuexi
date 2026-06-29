from __future__ import annotations

import torch
from torch import nn

from src.models.transformer import PositionalEncoding


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class HybridTCNTransformer(nn.Module):
    """TCN/CNN front-end plus Transformer encoder for multi-step forecasting."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_len: int = 90,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(num_features, d_model, kernel_size=1)
        self.tcn = nn.Sequential(
            TemporalResidualBlock(d_model, dilation=1, dropout=dropout),
            TemporalResidualBlock(d_model, dilation=2, dropout=dropout),
            TemporalResidualBlock(d_model, dilation=4, dropout=dropout),
        )
        self.positional_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.input_proj(z)
        z = self.tcn(z).transpose(1, 2)
        z = self.positional_encoding(z)
        encoded = self.encoder(z)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)
