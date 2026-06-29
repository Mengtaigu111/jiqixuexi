from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hybrid_model import HybridTCNTransformer
from src.models.lstm import LSTMForecaster
from src.models.transformer import PositionalEncoding


class MovingAverage(nn.Module):
    """Moving average over the time axis while preserving sequence length."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(x.shape)}")
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        z = x.transpose(1, 2)
        z = F.pad(z, (left, right), mode="replicate")
        z = F.avg_pool1d(z, kernel_size=self.kernel_size, stride=1)
        return z.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Split the series into slow trend and residual fluctuation components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        residual = x - trend
        return trend, residual


class VariableAttention(nn.Module):
    """Feature-wise gate learned from the whole lookback window."""

    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.scorer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x.mean(dim=1)
        weights = self.scorer(context).unsqueeze(1)
        return x * weights


class TargetDecompositionBackbone(nn.Module):
    """DLinear-style target-channel backbone for low-variance forecasting."""

    def __init__(self, input_len: int = 90, output_len: int = 90, kernel_size: int = 25):
        super().__init__()
        self.input_len = int(input_len)
        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)
        self.trend_linear = nn.Sequential(
            nn.LayerNorm(self.input_len),
            nn.Linear(self.input_len, output_len),
        )
        self.residual_linear = nn.Sequential(
            nn.LayerNorm(self.input_len),
            nn.Linear(self.input_len, output_len),
        )

    def _resize_to_input_len(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] == self.input_len:
            return target
        resized = F.interpolate(
            target.unsqueeze(1),
            size=self.input_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def forward(self, target_history: torch.Tensor) -> torch.Tensor:
        if target_history.ndim != 2:
            raise ValueError(f"Expected [batch, time], got {tuple(target_history.shape)}")
        target = self._resize_to_input_len(target_history).unsqueeze(-1)
        trend, residual = self.decomposition(target)
        trend = trend.squeeze(-1)
        residual = residual.squeeze(-1)
        return self.trend_linear(trend) + self.residual_linear(residual)


class DMSAFormer(nn.Module):
    """Decomposition-based Multi-Scale Attention Transformer.

    The model separates trend and residual signals. The trend branch gives a
    stable direct projection from the target trend, while the residual branch
    uses 3/7/30-day convolutions plus a Transformer encoder to model local,
    weekly, monthly, and longer-range dependencies.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_len: int = 90,
        input_len: int = 90,
        target_feature_index: int = 0,
        decomposition_kernel: int = 25,
        conv_kernel_sizes: tuple[int, ...] = (3, 7, 30),
    ):
        super().__init__()
        if not conv_kernel_sizes:
            raise ValueError("conv_kernel_sizes must not be empty")
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.target_feature_index = int(target_feature_index)
        self.conv_kernel_sizes = tuple(int(size) for size in conv_kernel_sizes)

        self.variable_attention = VariableAttention(num_features, hidden_dim=max(8, d_model // 2))
        self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
        self.target_backbone = TargetDecompositionBackbone(
            input_len=input_len,
            output_len=output_len,
            kernel_size=decomposition_kernel,
        )
        self.local_temporal_backbone = HybridTCNTransformer(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_len=output_len,
        )
        self.recurrent_backbone = LSTMForecaster(
            num_features=num_features,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            output_len=output_len,
        )

        # Multi-scale correction branch: 3-day, weekly, and monthly residual filters.
        self.residual_convs = nn.ModuleList(
            [nn.Conv1d(num_features, d_model, kernel_size=kernel_size) for kernel_size in self.conv_kernel_sizes]
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(d_model * len(self.conv_kernel_sizes), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
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
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len),
        )
        if self.output_len >= 365:
            branch_init = torch.tensor([0.0, 1.2], dtype=torch.float32)
        else:
            branch_init = torch.tensor([1.2, 0.0], dtype=torch.float32)
        self.branch_logits = nn.Parameter(branch_init)
        self.target_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.correction_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _same_length_conv(self, conv: nn.Conv1d, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        return conv(F.pad(x, (left, right), mode="replicate"))

    def _trend_input(self, trend: torch.Tensor) -> torch.Tensor:
        target_trend = trend[:, :, self.target_feature_index]
        if target_trend.shape[1] == self.input_len:
            return target_trend
        resized = F.interpolate(
            target_trend.unsqueeze(1),
            size=self.input_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, self.target_feature_index]
        y_backbone = self.target_backbone(target_history)

        attended = self.variable_attention(x)
        y_local = self.local_temporal_backbone(x)
        y_recurrent = self.recurrent_backbone(x)
        _, residual = self.decomposition(attended)

        residual_channels = residual.transpose(1, 2)
        multi_scale = [
            self._same_length_conv(conv, residual_channels, kernel_size)
            for conv, kernel_size in zip(self.residual_convs, self.conv_kernel_sizes, strict=True)
        ]
        z = torch.cat(multi_scale, dim=1).transpose(1, 2)
        z = self.residual_proj(z)
        z = self.positional_encoding(z)
        encoded = self.encoder(z)
        y_residual = self.residual_head(encoded.mean(dim=1))
        local_weight, recurrent_weight = torch.softmax(self.branch_logits, dim=0)
        y_main = local_weight * y_local + recurrent_weight * y_recurrent
        return y_main + self.target_scale * y_backbone + self.correction_scale * y_residual
