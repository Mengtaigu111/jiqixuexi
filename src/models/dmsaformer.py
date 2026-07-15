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
    """Window-normalized target-channel decomposition backbone."""

    def __init__(
        self,
        input_len: int = 90,
        output_len: int = 90,
        kernel_size: int = 25,
        use_window_norm: bool = True,
        use_decomposition: bool = True,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.use_window_norm = bool(use_window_norm)
        self.use_decomposition = bool(use_decomposition)
        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)
        self.trend_linear = nn.Linear(self.input_len, output_len)
        self.residual_linear = nn.Linear(self.input_len, output_len)

    def _resize_to_input_len(self, target_history: torch.Tensor) -> torch.Tensor:
        if target_history.shape[1] == self.input_len:
            return target_history
        resized = F.interpolate(
            target_history.unsqueeze(1),
            size=self.input_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(1)

    def forward(self, target_history: torch.Tensor) -> torch.Tensor:
        if target_history.ndim != 2:
            raise ValueError(f"Expected [batch, time], got {tuple(target_history.shape)}")
        series = self._resize_to_input_len(target_history)
        if self.use_window_norm:
            level = series.mean(dim=1, keepdim=True)
        else:
            level = series.new_zeros(series.shape[0], 1)
        centered = series - level
        if self.use_decomposition:
            trend, residual = self.decomposition(centered.unsqueeze(-1))
            projected = self.trend_linear(trend.squeeze(-1)) + self.residual_linear(residual.squeeze(-1))
        else:
            projected = self.trend_linear(centered)
        return projected + level


class DMSAFormer(nn.Module):
    """Decomposition-based Multi-Scale Attention Transformer.

    End-to-end forecaster built around two ideas that match the small-sample,
    long-horizon regime of daily household power data:

    1. Level-robust direct backbone. The target history is centered by its
       window mean (instance-level normalization), decomposed into trend and
       residual by a moving average, and each component is mapped to the
       future horizon by a dedicated linear head. The window mean is added
       back at the output, so the backbone extrapolates shape around a stable
       level instead of chasing absolute scale — this is what keeps the
       long-horizon output variance under control.

    2. Multi-scale attentive correction. A feature-gated multivariate branch
       applies 3/7/30-day convolutions (local, weekly, monthly patterns) and
       a Transformer encoder, producing an additive correction. Its gate is
       initialized at zero, so training starts from the robust backbone and
       the correction only grows where it reduces validation error.

    Ablation switches (`use_window_norm`, `use_decomposition`,
    `use_correction`, `use_variable_attention`) disable individual modules.
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
        use_window_norm: bool = True,
        use_decomposition: bool = True,
        use_correction: bool = True,
        use_variable_attention: bool = True,
    ):
        super().__init__()
        if not conv_kernel_sizes:
            raise ValueError("conv_kernel_sizes must not be empty")
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.target_feature_index = int(target_feature_index)
        self.conv_kernel_sizes = tuple(int(size) for size in conv_kernel_sizes)
        self.use_window_norm = bool(use_window_norm)
        self.use_decomposition = bool(use_decomposition)
        self.use_correction = bool(use_correction)
        self.use_variable_attention = bool(use_variable_attention)

        self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
        self.target_backbone = TargetDecompositionBackbone(
            input_len=input_len,
            output_len=output_len,
            kernel_size=decomposition_kernel,
            use_window_norm=use_window_norm,
            use_decomposition=use_decomposition,
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
        self.expert_logits = nn.Parameter(
            torch.tensor([1.2, 0.0], dtype=torch.float32)
            if self.output_len < 365
            else torch.tensor([0.0, 1.2], dtype=torch.float32)
        )
        self.target_backbone_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.expert_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        if self.use_correction:
            self.variable_attention = VariableAttention(num_features, hidden_dim=max(8, d_model // 2))
            self.correction_convs = nn.ModuleList(
                [nn.Conv1d(num_features, d_model, kernel_size=size) for size in self.conv_kernel_sizes]
            )
            self.correction_proj = nn.Sequential(
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
            final_linear = nn.Linear(dim_feedforward, output_len)
            # Zero-init the last projection (ResNet/Fixup-style residual init):
            # the forecast starts as the pure decomposition backbone, while all
            # earlier correction layers receive full gradients from step one.
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
            self.correction_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                final_linear,
            )
            self.correction_gate = nn.Parameter(torch.ones(1))

    def _same_length_conv(self, conv: nn.Conv1d, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        return conv(F.pad(x, (left, right), mode="replicate"))

    def _correction(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        features = x.clone()
        features[:, :, self.target_feature_index] = features[:, :, self.target_feature_index] - level
        if self.use_variable_attention:
            features = self.variable_attention(features)
        channels = features.transpose(1, 2)
        multi_scale = [
            self._same_length_conv(conv, channels, size)
            for conv, size in zip(self.correction_convs, self.conv_kernel_sizes, strict=True)
        ]
        z = torch.cat(multi_scale, dim=1).transpose(1, 2)
        z = self.correction_proj(z)
        z = self.positional_encoding(z)
        encoded = self.encoder(z)
        return self.correction_head(encoded.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, self.target_feature_index]
        prediction = self.target_backbone_scale * self.target_backbone(target_history)
        local_weight, recurrent_weight = torch.softmax(self.expert_logits, dim=0)
        expert_prediction = (
            local_weight * self.local_temporal_backbone(x)
            + recurrent_weight * self.recurrent_backbone(x)
        )
        prediction = prediction + self.expert_scale * expert_prediction
        if self.use_correction:
            if self.use_window_norm:
                level = target_history.mean(dim=1, keepdim=True)
            else:
                level = target_history.new_zeros(target_history.shape[0], 1)
            prediction = prediction + self.correction_gate * self._correction(x, level)
        return prediction
