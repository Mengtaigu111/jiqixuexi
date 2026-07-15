from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.dataset import make_dataloader
from src.evaluate import plot_error_curve, plot_prediction_curve
from src.models.hybrid_model import HybridTCNTransformer
from src.models.dmsaformer import SeriesDecomposition, VariableAttention
from src.models.transformer import PositionalEncoding
from src.train import build_model
from src.utils import compute_metrics, ensure_dir, inverse_transform_target


SEEDS = (2026, 2027, 2028, 2029, 2030)
DMSAFORMER_MODEL = "dmsaformer"
CALIBRATION_METHOD = "validation_affine_self"


class LegacyDMSAFormerCheckpoint(nn.Module):
    """Compatibility loader for pre-refactor DMSAFormer checkpoints."""

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
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.target_feature_index = int(target_feature_index)
        self.conv_kernel_sizes = tuple(int(size) for size in conv_kernel_sizes)
        self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
        self.trend_linear = nn.Linear(self.input_len, output_len)
        self.residual_linear = nn.Linear(self.input_len, output_len)
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
        self.correction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len),
        )
        self.correction_gate = nn.Parameter(torch.ones(1))

    def _same_length_conv(self, conv: nn.Conv1d, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        return conv(F.pad(x, (left, right), mode="replicate"))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, self.target_feature_index]
        series = self._resize_to_input_len(target_history)
        level = series.mean(dim=1, keepdim=True)
        centered = series - level
        trend, residual = self.decomposition(centered.unsqueeze(-1))
        prediction = self.trend_linear(trend.squeeze(-1)) + self.residual_linear(residual.squeeze(-1)) + level

        features = x.clone()
        features[:, :, self.target_feature_index] = features[:, :, self.target_feature_index] - level
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
        return prediction + self.correction_gate * self.correction_head(encoded.mean(dim=1))


class LegacyV2TargetBackbone(nn.Module):
    """DLinear-style target backbone used by the v2 DMSAFormer checkpoints."""

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
        target = self._resize_to_input_len(target_history).unsqueeze(-1)
        trend, residual = self.decomposition(target)
        return self.trend_linear(trend.squeeze(-1)) + self.residual_linear(residual.squeeze(-1))


class LegacyV2DMSAFormerCheckpoint(nn.Module):
    """Compatibility loader for DMSAFormer v2 checkpoints."""

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
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.target_feature_index = int(target_feature_index)
        self.conv_kernel_sizes = tuple(int(size) for size in conv_kernel_sizes)
        self.variable_attention = VariableAttention(num_features, hidden_dim=max(8, d_model // 2))
        self.decomposition = SeriesDecomposition(kernel_size=decomposition_kernel)
        self.target_backbone = LegacyV2TargetBackbone(
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
        self.residual_convs = nn.ModuleList(
            [nn.Conv1d(num_features, d_model, kernel_size=size) for size in self.conv_kernel_sizes]
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
        self.target_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.correction_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _same_length_conv(self, conv: nn.Conv1d, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        return conv(F.pad(x, (left, right), mode="replicate"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, self.target_feature_index]
        y_backbone = self.target_backbone(target_history)
        y_local = self.local_temporal_backbone(x)
        attended = self.variable_attention(x)
        _, residual = self.decomposition(attended)
        residual_channels = residual.transpose(1, 2)
        multi_scale = [
            self._same_length_conv(conv, residual_channels, size)
            for conv, size in zip(self.residual_convs, self.conv_kernel_sizes, strict=True)
        ]
        z = torch.cat(multi_scale, dim=1).transpose(1, 2)
        z = self.residual_proj(z)
        z = self.positional_encoding(z)
        encoded = self.encoder(z)
        y_residual = self.residual_head(encoded.mean(dim=1))
        return y_local + self.target_scale * y_backbone + self.correction_scale * y_residual


def fit_affine_calibration(prediction: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> tuple[float, float]:
    matrix = np.stack([prediction.reshape(-1), np.ones(prediction.size, dtype=prediction.dtype)], axis=1)
    target_vector = target.reshape(-1)
    penalty = np.diag([ridge, 0.0]).astype(matrix.dtype)
    scale, bias = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target_vector)
    return float(scale), float(bias)


def _is_legacy_dmsaformer_state(state_dict: dict[str, torch.Tensor]) -> bool:
    return "trend_linear.weight" in state_dict and "correction_convs.0.weight" in state_dict


def _is_v2_legacy_dmsaformer_state(state_dict: dict[str, torch.Tensor]) -> bool:
    return "target_scale" in state_dict and "residual_convs.0.weight" in state_dict


def build_model_for_checkpoint(checkpoint: dict) -> nn.Module:
    kwargs = checkpoint.get("model_kwargs", {})
    model_name = checkpoint.get("model_name", DMSAFORMER_MODEL)
    state_dict = checkpoint.get("model_state_dict", {})
    if model_name == DMSAFORMER_MODEL and _is_legacy_dmsaformer_state(state_dict):
        return LegacyDMSAFormerCheckpoint(
            checkpoint["num_features"],
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 128),
            dropout=kwargs.get("dropout", 0.1),
            output_len=checkpoint["output_len"],
        )
    if model_name == DMSAFORMER_MODEL and _is_v2_legacy_dmsaformer_state(state_dict):
        return LegacyV2DMSAFormerCheckpoint(
            checkpoint["num_features"],
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 128),
            dropout=kwargs.get("dropout", 0.1),
            output_len=checkpoint["output_len"],
        )
    return build_model(
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


@torch.no_grad()
def predict_checkpoint(
    model_name: str,
    horizon: int,
    seed: int,
    split: str,
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}_{horizon}_seed{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model_for_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = make_dataloader(Path(data_dir) / f"{split}_{horizon}.npz", batch_size=batch_size, shuffle=False)
    y_true, y_pred, dates = [], [], []
    for X, y, batch_dates in loader:
        y_true.append(y.numpy())
        y_pred.append(model(X).numpy())
        dates.append(batch_dates)
    if not y_true:
        raise ValueError(f"No samples available in {split}_{horizon}.npz")
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(dates)


def _original_scale_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_bundle) -> dict[str, float]:
    return compute_metrics(
        inverse_transform_target(y_true_scaled, scaler_bundle),
        inverse_transform_target(y_pred_scaled, scaler_bundle),
    )


def _write_prediction_outputs(
    horizon: int,
    seed: int,
    source_model: str,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_dates: np.ndarray,
    scaler_bundle,
    predictions_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    calibration_scale: float | None,
    calibration_bias: float | None,
    calibration_method: str,
) -> dict[str, object]:
    y_true = inverse_transform_target(y_true_scaled, scaler_bundle)
    y_pred = inverse_transform_target(y_pred_scaled, scaler_bundle)
    metrics = compute_metrics(y_true, y_pred)
    pred_path = predictions_dir / f"dmsaformer_{horizon}_seed{seed}.csv"
    metric_path = metrics_dir / f"dmsaformer_{horizon}_seed{seed}_test_metrics.csv"

    rows = []
    for sample_idx in range(y_true.shape[0]):
        for step in range(y_true.shape[1]):
            rows.append(
                {
                    "model": "dmsaformer",
                    "source_model": source_model,
                    "output_len": horizon,
                    "seed": seed,
                    "sample_index": sample_idx,
                    "step": step + 1,
                    "date": str(target_dates[sample_idx, step])[:10],
                    "y_true": float(y_true[sample_idx, step]),
                    "y_pred": float(y_pred[sample_idx, step]),
                    "calibration_scale": calibration_scale,
                    "calibration_bias": calibration_bias,
                    "calibration_method": calibration_method,
                }
            )
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "dmsaformer",
                "output_len": horizon,
                "seed": seed,
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "predictions": str(pred_path),
            }
        ]
    ).to_csv(metric_path, index=False)

    first_dates = pd.to_datetime(target_dates[0])
    first_metrics = compute_metrics(y_true[0], y_pred[0])
    plot_prediction_curve(
        first_dates,
        y_true[0],
        y_pred[0],
        "dmsaformer",
        horizon,
        seed,
        first_metrics["mse"],
        first_metrics["mae"],
        figures_dir / f"dmsaformer_{horizon}_seed{seed}_curve.png",
    )
    plot_error_curve(
        first_dates,
        y_pred[0] - y_true[0],
        "dmsaformer",
        horizon,
        seed,
        figures_dir / f"dmsaformer_{horizon}_seed{seed}_error.png",
    )
    return {
        "horizon": horizon,
        "seed": seed,
        "source_model": source_model,
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
        "calibration_method": calibration_method,
    }


def export_calibrated_dmsaformer(
    checkpoint_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data/processed",
    scaler_path: str | Path = "data/processed/scaler.pkl",
    predictions_dir: str | Path = "results/predictions",
    metrics_dir: str | Path = "results/metrics",
    figures_dir: str | Path = "results/figures",
) -> pd.DataFrame:
    scaler_bundle = joblib.load(scaler_path)
    predictions_path = ensure_dir(predictions_dir)
    metrics_path = ensure_dir(metrics_dir)
    figures_path = ensure_dir(figures_dir)

    choice_rows = []
    valid_rows = []
    for horizon in (90, 365):
        for seed in SEEDS:
            y_valid, pred_valid, _ = predict_checkpoint(DMSAFORMER_MODEL, horizon, seed, "valid", checkpoint_dir, data_dir)
            pre_metrics = _original_scale_metrics(y_valid, pred_valid, scaler_bundle)
            scale, bias = fit_affine_calibration(pred_valid, y_valid)
            calibrated_valid = scale * pred_valid + bias
            post_metrics = _original_scale_metrics(y_valid, calibrated_valid, scaler_bundle)
            valid_rows.append(
                {
                    "seed": seed,
                    "model": DMSAFORMER_MODEL,
                    "horizon": horizon,
                    "calibration_method": CALIBRATION_METHOD,
                    "valid_mse_before": pre_metrics["mse"],
                    "valid_mae_before": pre_metrics["mae"],
                    "valid_mse_after": post_metrics["mse"],
                    "valid_mae_after": post_metrics["mae"],
                    "calibration_scale": scale,
                    "calibration_bias": bias,
                }
            )
            y_test, pred_test, dates = predict_checkpoint(DMSAFORMER_MODEL, horizon, seed, "test", checkpoint_dir, data_dir)
            calibrated_test = scale * pred_test + bias
            choice_rows.append(
                _write_prediction_outputs(
                    horizon,
                    seed,
                    DMSAFORMER_MODEL,
                    y_test,
                    calibrated_test,
                    dates,
                    scaler_bundle,
                    predictions_path,
                    metrics_path,
                    figures_path,
                    calibration_scale=scale,
                    calibration_bias=bias,
                    calibration_method=CALIBRATION_METHOD,
                )
            )

    choices = pd.DataFrame(choice_rows)
    choices.to_csv(metrics_path / "dmsaformer_calibration_choices.csv", index=False)
    pd.DataFrame(valid_rows).to_csv(metrics_path / "dmsaformer_self_calibration_validation_metrics.csv", index=False)
    return choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validation-calibrated DMSAFormer self predictions.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--scaler_path", default="data/processed/scaler.pkl")
    parser.add_argument("--predictions_dir", default="results/predictions")
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    return parser.parse_args()


def main() -> None:
    choices = export_calibrated_dmsaformer(**vars(parse_args()))
    print(choices.to_string(index=False))


if __name__ == "__main__":
    main()
