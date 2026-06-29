from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_aggregate_daily_power_handles_missing_values_and_remainder():
    from src.data_preprocess import aggregate_daily_power

    raw = pd.DataFrame(
        {
            "Date": ["01/01/2007", "01/01/2007", "02/01/2007"],
            "Time": ["00:00:00", "00:01:00", "00:00:00"],
            "Global_active_power": ["1.0", "2.0", "3.0"],
            "Global_reactive_power": ["0.1", "?", "0.3"],
            "Voltage": ["240", "242", "244"],
            "Global_intensity": ["4", "5", "6"],
            "Sub_metering_1": ["1", "2", "3"],
            "Sub_metering_2": ["4", "5", "6"],
            "Sub_metering_3": ["7", "8", "9"],
            "RR": ["10", "", "20"],
            "NBJRR1": ["1", "", "2"],
            "NBJRR5": ["0", "", "1"],
            "NBJRR10": ["0", "", "0"],
            "NBJBROU": ["3", "", "4"],
        }
    )

    daily = aggregate_daily_power(raw)

    assert list(daily["date"].dt.strftime("%Y-%m-%d")) == ["2007-01-01", "2007-01-02"]
    first = daily.iloc[0]
    assert first["global_active_power"] == 3.0
    assert first["global_reactive_power"] == 0.2
    assert first["voltage"] == 241.0
    assert first["global_intensity"] == 4.5
    assert first["sub_metering_remainder"] == 23.0
    assert first["RR"] == 10.0
    assert first["is_weekend"] == 0


def test_aggregate_daily_power_rejects_unrecoverable_core_sensor_missing_values():
    from src.data_preprocess import aggregate_daily_power

    raw = pd.DataFrame(
        {
            "Date": ["01/01/2007", "01/01/2007"],
            "Time": ["00:00:00", "00:01:00"],
            "Global_active_power": ["?", "?"],
            "Global_reactive_power": ["?", "?"],
            "Voltage": ["?", "?"],
            "Global_intensity": ["?", "?"],
            "Sub_metering_1": ["?", "?"],
            "Sub_metering_2": ["?", "?"],
            "Sub_metering_3": ["?", "?"],
        }
    )

    with pytest.raises(ValueError, match="Unrecoverable missing values"):
        aggregate_daily_power(raw)


def test_build_windows_returns_expected_shapes_and_dates():
    from src.data_preprocess import build_windows

    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    target = np.arange(10, dtype=np.float32)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    X, y, target_dates = build_windows(
        features,
        target,
        dates,
        input_len=3,
        output_len=2,
        stride=2,
    )

    assert X.shape == (3, 3, 2)
    assert y.shape == (3, 2)
    assert target_dates.shape == (3, 2)
    np.testing.assert_array_equal(X[0], features[:3])
    np.testing.assert_array_equal(y[0], target[3:5])
    assert str(target_dates[0, 0])[:10] == "2020-01-04"


def test_split_rows_reserves_enough_validation_for_long_horizon_windows():
    from src.data_preprocess import _sample_mask_from_rows, _split_rows, build_windows

    days = 1442
    input_len = 90
    output_len = 365
    dates = pd.date_range("2006-12-16", periods=days, freq="D")
    daily = pd.DataFrame({"date": dates})
    features = np.zeros((days, 2), dtype=np.float32)
    target = np.zeros(days, dtype=np.float32)

    _, _, target_dates = build_windows(features, target, dates, input_len=input_len, output_len=output_len)
    train_rows, valid_rows, test_rows = _split_rows(daily, output_lens=(90, 365))

    valid_mask = _sample_mask_from_rows(target_dates, dates.to_numpy(dtype="datetime64[D]"), valid_rows)
    test_mask = _sample_mask_from_rows(target_dates, dates.to_numpy(dtype="datetime64[D]"), test_rows)

    assert train_rows.sum() > input_len + output_len
    assert valid_rows.sum() >= input_len + output_len
    assert test_rows.sum() >= input_len + output_len
    assert valid_mask.sum() > 0
    assert test_mask.sum() > 0


def test_power_window_dataset_returns_tensors_and_date_index(tmp_path: Path):
    from src.dataset import PowerWindowDataset

    npz_path = tmp_path / "train_90.npz"
    np.savez(
        npz_path,
        X=np.zeros((2, 90, 4), dtype=np.float32),
        y=np.ones((2, 90), dtype=np.float32),
        target_dates=np.array(
            [pd.date_range("2020-01-01", periods=90), pd.date_range("2020-04-01", periods=90)],
            dtype="datetime64[D]",
        ),
        feature_names=np.array(["a", "b", "c", "d"]),
    )

    dataset = PowerWindowDataset(npz_path)
    X, y, date_index = dataset[0]

    assert len(dataset) == 2
    assert X.shape == (90, 4)
    assert y.shape == (90,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32
    assert str(date_index[0])[:10] == "2020-01-01"


def test_models_emit_requested_prediction_horizon():
    from src.models.hybrid_model import HybridTCNTransformer
    from src.models.lstm import LSTMForecaster
    from src.models.transformer import TransformerForecaster

    batch = torch.randn(2, 90, 6)
    model_specs = [
        (LSTMForecaster, {"num_features": 6, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "output_len": 7}),
        (
            TransformerForecaster,
            {
                "num_features": 6,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "output_len": 7,
            },
        ),
        (
            HybridTCNTransformer,
            {
                "num_features": 6,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "output_len": 7,
            },
        ),
    ]

    for cls, kwargs in model_specs:
        model = cls(**kwargs)
        output = model(batch)
        assert output.shape == (2, 7)


def test_dmsaformer_components_and_model_emit_requested_prediction_horizon():
    from src.models.dmsaformer import (
        DMSAFormer,
        SeriesDecomposition,
        TargetDecompositionBackbone,
        VariableAttention,
    )

    batch = torch.randn(2, 90, 6)

    decomposition = SeriesDecomposition(kernel_size=5)
    trend, residual = decomposition(batch)
    assert trend.shape == batch.shape
    assert residual.shape == batch.shape
    torch.testing.assert_close(trend + residual, batch)

    attention = VariableAttention(num_features=6, hidden_dim=8)
    weighted = attention(batch)
    assert weighted.shape == batch.shape

    target_backbone = TargetDecompositionBackbone(input_len=90, output_len=7, kernel_size=5)
    target_output = target_backbone(batch[:, :, 0])
    assert target_output.shape == (2, 7)

    model = DMSAFormer(
        num_features=6,
        d_model=12,
        nhead=4,
        num_layers=1,
        dim_feedforward=24,
        dropout=0.0,
        output_len=7,
    )
    assert isinstance(model.target_backbone, TargetDecompositionBackbone)
    assert hasattr(model, "local_temporal_backbone")
    assert hasattr(model, "recurrent_backbone")
    output = model(batch)
    assert output.shape == (2, 7)
