from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch


def test_compute_metrics_reports_mse_and_mae():
    from src.utils import compute_metrics

    metrics = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 2.0]))

    assert metrics["mse"] == 5.0 / 3.0
    assert metrics["mae"] == 1.0


def test_build_model_dispatches_all_required_models():
    from src.train import build_model

    for model_name in ["lstm", "transformer", "hybrid", "dmsaformer"]:
        model = build_model(
            model_name=model_name,
            num_features=5,
            output_len=3,
            hidden_size=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        )
        assert model(torch.zeros((1, 90, 5), dtype=torch.float32)).shape == (1, 3)


def test_train_dmsaformer_wrapper_maps_pred_len_to_existing_train_args():
    import train_dmsaformer

    args = train_dmsaformer.parse_args(["--pred_len", "90", "--seed", "2026", "--epochs", "1"])

    assert args.model == "dmsaformer"
    assert args.output_len == 90
    assert args.seed == 2026
    assert args.epochs == 1


def test_calibrated_dmsaformer_helpers_fit_affine_and_gate_short_horizon_experts():
    from src.calibrated_dmsaformer import fit_affine_calibration, select_short_horizon_experts

    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = 2.0 * pred + 0.5

    scale, bias = fit_affine_calibration(pred, target, ridge=0.0)

    assert abs(scale - 2.0) < 1e-6
    assert abs(bias - 0.5) < 1e-6

    valid_metrics = pd.DataFrame(
        [
            {"seed": 1, "model": "hybrid", "valid_mse": 100.0},
            {"seed": 1, "model": "transformer", "valid_mse": 99.0},
            {"seed": 2, "model": "hybrid", "valid_mse": 130.0},
            {"seed": 2, "model": "transformer", "valid_mse": 60.0},
            {"seed": 3, "model": "hybrid", "valid_mse": 160.0},
            {"seed": 3, "model": "transformer", "valid_mse": 159.0},
        ]
    )

    choices = select_short_horizon_experts(valid_metrics, threshold_multiplier=1.0)

    assert choices == {1: "hybrid", 2: "transformer", 3: "hybrid"}


def test_summarize_metrics_groups_by_model_and_horizon(tmp_path: Path):
    from src.summarize_results import summarize_metrics

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {"model": "lstm", "output_len": 90, "seed": 2026, "test_mse": 4.0, "test_mae": 2.0},
            {"model": "lstm", "output_len": 90, "seed": 2027, "test_mse": 6.0, "test_mae": 4.0},
            {"model": "hybrid", "output_len": 365, "seed": 2026, "test_mse": 9.0, "test_mae": 3.0},
        ]
    ).to_csv(metrics_dir / "test_metrics.csv", index=False)

    summary = summarize_metrics(metrics_dir)

    lstm = summary[(summary["Model"] == "lstm") & (summary["Horizon"] == 90)].iloc[0]
    assert lstm["MSE mean"] == 5.0
    assert round(lstm["MSE std"], 6) == round(np.std([4.0, 6.0], ddof=1), 6)
    assert lstm["MAE mean"] == 3.0
    assert set(summary["Model"]) == {"lstm", "hybrid"}


def test_summarize_metrics_can_filter_to_requested_models(tmp_path: Path):
    from src.summarize_results import summarize_metrics

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {"model": "lstm", "output_len": 90, "seed": 2026, "test_mse": 4.0, "test_mae": 2.0},
            {"model": "hybrid", "output_len": 90, "seed": 2026, "test_mse": 3.0, "test_mae": 1.5},
            {"model": "dmsaformer", "output_len": 90, "seed": 2026, "test_mse": 9.0, "test_mae": 4.5},
        ]
    ).to_csv(metrics_dir / "test_metrics.csv", index=False)

    summary = summarize_metrics(metrics_dir, model_names=["lstm", "hybrid"])

    assert set(summary["Model"]) == {"lstm", "hybrid"}
    assert "dmsaformer" not in set(summary["Model"])


def test_write_summary_outputs_does_not_require_tabulate(tmp_path: Path):
    from src.summarize_results import write_summary_outputs

    summary = pd.DataFrame(
        [
            {
                "Model": "lstm",
                "Horizon": 90,
                "MSE mean": 1.0,
                "MSE std": 0.1,
                "MAE mean": 0.5,
                "MAE std": 0.05,
                "Runs": 2,
            }
        ]
    )

    _, md_path = write_summary_outputs(summary, tmp_path)

    text = md_path.read_text(encoding="utf-8")
    assert "| Model | Horizon | MSE mean |" in text
    assert "| lstm | 90 | 1.0 |" in text


def test_plot_prediction_curve_writes_png(tmp_path: Path):
    from src.evaluate import plot_prediction_curve

    out_path = tmp_path / "curve.png"
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    plot_prediction_curve(
        dates=dates,
        y_true=np.array([1, 2, 3, 4, 5], dtype=float),
        y_pred=np.array([1, 2, 2, 4, 6], dtype=float),
        model_name="lstm",
        output_len=5,
        seed=2026,
        mse=0.4,
        mae=0.2,
        out_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_export_dmsaformer_artifacts_writes_required_goal_files(tmp_path: Path):
    from src.export_dmsaformer_artifacts import export_dmsaformer_artifacts

    metrics_dir = tmp_path / "results" / "metrics"
    figures_dir = tmp_path / "results" / "figures"
    predictions_dir = tmp_path / "results" / "predictions"
    root_figures_dir = tmp_path / "figures"
    metrics_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    for horizon in [90, 365]:
        pd.DataFrame(
            [
                {
                    "model": "dmsaformer",
                    "output_len": horizon,
                    "seed": 2026,
                    "test_mse": float(horizon),
                    "test_mae": float(horizon) / 10,
                    "predictions": f"pred_{horizon}.csv",
                }
            ]
        ).to_csv(metrics_dir / f"dmsaformer_{horizon}_seed2026_test_metrics.csv", index=False)
        (figures_dir / f"dmsaformer_{horizon}_seed2026_curve.png").write_bytes(b"png")
        pd.DataFrame(
            [
                {
                    "model": "dmsaformer",
                    "output_len": horizon,
                    "seed": 2026,
                    "sample_index": 0,
                    "step": step,
                    "date": f"2020-01-{step:02d}",
                    "y_true": float(step),
                    "y_pred": float(step) + 0.5,
                }
                for step in range(1, min(horizon, 3) + 1)
            ]
        ).to_csv(predictions_dir / f"dmsaformer_{horizon}_seed2026.csv", index=False)

    pd.DataFrame(
        [
            {
                "Model": "dmsaformer",
                "Horizon": 90,
                "MSE mean": 90.0,
                "MSE std": 0.0,
                "MAE mean": 9.0,
                "MAE std": 0.0,
                "Runs": 1,
            }
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    export_dmsaformer_artifacts(
        results_dir=tmp_path / "results",
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
        root_figures_dir=root_figures_dir,
    )

    assert (tmp_path / "results" / "dmsaformer_90_results.csv").exists()
    assert (tmp_path / "results" / "dmsaformer_365_results.csv").exists()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (root_figures_dir / "dmsaformer_90_prediction.png").stat().st_size > 1000
    assert (root_figures_dir / "dmsaformer_365_prediction.png").stat().st_size > 1000


def test_generate_pdf_also_writes_required_report_pdf_name(tmp_path: Path):
    from src.generate_report_pdf import generate_pdf

    report_dir = tmp_path / "report"
    metrics_dir = tmp_path / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Model": "lstm",
                "Horizon": 90,
                "MSE mean": 1.0,
                "MSE std": 0.1,
                "MAE mean": 0.5,
                "MAE std": 0.05,
                "Runs": 5,
            }
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_path = generate_pdf(report_dir=report_dir, results_dir=tmp_path / "results")

    required_path = report_dir / "report.pdf"
    assert out_path == report_dir / "ML_household_power_report.pdf"
    assert out_path.exists()
    assert required_path.exists()
    assert required_path.stat().st_size == out_path.stat().st_size
    assert not [warning for warning in caught if "Glyph" in str(warning.message)]
