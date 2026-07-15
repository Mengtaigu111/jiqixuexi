from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import torch


def _pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


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


def test_calibrated_dmsaformer_helper_fits_affine_calibration():
    from src.calibrated_dmsaformer import fit_affine_calibration

    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = 2.0 * pred + 0.5

    scale, bias = fit_affine_calibration(pred, target, ridge=0.0)

    assert abs(scale - 2.0) < 1e-6
    assert abs(bias - 0.5) < 1e-6


def test_calibrated_dmsaformer_export_uses_only_dmsaformer_predictions(monkeypatch, tmp_path: Path):
    import src.calibrated_dmsaformer as calibrated

    calls: list[tuple[str, int, int, str]] = []

    def fake_predict_checkpoint(model_name, horizon, seed, split, checkpoint_dir, data_dir):
        calls.append((model_name, int(horizon), int(seed), split))
        y_true = np.array([[1.0, 2.0]], dtype=np.float32)
        if split == "valid":
            y_pred = np.array([[0.25, 0.75]], dtype=np.float32)
        else:
            y_pred = np.array([[0.5, 1.0]], dtype=np.float32)
        dates = np.array([["2020-01-01", "2020-01-02"]])
        return y_true, y_pred, dates

    monkeypatch.setattr(calibrated, "SEEDS", (2026,))
    monkeypatch.setattr(calibrated.joblib, "load", lambda _: {})
    monkeypatch.setattr(calibrated, "predict_checkpoint", fake_predict_checkpoint)
    monkeypatch.setattr(calibrated, "plot_prediction_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibrated, "plot_error_curve", lambda *args, **kwargs: None)

    choices = calibrated.export_calibrated_dmsaformer(
        checkpoint_dir=tmp_path / "checkpoints",
        data_dir=tmp_path / "data",
        scaler_path=tmp_path / "scaler.pkl",
        predictions_dir=tmp_path / "predictions",
        metrics_dir=tmp_path / "metrics",
        figures_dir=tmp_path / "figures",
    )

    assert set(choices["source_model"]) == {"dmsaformer"}
    assert set(choices["calibration_method"]) == {"validation_affine_self"}
    assert set(choices["horizon"]) == {90, 365}
    assert {(model, horizon, split) for model, horizon, _, split in calls} == {
        ("dmsaformer", 90, "valid"),
        ("dmsaformer", 90, "test"),
        ("dmsaformer", 365, "valid"),
        ("dmsaformer", 365, "test"),
    }

    prediction = pd.read_csv(tmp_path / "predictions" / "dmsaformer_90_seed2026.csv")
    assert set(prediction["source_model"]) == {"dmsaformer"}
    assert set(prediction["calibration_method"]) == {"validation_affine_self"}


def test_calibrated_dmsaformer_builds_legacy_model_for_old_checkpoint_keys():
    from src.calibrated_dmsaformer import build_model_for_checkpoint

    checkpoint = {
        "model_name": "dmsaformer",
        "num_features": 3,
        "output_len": 2,
        "model_kwargs": {
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        "model_state_dict": {
            "trend_linear.weight": torch.zeros((2, 90)),
            "correction_convs.0.weight": torch.zeros((8, 3, 3)),
        },
    }

    model = build_model_for_checkpoint(checkpoint)
    output = model(torch.zeros((1, 90, 3), dtype=torch.float32))

    assert hasattr(model, "trend_linear")
    assert output.shape == (1, 2)


def test_calibrated_dmsaformer_builds_v2_legacy_model_for_refactored_checkpoint_keys():
    from src.calibrated_dmsaformer import build_model_for_checkpoint

    checkpoint = {
        "model_name": "dmsaformer",
        "num_features": 3,
        "output_len": 2,
        "model_kwargs": {
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        "model_state_dict": {
            "target_scale": torch.ones(()),
            "correction_scale": torch.ones(()),
            "target_backbone.trend_linear.1.weight": torch.zeros((2, 90)),
            "residual_convs.0.weight": torch.zeros((8, 3, 3)),
        },
    }

    model = build_model_for_checkpoint(checkpoint)
    output = model(torch.zeros((1, 90, 3), dtype=torch.float32))

    assert hasattr(model, "target_scale")
    assert hasattr(model, "local_temporal_backbone")
    assert output.shape == (1, 2)


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


def test_generate_pdf_report_includes_dense_course_report_sections(tmp_path: Path):
    from src.generate_report_pdf import build_report_text_pages

    metrics_dir = tmp_path / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Model": "dmsaformer", "Horizon": 90, "MSE mean": 163105.98, "MSE std": 6543.60, "MAE mean": 312.37, "MAE std": 6.08, "Runs": 5},
            {"Model": "lstm", "Horizon": 90, "MSE mean": 163266.74, "MSE std": 1593.57, "MAE mean": 312.45, "MAE std": 2.17, "Runs": 5},
            {"Model": "transformer", "Horizon": 90, "MSE mean": 156632.34, "MSE std": 3657.48, "MAE mean": 305.30, "MAE std": 4.98, "Runs": 5},
            {"Model": "dmsaformer", "Horizon": 365, "MSE mean": 294854.83, "MSE std": 31206.14, "MAE mean": 430.28, "MAE std": 27.09, "Runs": 5},
            {"Model": "lstm", "Horizon": 365, "MSE mean": 316352.06, "MSE std": 16271.97, "MAE mean": 446.40, "MAE std": 12.26, "Runs": 5},
            {"Model": "transformer", "Horizon": 365, "MSE mean": 442238.94, "MSE std": 45530.66, "MAE mean": 545.82, "MAE std": 32.61, "Runs": 5},
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    pages = build_report_text_pages(metrics_dir / "summary.csv")
    combined = "\n".join(f"{page.title}\n{page.body}" for page in pages)

    for required_phrase in [
        "数据预处理与样本构造",
        "MSE",
        "MAE",
        "趋势分解模块",
        "多尺度卷积模块",
        "变量注意力模块",
        "validation-calibrated self",
        "相对提升",
        "稳定性分析",
        "预测曲线分析",
        "尖峰负荷响应不足",
    ]:
        assert required_phrase in combined

    assert len(pages) >= 8
    assert "四种模型" not in combined
    assert "四模型" not in combined
    assert "40 次" not in combined
    assert "30 次" in combined


def test_generated_pdf_has_enough_pages_for_course_report(tmp_path: Path):
    from src.generate_report_pdf import generate_pdf

    report_dir = tmp_path / "report"
    metrics_dir = tmp_path / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Model": "dmsaformer", "Horizon": 90, "MSE mean": 163105.98, "MSE std": 6543.60, "MAE mean": 312.37, "MAE std": 6.08, "Runs": 5},
            {"Model": "lstm", "Horizon": 90, "MSE mean": 163266.74, "MSE std": 1593.57, "MAE mean": 312.45, "MAE std": 2.17, "Runs": 5},
            {"Model": "transformer", "Horizon": 90, "MSE mean": 156632.34, "MSE std": 3657.48, "MAE mean": 305.30, "MAE std": 4.98, "Runs": 5},
            {"Model": "dmsaformer", "Horizon": 365, "MSE mean": 294854.83, "MSE std": 31206.14, "MAE mean": 430.28, "MAE std": 27.09, "Runs": 5},
            {"Model": "lstm", "Horizon": 365, "MSE mean": 316352.06, "MSE std": 16271.97, "MAE mean": 446.40, "MAE std": 12.26, "Runs": 5},
            {"Model": "transformer", "Horizon": 365, "MSE mean": 442238.94, "MSE std": 45530.66, "MAE mean": 545.82, "MAE std": 32.61, "Runs": 5},
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    out_path = generate_pdf(report_dir=report_dir, results_dir=tmp_path / "results")

    assert 12 <= _pdf_page_count(out_path) <= 15


def test_html_report_renderer_produces_formal_course_report_layout(tmp_path: Path):
    from src.generate_report_html_pdf import build_report_html, write_report_html

    metrics_dir = tmp_path / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Model": "dmsaformer", "Horizon": 90, "MSE mean": 163105.98, "MSE std": 6543.60, "MAE mean": 312.37, "MAE std": 6.08, "Runs": 5},
            {"Model": "transformer", "Horizon": 90, "MSE mean": 156632.34, "MSE std": 3657.48, "MAE mean": 305.30, "MAE std": 4.98, "Runs": 5},
            {"Model": "dmsaformer", "Horizon": 365, "MSE mean": 294854.83, "MSE std": 31206.14, "MAE mean": 430.28, "MAE std": 27.09, "Runs": 5},
            {"Model": "lstm", "Horizon": 365, "MSE mean": 316352.06, "MSE std": 16271.97, "MAE mean": 446.40, "MAE std": 12.26, "Runs": 5},
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    html = build_report_html(report_dir=tmp_path / "report", results_dir=tmp_path / "results")

    for required in [
        'class="cover"',
        'class="toc-page"',
        'class="report-shell"',
        'class="booktabs"',
        'class="figure-caption"',
        'class="running-header"',
        'class="page-footer"',
        "基于深度学习的家庭电力消耗多变量时间序列预测",
        "数据预处理与样本构造",
        "DMSAFormer 校准策略与伪代码",
        "MSE 与 MAE 指标对比",
    ]:
        assert required in html

    assert "counter(" not in html
    assert "box-shadow" not in html
    assert "border-radius" not in html
    assert "四种模型" not in html
    assert "四模型" not in html
    assert "40 次" not in html
    assert "30 次" in html

    out_path = write_report_html(report_dir=tmp_path / "report", results_dir=tmp_path / "results")
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == html


def test_html_report_pdf_converter_entrypoint_uses_local_playwright_script(tmp_path: Path):
    from src.generate_report_html_pdf import html_to_pdf_command

    html_path = tmp_path / "report.html"
    pdf_path = tmp_path / "report.pdf"
    html_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    command = html_to_pdf_command(html_path, pdf_path)

    assert command[0] == "node"
    assert Path(command[1]).name == "html_to_pdf.js"
    assert Path(command[1]).exists()
    assert command[-2:] == [str(html_path), str(pdf_path)]


def test_html_report_starts_references_on_separate_page(tmp_path: Path):
    from src.generate_report_html_pdf import build_report_html

    metrics_dir = tmp_path / "results" / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Model": "dmsaformer", "Horizon": 90, "MSE mean": 163105.98, "MSE std": 6543.60, "MAE mean": 312.37, "MAE std": 6.08, "Runs": 5},
            {"Model": "transformer", "Horizon": 90, "MSE mean": 156632.34, "MSE std": 3657.48, "MAE mean": 305.30, "MAE std": 4.98, "Runs": 5},
            {"Model": "dmsaformer", "Horizon": 365, "MSE mean": 294854.83, "MSE std": 31206.14, "MAE mean": 430.28, "MAE std": 27.09, "Runs": 5},
            {"Model": "lstm", "Horizon": 365, "MSE mean": 316352.06, "MSE std": 16271.97, "MAE mean": 446.40, "MAE std": 12.26, "Runs": 5},
        ]
    ).to_csv(metrics_dir / "summary.csv", index=False)

    html = build_report_html(report_dir=tmp_path / "report", results_dir=tmp_path / "results")

    assert '<section id="sec-11" class="page-break-before">' in html
