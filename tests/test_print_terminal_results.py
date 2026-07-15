import csv
import subprocess
import sys
from pathlib import Path


def write_metric(path: Path, model: str, horizon: int, seed: int, mse: float, mae: float):
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [
                ["model", "output_len", "seed", "test_mse", "test_mae", "predictions"],
                [model, horizon, seed, mse, mae, f"results/predictions/{model}_{horizon}_seed{seed}.csv"],
            ]
        )


def test_print_terminal_results_reads_per_seed_csvs_and_prints_each_run(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    summary = metrics_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [
                ["Model", "Horizon", "MSE mean", "MSE std", "MAE mean", "MAE std", "Runs"],
                ["transformer", "90", "154133.13478416374", "2800.310094402", "302.49097110942445", "3.899", "2"],
                ["lstm", "365", "295920.885338033", "0", "428.5710935327525", "0", "1"],
                ["dmsaformer", "365", "285797.318624635", "0", "420.8585150281066", "0", "1"],
            ]
        )
    write_metric(metrics_dir / "transformer_90_seed2026_test_metrics.csv", "transformer", 90, 2026, 156113.254481926, 305.2473367057872)
    write_metric(metrics_dir / "transformer_90_seed2027_test_metrics.csv", "transformer", 90, 2027, 152153.0150864015, 299.7346055130617)
    write_metric(metrics_dir / "lstm_365_seed2026_test_metrics.csv", "lstm", 365, 2026, 295920.885338033, 428.5710935327525)
    write_metric(metrics_dir / "dmsaformer_365_seed2026_test_metrics.csv", "dmsaformer", 365, 2026, 285797.318624635, 420.8585150281066)
    write_metric(metrics_dir / "hybrid_90_seed2026_test_metrics.csv", "hybrid", 90, 2026, 100.0, 10.0)

    result = subprocess.run(
        [sys.executable, "print_terminal_results.py", "--metrics-dir", str(metrics_dir), "--summary", str(summary)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout
    assert f"单次结果目录: {metrics_dir}" in output
    assert "实验规模: 4 次单独实验" in output
    assert f"正式模型过滤: {summary}" in output
    assert "90 天预测单次结果" in output
    assert "365 天预测单次结果" in output
    assert "Transformer  seed=2026  MSE=156113.25" in output
    assert "Transformer  seed=2027  MSE=152153.02" in output
    assert "LSTM         seed=2026  MSE=295920.89" in output
    assert "DMSAFormer   seed=2026  MSE=285797.32" in output
    assert "hybrid" not in output
    assert "结论: 已列出每个 seed 的单次 test MSE/MAE。" in output
