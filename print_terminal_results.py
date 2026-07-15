#!/usr/bin/env python3
"""Print final experiment results for a terminal screenshot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "Model",
    "Horizon",
    "MSE mean",
    "MSE std",
    "MAE mean",
    "MAE std",
    "Runs",
}
RUN_COLUMNS = {
    "model",
    "output_len",
    "seed",
    "test_mse",
    "test_mae",
    "predictions",
}

MODEL_NAMES = {
    "dmsaformer": "DMSAFormer",
    "lstm": "LSTM",
    "transformer": "Transformer",
}


def model_label(name: str) -> str:
    return MODEL_NAMES.get(name.lower(), name)


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"结果文件缺少列: {', '.join(sorted(missing))}")
        return list(reader)


def read_single_run_metrics(metrics_dir: Path) -> list[dict[str, str]]:
    if not metrics_dir.exists():
        raise FileNotFoundError(f"找不到单次结果目录: {metrics_dir}")

    rows: list[dict[str, str]] = []
    for path in sorted(metrics_dir.glob("*_test_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = RUN_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} 缺少列: {', '.join(sorted(missing))}")
            file_rows = list(reader)
            if len(file_rows) != 1:
                raise ValueError(f"{path} 应该只包含 1 行 test 指标，实际为 {len(file_rows)} 行")
            rows.append(file_rows[0])

    if not rows:
        raise ValueError(f"{metrics_dir} 下没有找到 *_test_metrics.csv")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def print_single_runs(rows: list[dict[str, str]], metrics_dir: Path, summary_filter: Path | None = None) -> None:
    by_horizon: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_horizon[as_int(row, "output_len")].append(row)

    print("=" * 78)
    print("家庭电力消耗预测实验单次结果")
    print("口径: 原始 test 指标，未做后处理校准")
    print(f"单次结果目录: {metrics_dir}")
    if summary_filter is not None:
        print(f"正式模型过滤: {summary_filter}")
    print(f"实验规模: {len(rows)} 次单独实验")
    print("=" * 78)

    for horizon in sorted(by_horizon):
        ranked = sorted(
            by_horizon[horizon],
            key=lambda row: (row["model"].lower(), as_int(row, "seed")),
        )
        print()
        print(f"{horizon} 天预测单次结果")
        print("-" * 78)
        for row in ranked:
            model = model_label(row["model"])
            seed = as_int(row, "seed")
            mse = as_float(row, "test_mse")
            mae = as_float(row, "test_mae")
            print(f"{model:<12} seed={seed}  MSE={mse:<12.2f}  MAE={mae:<9.2f}")

    print()
    print("结论: 已列出每个 seed 的单次 test MSE/MAE。")


def filter_runs_by_summary(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed = {
        (row["Model"].lower(), as_int(row, "Horizon"))
        for row in summary_rows
    }
    return [
        row
        for row in rows
        if (row["model"].lower(), as_int(row, "output_len")) in allowed
    ]


def print_results(rows: list[dict[str, str]], metrics_path: Path) -> None:
    by_horizon: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_horizon[as_int(row, "Horizon")].append(row)

    total_runs = sum(as_int(row, "Runs") for row in rows)

    print("=" * 72)
    print("家庭电力消耗预测实验结果汇总")
    print("口径: 原始 test 指标，未做后处理校准")
    print(f"数据源: {metrics_path}")
    print(f"实验规模: {len(rows)} 组模型/预测长度组合，合计 {total_runs} 次正式实验")
    print("=" * 72)

    best_by_horizon: dict[int, str] = {}
    for horizon in sorted(by_horizon):
        ranked = sorted(by_horizon[horizon], key=lambda row: as_float(row, "MSE mean"))
        print()
        print(f"{horizon} 天预测结果")
        print("-" * 72)
        for rank, row in enumerate(ranked, start=1):
            model = model_label(row["Model"])
            mse = format_metric(as_float(row, "MSE mean"), as_float(row, "MSE std"))
            mae = format_metric(as_float(row, "MAE mean"), as_float(row, "MAE std"))
            runs = as_int(row, "Runs")
            print(f"{rank}. {model:<12} MSE={mse:<22} MAE={mae:<18} Runs={runs}")
        best_by_horizon[horizon] = model_label(ranked[0]["Model"])

    conclusion_parts = [
        f"{horizon} 天最优模型为 {best_by_horizon[horizon]}"
        for horizon in sorted(best_by_horizon)
    ]
    print()
    print("结论: " + "；".join(conclusion_parts) + "。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print final test metrics for screenshot capture.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/metrics/summary.csv"),
        help="Path to summary.csv generated from raw test metrics.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory containing *_test_metrics.csv files for each seed.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the mean/std summary table.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Print all *_test_metrics.csv files, including models not listed in summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        rows = read_summary(args.summary)
        print_results(rows, args.summary)
        return

    rows = read_single_run_metrics(args.metrics_dir)
    summary_filter = None
    if not args.all_models and args.summary.exists():
        summary_rows = read_summary(args.summary)
        rows = filter_runs_by_summary(rows, summary_rows)
        if not rows:
            raise ValueError(f"按 {args.summary} 过滤后没有可打印的单次结果")
        summary_filter = args.summary
    print_single_runs(rows, args.metrics_dir, summary_filter)


if __name__ == "__main__":
    main()
