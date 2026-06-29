from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_dir

TARGET_COL = "global_active_power"
POWER_SUM_COLS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
POWER_MEAN_COLS = ["voltage", "global_intensity"]
CORE_SENSOR_COLS = POWER_SUM_COLS + POWER_MEAN_COLS
WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
TIME_FEATURES = [
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
]


@dataclass(frozen=True)
class PreprocessResult:
    daily_path: Path
    scaler_path: Path
    generated_files: list[Path]
    feature_names: list[str]


def _canonical_column(name: str) -> str:
    cleaned = name.strip().replace(" ", "_")
    lower = cleaned.lower()
    mapping = {
        "date": "date",
        "time": "time",
        "datetime": "datetime",
        "timestamp": "datetime",
        "global_active_power": "global_active_power",
        "global_reactive_power": "global_reactive_power",
        "voltage": "voltage",
        "global_intensity": "global_intensity",
        "sub_metering_1": "sub_metering_1",
        "sub_metering_2": "sub_metering_2",
        "sub_metering_3": "sub_metering_3",
        "sub_metering_remainder": "sub_metering_remainder",
        "rr": "RR",
        "nbjrr1": "NBJRR1",
        "nbjrr5": "NBJRR5",
        "nbjrr10": "NBJRR10",
        "nbjbrou": "NBJBROU",
    }
    return mapping.get(lower, cleaned)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_canonical_column(col) for col in out.columns]
    return out


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        return pd.to_datetime(combined, errors="coerce", dayfirst=True)
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    raise ValueError("Input data must contain Date/Time, datetime, or date columns.")


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.replace(["?", "", " ", "nan", "NaN", "None", None], np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _first_non_null(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    return float(values.iloc[0])


def aggregate_daily_power(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)
    df["datetime"] = _parse_datetime(df)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        raise ValueError("No valid timestamps found after parsing input data.")

    for col in CORE_SENSOR_COLS + WEATHER_COLS + ["sub_metering_remainder"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = _to_numeric(df[col])

    df[CORE_SENSOR_COLS] = df[CORE_SENSOR_COLS].ffill().bfill()
    missing_core = df[CORE_SENSOR_COLS].isna().sum()
    missing_core = missing_core[missing_core > 0]
    if not missing_core.empty:
        details = ", ".join(f"{col}={int(count)}" for col, count in missing_core.items())
        raise ValueError(f"Unrecoverable missing values in core sensor columns after time fill: {details}")

    optional_cols = WEATHER_COLS + ["sub_metering_remainder"]
    df[optional_cols] = df[optional_cols].ffill().bfill().fillna(0.0)
    df["date"] = df["datetime"].dt.floor("D")

    agg: dict[str, str | callable] = {}
    for col in POWER_SUM_COLS:
        agg[col] = "sum"
    for col in POWER_MEAN_COLS:
        agg[col] = "mean"
    for col in WEATHER_COLS:
        agg[col] = _first_non_null
    if "sub_metering_remainder" in df.columns and df["sub_metering_remainder"].notna().any():
        agg["sub_metering_remainder"] = "sum"

    daily = df.groupby("date", as_index=False).agg(agg)
    if (
        "sub_metering_remainder" not in daily.columns
        or np.isclose(daily["sub_metering_remainder"].abs().sum(), 0.0)
    ):
        daily["sub_metering_remainder"] = (
            daily["global_active_power"] * 1000.0 / 60.0
            - daily[["sub_metering_1", "sub_metering_2", "sub_metering_3"]].sum(axis=1)
        )

    daily = add_time_features(daily)
    ordered = ["date"] + POWER_SUM_COLS + ["sub_metering_remainder"] + POWER_MEAN_COLS + WEATHER_COLS + TIME_FEATURES
    return daily[ordered].sort_values("date").reset_index(drop=True)


def add_time_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek.astype(float)
    out["month"] = out["date"].dt.month.astype(float)
    out["day_of_year"] = out["date"].dt.dayofyear.astype(float)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(float)
    angle = 2.0 * np.pi * out["day_of_year"] / 366.0
    out["day_of_year_sin"] = np.sin(angle)
    out["day_of_year_cos"] = np.cos(angle)
    return out


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    dates: Iterable,
    input_len: int = 90,
    output_len: int = 90,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_arr = np.asarray(features, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
    date_arr = np.asarray(pd.to_datetime(list(dates)).values, dtype="datetime64[D]")
    if len(feature_arr) != len(target_arr) or len(feature_arr) != len(date_arr):
        raise ValueError("features, target, and dates must have the same length.")

    X, y, target_dates = [], [], []
    total = input_len + output_len
    for start in range(0, len(feature_arr) - total + 1, stride):
        input_end = start + input_len
        output_end = input_end + output_len
        X.append(feature_arr[start:input_end])
        y.append(target_arr[input_end:output_end])
        target_dates.append(date_arr[input_end:output_end])

    if not X:
        return (
            np.empty((0, input_len, feature_arr.shape[1]), dtype=np.float32),
            np.empty((0, output_len), dtype=np.float32),
            np.empty((0, output_len), dtype="datetime64[D]"),
        )
    return np.stack(X), np.stack(y), np.stack(target_dates)


def discover_data_files(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir)
    candidates = {
        "train": ["train.csv"],
        "test": ["test.csv", "tes.csv"],
        "raw": [
            "household_power_consumption.txt",
            "household_power_consumption.csv",
            "individual_household_electric_power_consumption.txt",
            "individual_household_electric_power_consumption.csv",
        ],
        "weather": ["weather.csv", "climate.csv", "climat.csv"],
    }
    found: dict[str, Path] = {}
    for key, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                found[key] = path
                break
    return found


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep=";")


def load_daily_data(data_dir: str | Path) -> pd.DataFrame:
    files = discover_data_files(data_dir)
    frames: list[pd.DataFrame] = []
    if "train" in files:
        train = _read_csv_flexible(files["train"])
        train["_split"] = "train"
        frames.append(train)
    if "test" in files:
        test = _read_csv_flexible(files["test"])
        test["_split"] = "test"
        frames.append(test)
    if not frames and "raw" in files:
        raw = _read_csv_flexible(files["raw"])
        raw["_split"] = "all"
        frames.append(raw)
    if not frames:
        raise FileNotFoundError(
            f"No train.csv, test.csv/tes.csv, or household power raw file found in {Path(data_dir)}"
        )

    daily_parts = []
    for frame in frames:
        split = frame["_split"].iloc[0]
        daily = aggregate_daily_power(frame.drop(columns=["_split"]))
        daily["_split"] = split
        daily_parts.append(daily)

    daily_all = pd.concat(daily_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    daily_all = daily_all.drop_duplicates(subset=["date"], keep="last")
    if "weather" in files:
        weather = normalize_columns(_read_csv_flexible(files["weather"]))
        weather_daily = aggregate_daily_power(weather) if "global_active_power" in weather.columns else weather
        if "date" in weather_daily.columns:
            weather_daily["date"] = pd.to_datetime(weather_daily["date"], errors="coerce", dayfirst=True)
            weather_cols = [col for col in WEATHER_COLS if col in weather_daily.columns]
            daily_all = daily_all.drop(columns=[col for col in weather_cols if col in daily_all.columns], errors="ignore")
            daily_all = daily_all.merge(weather_daily[["date"] + weather_cols], on="date", how="left")
            daily_all[weather_cols] = daily_all[weather_cols].ffill().bfill().fillna(0.0)
    return daily_all.sort_values("date").reset_index(drop=True)


def _split_rows(
    daily: pd.DataFrame,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    input_len: int = 90,
    output_lens: tuple[int, ...] = (90, 365),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_col = daily.get("_split")
    n = len(daily)
    if split_col is not None and set(split_col.unique()) >= {"train", "test"}:
        train_rows = split_col.to_numpy() == "train"
        test_rows = split_col.to_numpy() == "test"
        train_indices = np.where(train_rows)[0]
        valid_rows = np.zeros(n, dtype=bool)
        min_span = input_len + max(output_lens)
        if len(train_indices) >= min_span * 2:
            valid_count = max(min_span, int(len(train_indices) * valid_ratio))
            valid_rows[train_indices[-valid_count:]] = True
            train_rows = train_rows & ~valid_rows
        return train_rows, valid_rows, test_rows

    min_span = input_len + max(output_lens)
    if n >= min_span * 3:
        train_end = max(min_span, int(n * train_ratio))
        valid_end = max(train_end + min_span, int(n * (train_ratio + valid_ratio)))
        if valid_end + min_span > n:
            valid_end = n - min_span
            train_end = max(min_span, valid_end - min_span)
        train_rows = np.arange(n) < train_end
        valid_rows = (np.arange(n) >= train_end) & (np.arange(n) < valid_end)
        test_rows = np.arange(n) >= valid_end
    else:
        cutoff = max(1, int(n * train_ratio))
        train_rows = np.arange(n) < cutoff
        test_rows = ~train_rows
        train_indices = np.where(train_rows)[0]
        valid_rows = np.zeros(n, dtype=bool)
        if len(train_indices) > 10:
            valid_count = max(1, int(len(train_indices) * 0.1))
            valid_rows[train_indices[-valid_count:]] = True
            train_rows = train_rows & ~valid_rows
    return train_rows, valid_rows, test_rows


def _sample_mask_from_rows(target_dates: np.ndarray, dates: np.ndarray, row_mask: np.ndarray) -> np.ndarray:
    allowed = set(np.asarray(dates[row_mask], dtype="datetime64[D]").tolist())
    return np.array([all(day in allowed for day in sample_dates.tolist()) for sample_dates in target_dates], dtype=bool)


def preprocess_project_data(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    input_len: int = 90,
    output_lens: tuple[int, ...] = (90, 365),
    stride: int = 1,
) -> PreprocessResult:
    out_dir = ensure_dir(output_dir)
    daily = load_daily_data(data_dir)
    daily_path = out_dir / "daily_power.csv"
    daily.to_csv(daily_path, index=False)

    train_rows, valid_rows, test_rows = _split_rows(daily, input_len=input_len, output_lens=output_lens)
    feature_cols = [
        col
        for col in daily.columns
        if col not in {"date", "_split"} and pd.api.types.is_numeric_dtype(daily[col])
    ]
    if TARGET_COL not in feature_cols:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    features = daily[feature_cols].astype(float).to_numpy()
    target = daily[TARGET_COL].astype(float).to_numpy().reshape(-1, 1)
    feature_scaler = StandardScaler().fit(features[train_rows])
    target_scaler = StandardScaler().fit(target[train_rows])
    scaled_features = feature_scaler.transform(features).astype(np.float32)
    scaled_target = target_scaler.transform(target).reshape(-1).astype(np.float32)

    scaler_bundle = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_cols,
        "target": TARGET_COL,
        "input_len": input_len,
        "output_lens": output_lens,
    }
    scaler_path = out_dir / "scaler.pkl"
    joblib.dump(scaler_bundle, scaler_path)

    generated: list[Path] = [daily_path, scaler_path]
    dates = pd.to_datetime(daily["date"])
    date_arr = np.asarray(dates.values, dtype="datetime64[D]")
    for output_len in output_lens:
        X, y, target_dates = build_windows(scaled_features, scaled_target, dates, input_len, output_len, stride)
        masks = {
            "train": _sample_mask_from_rows(target_dates, date_arr, train_rows),
            "valid": _sample_mask_from_rows(target_dates, date_arr, valid_rows),
            "test": _sample_mask_from_rows(target_dates, date_arr, test_rows),
        }
        for split, mask in masks.items():
            path = out_dir / f"{split}_{output_len}.npz"
            np.savez(
                path,
                X=X[mask].astype(np.float32),
                y=y[mask].astype(np.float32),
                target_dates=target_dates[mask],
                feature_names=np.asarray(feature_cols),
            )
            generated.append(path)

    return PreprocessResult(daily_path, scaler_path, generated, feature_cols)


def generate_smoke_raw_data(path: str | Path, days: int = 520) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    timestamps = pd.date_range("2007-01-01", periods=days * 24, freq="h")
    t = np.arange(len(timestamps))
    daily_cycle = np.sin(2 * np.pi * t / 24)
    yearly = np.sin(2 * np.pi * t / (24 * 365))
    active = 1.2 + 0.2 * daily_cycle + 0.4 * yearly + 0.01 * (t % 17)
    frame = pd.DataFrame(
        {
            "Date": timestamps.strftime("%d/%m/%Y"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Global_active_power": active,
            "Global_reactive_power": 0.2 + 0.02 * daily_cycle,
            "Voltage": 240 + 2 * yearly,
            "Global_intensity": 5 + daily_cycle,
            "Sub_metering_1": 1 + (t % 5),
            "Sub_metering_2": 2 + (t % 7),
            "Sub_metering_3": 3 + (t % 11),
            "RR": 10 + (t // 24) % 4,
            "NBJRR1": 1,
            "NBJRR5": 0,
            "NBJRR10": 0,
            "NBJBROU": 0,
        }
    )
    frame.to_csv(out, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess household power data into daily forecasting windows.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--output_lens", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--generate_smoke_data", action="store_true")
    parser.add_argument("--smoke_days", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_smoke_data:
        generate_smoke_raw_data(Path(args.data_dir) / "household_power_consumption.csv", days=args.smoke_days)
    result = preprocess_project_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_len=args.input_len,
        output_lens=tuple(args.output_lens),
        stride=args.stride,
    )
    print(f"Saved daily data: {result.daily_path}")
    print(f"Saved scaler: {result.scaler_path}")
    for path in result.generated_files:
        print(path)


if __name__ == "__main__":
    main()
