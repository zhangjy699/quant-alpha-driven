"""Prepare company All-A daily OHLCV data for CogAlpha evaluation.

This script is intentionally a thin adapter. Fill `_fetch_company_daily_panel`
with the company database/API call, then keep the remaining normalization,
split, forward-return construction, and artifact writing path unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cogalpha.config import BaselineExperimentConfig
from cogalpha.data import build_baseline_market_data, normalize_ohlcv_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare company All-A data for CogAlpha formal MVP evaluation."
    )
    parser.add_argument("--output-dir", default="data/processed/company_all_a")
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional local CSV/Parquet export for debugging. If omitted, the "
            "company API fetch stub is used."
        ),
    )
    parser.add_argument(
        "--exclude-st",
        action="store_true",
        help="Drop rows flagged by an optional is_st column.",
    )
    args = parser.parse_args()

    config = BaselineExperimentConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw_company_panel(args.input_file, config)
    raw = _standardize_company_columns(raw)
    raw = _filter_company_rows(raw, exclude_st=args.exclude_st)
    panel = normalize_ohlcv_panel(raw)
    dataset = build_baseline_market_data(panel, config)

    panel_path = output_dir / "ohlcv_panel.parquet"
    panel.reset_index().to_parquet(panel_path, index=False)

    split_paths: dict[str, dict[str, str]] = {}
    for split_name in ("train", "valid", "test"):
        split = dataset.split(split_name)
        split_dates = split.ohlcv_panel.index.get_level_values("date")
        non_null_returns = int(split.forward_returns.notna().sum().sum())
        ohlcv_path = output_dir / f"{split_name}_ohlcv.parquet"
        returns_path = output_dir / f"{split_name}_forward_returns.parquet"
        split.ohlcv_panel.reset_index().to_parquet(ohlcv_path, index=False)
        split.forward_returns.to_parquet(returns_path)
        split_paths[split_name] = {
            "ohlcv": str(ohlcv_path),
            "forward_returns": str(returns_path),
            "rows": str(len(split.ohlcv_panel)),
            "dates": str(split.ohlcv_panel.index.get_level_values("date").nunique()),
            "assets": str(split.ohlcv_panel.index.get_level_values("asset").nunique()),
            "actual_start": str(split_dates.min().date()),
            "actual_end": str(split_dates.max().date()),
            "non_null_forward_returns": str(non_null_returns),
        }

    dates = panel.index.get_level_values("date")
    assets = panel.index.get_level_values("asset")
    data_version_payload = {
        "source": "company_all_a",
        "input_file": args.input_file,
        "horizon_days": dataset.horizon_days,
        "return_price_column": config.return_price_column,
        "trade_delay_days": config.trade_delay_days,
        "exclude_st": args.exclude_st,
        "split": config.split.model_dump(mode="json"),
        "fitness_gate": config.fitness_gate.model_dump(mode="json"),
    }
    data_version = hashlib.sha256(
        json.dumps(data_version_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "source": "company_all_a",
        "data_version": data_version,
        "data_version_payload": data_version_payload,
        "dataset": "ALL_A",
        "horizon_days": dataset.horizon_days,
        "return_price_column": config.return_price_column,
        "trade_delay_days": config.trade_delay_days,
        "input_contract": {
            "frequency": "daily",
            "input_columns": ["open", "high", "low", "close", "volume"],
            "target": "10-day forward return with buying and selling at open price",
            "timing_contract": (
                "AlphaCandidate observes date t daily OHLCV, enters at the next open, "
                "and exits after 10 trading opens."
            ),
        },
        "full_panel": {
            "path": str(panel_path),
            "rows": len(panel),
            "dates": dates.nunique(),
            "assets": assets.nunique(),
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        "splits": split_paths,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _load_raw_company_panel(
    input_file: str | None,
    config: BaselineExperimentConfig,
) -> pd.DataFrame:
    if input_file is not None:
        return _read_local_input(input_file)
    return _fetch_company_daily_panel(
        start_date=str(config.split.train_start),
        end_date=str(config.split.test_end),
    )


def _read_local_input(input_file: str) -> pd.DataFrame:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    raise ValueError("Unsupported input file format. Expected CSV or Parquet.")


def _fetch_company_daily_panel(*, start_date: str, end_date: str) -> pd.DataFrame:
    """Return raw company All-A daily data.

    Fill this function with the company database/API call. The returned frame
    should contain one row per stock per trading day. Recommended raw columns:

    Required price-volume columns, either adjusted or raw:
      trade_date, stock_code, open, high, low, close, volume

    Preferred adjusted columns, if available:
      adj_open, adj_high, adj_low, adj_close

    Optional filter/debug columns:
      is_trading, is_suspended, is_st, list_date, delist_date, exchange,
      pre_close, amount, turnover, adj_factor, limit_up, limit_down
    """

    raise NotImplementedError(
        "Fill _fetch_company_daily_panel with the company API call, or pass "
        "--input-file with a local CSV/Parquet export."
    )


def _standardize_company_columns(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    rename_map = {
        "trade_date": "date",
        "datetime": "date",
        "ticker": "asset",
        "symbol": "asset",
        "stock_code": "asset",
        "code": "asset",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame})

    adjusted_columns = {
        "adj_open": "open",
        "adj_high": "high",
        "adj_low": "low",
        "adj_close": "close",
    }
    if all(column in frame for column in adjusted_columns):
        frame = frame.rename(columns=adjusted_columns)

    required = ["date", "asset", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Company data is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["asset"] = frame["asset"].map(_normalize_asset_code)
    return frame


def _filter_company_rows(raw: pd.DataFrame, *, exclude_st: bool) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.loc[frame["date"].notna()]
    frame = frame.loc[frame["asset"].astype(str).str.len() > 0]

    if "list_date" in frame:
        list_dates = pd.to_datetime(frame["list_date"], errors="coerce")
        frame = frame.loc[list_dates.isna() | (frame["date"] >= list_dates)]
    if "delist_date" in frame:
        delist_dates = pd.to_datetime(frame["delist_date"], errors="coerce")
        frame = frame.loc[delist_dates.isna() | (frame["date"] <= delist_dates)]
    if "is_trading" in frame:
        frame = frame.loc[_truthy(frame["is_trading"])]
    if "is_suspended" in frame:
        frame = frame.loc[~_truthy(frame["is_suspended"])]
    if exclude_st and "is_st" in frame:
        frame = frame.loc[~_truthy(frame["is_st"])]

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame.loc[frame["volume"] > 0]
    return frame


def _truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _normalize_asset_code(value: object) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    if raw.startswith(("SH", "SZ", "BJ")) and len(raw) >= 8:
        return raw.replace(".", "")
    if "." in raw:
        code, exchange = raw.split(".", 1)
        exchange = exchange[:2]
        if exchange in {"SH", "SZ", "BJ"}:
            return f"{exchange}{code.zfill(6)}"
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 6:
        if digits.startswith(("6", "9")):
            return f"SH{digits}"
        if digits.startswith(("0", "2", "3")):
            return f"SZ{digits}"
        if digits.startswith(("4", "8")):
            return f"BJ{digits}"
    return raw


if __name__ == "__main__":
    main()
