"""Prepare company All-A raw data in Qlib-compatible local artifacts.

The company DB server may fail on large date ranges, so this script fetches
daily data in date chunks, caches each chunk as parquet, then writes the same
raw files consumed by `prepare_hf_qlib_csi300.py --offline`.

Fill the two company DB call stubs with the pre-adjusted OHLC interface and the
daily quote interface. Keep the rest of the script as a deterministic adapter
from company DataFrame output to Qlib-style raw files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch company All-A data by chunk and write Qlib-style raw files."
    )
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-01")
    parser.add_argument("--raw-dir", default="data/raw/company_all_a")
    parser.add_argument("--chunk", choices=["monthly", "quarterly"], default="quarterly")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch chunks even when local parquet cache files already exist.",
    )
    parser.add_argument(
        "--exclude-st",
        action="store_true",
        help="Drop rows flagged by an optional is_st column before caching.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    cache_dir = raw_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    for start_date, end_date in date_chunks(args.start_date, args.end_date, args.chunk):
        cache_path = cache_dir / f"daily_{start_date}_{end_date}.parquet"
        if not cache_path.exists() or args.refresh:
            frame = fetch_company_daily_panel_from_interfaces(
                start_date=start_date,
                end_date=end_date,
            )
            frame = standardize_company_columns(frame)
            frame = filter_company_rows(frame, exclude_st=args.exclude_st)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
        chunk_paths.append(cache_path)

    if not chunk_paths:
        raise ValueError("No date chunks were generated.")

    raw = pd.concat([pd.read_parquet(path) for path in chunk_paths], ignore_index=True)
    raw = _dedupe_and_sort(raw)
    if raw.empty:
        raise ValueError("Company All-A data is empty after filtering.")

    hdf_path = raw_dir / "daily_pv.h5"
    universe_zip_path = raw_dir / "cn_data.zip"
    write_qlib_hdf(raw, hdf_path)
    write_all_a_universe(raw, universe_zip_path)

    manifest = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "chunk": args.chunk,
        "exclude_st": args.exclude_st,
        "price_source": "company_preadjusted_ohlc_interface",
        "auxiliary_source": "company_daily_quote_interface",
        "cache_dir": str(cache_dir),
        "chunks": [str(path) for path in chunk_paths],
        "daily_pv_hdf": str(hdf_path),
        "universe_zip": str(universe_zip_path),
        "universe_file": "cn_data/instruments/all_a.txt",
        "rows": int(len(raw)),
        "assets": int(raw["asset"].nunique()),
        "dates": int(raw["date"].nunique()),
        "actual_start": str(raw["date"].min().date()),
        "actual_end": str(raw["date"].max().date()),
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def fetch_company_daily_panel_from_interfaces(
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return one merged All-A chunk from pre-adjusted OHLC and quote interfaces."""

    adjusted = fetch_company_preadjusted_panel(start_date=start_date, end_date=end_date)
    quote = fetch_company_quote_panel(start_date=start_date, end_date=end_date)
    adjusted = normalize_key_columns(adjusted)
    quote = normalize_key_columns(quote)
    return adjusted.merge(quote, on=["date", "asset"], how="left", suffixes=("", "_quote"))


def fetch_company_preadjusted_panel(*, start_date: str, end_date: str) -> pd.DataFrame:
    """Return pre-adjusted OHLC for one date chunk.

    Fill this function with the company pre-adjusted price DB call. The returned
    DataFrame must have one row per stock per trading day.

    Required columns, using either normalized or company names:
      date/trade_date/datetime
      asset/stock_code/ticker/symbol/code
      open/high/low/close, or adj_open/adj_high/adj_low/adj_close
    """

    raise NotImplementedError(
        "Fill fetch_company_preadjusted_panel with the company pre-adjusted "
        f"OHLC call for {start_date} to {end_date}."
    )


def fetch_company_quote_panel(*, start_date: str, end_date: str) -> pd.DataFrame:
    """Return quote, volume, and trading-state fields for one date chunk.

    Fill this function with the company quote DB call. The returned DataFrame must have
    one row per stock per trading day.

    Required columns, using either normalized or company names:
      date/trade_date/datetime
      asset/stock_code/ticker/symbol/code
      volume

    Recommended optional columns:
      amount, trade_count, change, pct_change, adj_factor, turnover,
      recent_trade_date, trading_status, suspended_days, delisting_period,
      is_trading, is_suspended, security_type, exchange
    """

    raise NotImplementedError(
        "Fill fetch_company_quote_panel with the company quote call for "
        f"{start_date} to {end_date}."
    )


def normalize_key_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Company DB call must return a pandas DataFrame.")
    frame = raw.copy()
    rename_map = {
        "trade_date": "date",
        "datetime": "date",
        "ticker": "asset",
        "symbol": "asset",
        "stock_code": "asset",
        "code": "asset",
    }
    frame = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame})
    missing = [column for column in ("date", "asset") if column not in frame]
    if missing:
        raise ValueError(f"Company data is missing key columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["asset"] = frame["asset"].map(normalize_asset_code)
    return frame


def standardize_company_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Company DB call must return a pandas DataFrame.")

    frame = normalize_key_columns(raw)

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

    return frame


def filter_company_rows(raw: pd.DataFrame, *, exclude_st: bool) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.loc[frame["date"].notna()]
    frame = frame.loc[frame["asset"].astype(str).str.len() > 0]

    if "security_type" in frame:
        allowed_security_types = {"stock", "a_share", "ashare", "a-stock", "a股"}
        security_type = frame["security_type"].astype(str).str.strip().str.lower()
        frame = frame.loc[security_type.isin(allowed_security_types)]

    if "is_trading" in frame:
        frame = frame.loc[truthy(frame["is_trading"])]

    if "is_suspended" in frame:
        frame = frame.loc[~truthy(frame["is_suspended"])]

    if "trading_status" in frame:
        frame = frame.loc[normal_trading_status(frame["trading_status"])]

    if "delisting_period" in frame:
        frame = frame.loc[~truthy(frame["delisting_period"])]

    if "suspended_days" in frame:
        suspended_days = pd.to_numeric(frame["suspended_days"], errors="coerce").fillna(0)
        frame = frame.loc[suspended_days == 0]

    if exclude_st and "is_st" in frame:
        frame = frame.loc[~truthy(frame["is_st"])]

    if "list_date" in frame:
        list_dates = pd.to_datetime(frame["list_date"], errors="coerce")
        frame = frame.loc[list_dates.isna() | (frame["date"] >= list_dates)]

    if "delist_date" in frame:
        delist_dates = pd.to_datetime(frame["delist_date"], errors="coerce")
        frame = frame.loc[delist_dates.isna() | (frame["date"] <= delist_dates)]

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame.loc[frame["volume"] > 0]
    return frame


def write_qlib_hdf(raw: pd.DataFrame, path: Path) -> None:
    qlib = raw.copy()
    qlib["$open"] = qlib["open"]
    qlib["$close"] = qlib["close"]
    qlib["$high"] = qlib["high"]
    qlib["$low"] = qlib["low"]
    qlib["$volume"] = qlib["volume"]
    qlib["$factor"] = qlib["adj_factor"] if "adj_factor" in qlib else 1.0

    qlib = qlib.rename(columns={"date": "datetime", "asset": "instrument"})
    qlib = qlib.set_index(["datetime", "instrument"])
    qlib = qlib[["$open", "$close", "$high", "$low", "$volume", "$factor"]]
    qlib = qlib.sort_index().astype("float32")

    path.parent.mkdir(parents=True, exist_ok=True)
    qlib.to_hdf(path, key="data", mode="w")


def write_all_a_universe(raw: pd.DataFrame, zip_path: Path) -> None:
    ranges = (
        raw.groupby("asset", sort=True)["date"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values("asset")
    )
    lines = [
        f"{row.asset}\t{row['min'].date()}\t{row['max'].date()}"
        for _, row in ranges.iterrows()
    ]
    text = "\n".join(lines) + "\n"

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("cn_data/instruments/all_a.txt", text)


def date_chunks(start_date: str, end_date: str, chunk: str):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError(f"start_date {start.date()} is after end_date {end.date()}.")

    freq = "MS" if chunk == "monthly" else "QS"
    starts = list(pd.date_range(start=start, end=end, freq=freq))
    if not starts or starts[0] != start:
        starts = [start] + starts

    for current, next_start in zip(starts, [*starts[1:], end + pd.Timedelta(days=1)]):
        chunk_end = min(next_start - pd.Timedelta(days=1), end)
        if current <= chunk_end:
            yield str(current.date()), str(chunk_end.date())


def normalize_asset_code(value: object) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""
    if raw.startswith(("SH", "SZ", "BJ")):
        return raw.replace(".", "")
    if "." in raw:
        code, exchange = raw.split(".", 1)
        exchange = exchange[:2]
        if exchange in {"SH", "SZ", "BJ"}:
            return f"{exchange}{code.zfill(6)}"
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) == 6:
        if digits.startswith(("6", "9")):
            return f"SH{digits}"
        if digits.startswith(("0", "2", "3")):
            return f"SZ{digits}"
        if digits.startswith(("4", "8")):
            return f"BJ{digits}"
    return raw


def truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "是", "退市整理", "停牌"})


def normal_trading_status(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"交易", "正常交易", "normal", "trade", "trading", "1"})


def _dedupe_and_sort(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "asset"])
    frame = frame.drop_duplicates(subset=["date", "asset"], keep="last")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
