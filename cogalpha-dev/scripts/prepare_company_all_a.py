"""Fetch company All-A data by chunks and write Qlib-style raw files.


Only fill the two fetch functions below. The rest of the script:
 1. fetches data in monthly/quarterly chunks,
 2. caches each cleaned chunk as parquet,
 3. writes `daily_pv.h5` and `cn_data.zip` for the existing offline prepare step.
"""


from __future__ import annotations


import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from tq_data_client import DBServer

import pandas as pd



def fetch_company_preadjusted_panel(*, start_date: str, end_date: str) -> pd.DataFrame:
    """Return pre-adjusted OHLC from the company DB.

    Required company columns:
        trading_date, wind_code, open_price, high_price, low_price, close_price
    """
    client = DBServer(host='192.168.1.27',port=30801)
    params = {
        'start_date': {start_date},
        'end_date': {end_date},
    }
    url = f'{client.url_root}/tqmain/equity_adj_daily_f'
    result = client.get_data(params, url=url)
    return result


def fetch_company_quote_panel(*, start_date: str, end_date: str) -> pd.DataFrame:
    """Return quote/volume/status fields from the company DB.


    Required company columns:
        trading_date, wind_code, volume


    Useful optional columns:
        amount, deal_number, change, pct_change, swing, vwap, adj_factor, turn,
        last_tradeday, last_tradeday_mkt, is_st, is_delisting, trade_status,
        susp_days, susp_reason, max_up_or_down, maxup, maxdown
    """

    client = DBServer(host='192.168.1.27',port=30801)
    params = {
        'start_date': {start_date},
        'end_date': {end_date},
    }
    url = f'{client.url_root}/tqmain/equity_daily'
    result = client.get_data(params, url=url)
    return result



def main() -> None:
   parser = argparse.ArgumentParser(
       description="Build company All-A Qlib-style raw files from DB chunks."
   )
   parser.add_argument("--start-date", default="2018-01-01")
   parser.add_argument("--end-date", default="2026-06-01")
   parser.add_argument("--raw-dir", default="data/raw/company_all_a")
   parser.add_argument("--chunk", choices=["monthly", "quarterly"], default="quarterly")
   parser.add_argument("--refresh", action="store_true")
   parser.add_argument("--exclude-st", action="store_true")
   args = parser.parse_args()


   raw_dir = Path(args.raw_dir)
   cache_dir = raw_dir / "cache"
   cache_dir.mkdir(parents=True, exist_ok=True)


   cache_paths: list[Path] = []
   for start_date, end_date in date_chunks(args.start_date, args.end_date, args.chunk):
       cache_path = cache_dir / f"daily_{start_date}_{end_date}.parquet"
       if args.refresh or not cache_path.exists():
           chunk = fetch_and_clean_chunk(
               start_date=start_date,
               end_date=end_date,
               exclude_st=args.exclude_st,
           )
           chunk.to_parquet(cache_path, index=False)
       cache_paths.append(cache_path)


   data = pd.concat([pd.read_parquet(path) for path in cache_paths], ignore_index=True)
   data = data.drop_duplicates(["date", "asset"], keep="last")
   data = data.sort_values(["date", "asset"]).reset_index(drop=True)
   if data.empty:
       raise ValueError("No usable company All-A rows after cleaning.")


   hdf_path = raw_dir / "daily_pv.h5"
   universe_path = raw_dir / "cn_data.zip"
   write_qlib_hdf(data, hdf_path)
   write_universe(data, universe_path)
   write_manifest(
       raw_dir=raw_dir,
       cache_paths=cache_paths,
       data=data,
       args=args,
       hdf_path=hdf_path,
       universe_path=universe_path,
   )




def fetch_and_clean_chunk(
   *,
   start_date: str,
   end_date: str,
   exclude_st: bool,
) -> pd.DataFrame:
   adjusted = fetch_company_preadjusted_panel(start_date=start_date, end_date=end_date)
   quote = fetch_company_quote_panel(start_date=start_date, end_date=end_date)


   adjusted = rename_company_columns(adjusted)
   quote = rename_company_columns(quote)
   quote = quote.drop(columns=['open', 'high', 'low', 'close'], errors='ignore')
   data = adjusted.merge(quote, on=["date", "asset"], how="left")
   return clean_chunk(data, exclude_st=exclude_st)




def rename_company_columns(frame: pd.DataFrame) -> pd.DataFrame:
   return frame.rename(
       columns={
           "trading_date": "date",
           "wind_code": "asset",
           "open_price": "open",
           "high_price": "high",
           "low_price": "low",
           "close_price": "close",
           "deal_number": "trade_count",
           "turn": "turnover",
           "last_tradeday": "recent_trade_date",
           "last_tradeday_mkt": "recent_market_trade_date",
           "is_delisting": "delisting_period",
           "trade_status": "trading_status",
           "susp_days": "suspended_days",
           "max_up_or_down": "limit_status",
       }
   )




def clean_chunk(data: pd.DataFrame, *, exclude_st: bool) -> pd.DataFrame:
   required = ["date", "asset", "open", "high", "low", "close", "volume"]
   missing = [column for column in required if column not in data]
   if missing:
       raise ValueError(f"Missing required company columns after merge: {missing}")


   frame = data.copy()
   frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
   frame["asset"] = frame["asset"].map(normalize_asset)
   frame = frame.loc[frame["date"].notna()]
   frame = frame.loc[frame["asset"].astype(str).str.len() > 0]


   if "trading_status" in frame:
       frame = frame.loc[frame["trading_status"].astype(str).str.strip() == "交易"]
   if "suspended_days" in frame:
       suspended_days = pd.to_numeric(frame["suspended_days"], errors="coerce").fillna(0)
       frame = frame.loc[suspended_days == 0]
   if "delisting_period" in frame:
       frame = frame.loc[~truthy(frame["delisting_period"])]
   if exclude_st and "is_st" in frame:
       frame = frame.loc[~truthy(frame["is_st"])]


   for column in ("open", "high", "low", "close", "volume"):
       frame[column] = pd.to_numeric(frame[column], errors="coerce")
   frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
   return frame.loc[frame["volume"] > 0].reset_index(drop=True)




def write_qlib_hdf(data: pd.DataFrame, path: Path) -> None:
   qlib = pd.DataFrame(
       {
           "$open": data["open"],
           "$close": data["close"],
           "$high": data["high"],
           "$low": data["low"],
           "$volume": data["volume"],
           "$factor": data["adj_factor"] if "adj_factor" in data else 1.0,
           "datetime": data["date"],
           "instrument": data["asset"],
       }
   )
   qlib = qlib.set_index(["datetime", "instrument"]).sort_index()
   qlib = qlib[["$open", "$close", "$high", "$low", "$volume", "$factor"]]
   path.parent.mkdir(parents=True, exist_ok=True)
   qlib.astype("float32").to_hdf(path, key="data", mode="w")




def write_universe(data: pd.DataFrame, path: Path) -> None:
   ranges = data.groupby("asset", sort=True)["date"].agg(["min", "max"]).reset_index()
   lines = [
       f"{row.asset}\t{row['min'].date()}\t{row['max'].date()}"
       for _, row in ranges.iterrows()
   ]
   path.parent.mkdir(parents=True, exist_ok=True)
   with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
       archive.writestr("cn_data/instruments/all_a.txt", "\n".join(lines) + "\n")




def write_manifest(
   *,
   raw_dir: Path,
   cache_paths: list[Path],
   data: pd.DataFrame,
   args: argparse.Namespace,
   hdf_path: Path,
   universe_path: Path,
) -> None:
   manifest = {
       "start_date": args.start_date,
       "end_date": args.end_date,
       "chunk": args.chunk,
       "exclude_st": args.exclude_st,
       "cache_paths": [str(path) for path in cache_paths],
       "daily_pv_hdf": str(hdf_path),
       "universe_zip": str(universe_path),
       "universe_file": "cn_data/instruments/all_a.txt",
       "rows": int(len(data)),
       "assets": int(data["asset"].nunique()),
       "dates": int(data["date"].nunique()),
       "actual_start": str(data["date"].min().date()),
       "actual_end": str(data["date"].max().date()),
   }
   (raw_dir / "manifest.json").write_text(
       json.dumps(manifest, indent=2, sort_keys=True),
       encoding="utf-8",
   )
   print(json.dumps(manifest, indent=2, sort_keys=True))




def date_chunks(start_date: str, end_date: str, chunk: str):
   start = pd.Timestamp(start_date)
   end = pd.Timestamp(end_date)
   if start > end:
       raise ValueError(f"start-date {start.date()} is after end-date {end.date()}.")


   freq = "MS" if chunk == "monthly" else "QS"
   starts = list(pd.date_range(start=start, end=end, freq=freq))
   if not starts or starts[0] != start:
       starts = [start] + starts


   for current, next_start in zip(starts, [*starts[1:], end + pd.Timedelta(days=1)]):
       current_end = min(next_start - pd.Timedelta(days=1), end)
       if current <= current_end:
           yield str(current.date()), str(current_end.date())




def normalize_asset(value: object) -> str:
   raw = str(value).strip().upper()
   if "." in raw:
       code, exchange = raw.split(".", 1)
       return f"{exchange[:2]}{code.zfill(6)}"
   return raw.replace(".", "")




def truthy(values: pd.Series) -> pd.Series:
   if values.dtype == bool:
       return values.fillna(False)
   text = values.astype(str).str.strip().str.lower()
   return text.isin({"1", "true", "t", "yes", "y", "是", "退市整理", "停牌"})




if __name__ == "__main__":
   main()



