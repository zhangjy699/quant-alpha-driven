"""Compact full factor-pool backtest outputs into one comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_DIR = PROJECT_ROOT / "outputs" / "backtests" / "full-factor-pool"
BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "backtests" / "compact-factor-pool"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact Alphalens full factor-pool backtest outputs."
    )
    parser.add_argument("--input-dir", default=str(BASE_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(BASE_OUTPUT_DIR))
    parser.add_argument("--period", default="1D")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    records = compact_backtest_outputs(
        input_dir=input_dir,
        period=args.period,
    )
    path = write_overall_csv_from_records(records, input_dir=input_dir, output_dir=output_dir)
    print(path)


def write_overall_csv(
    *,
    input_dir: Path,
    output_dir: Path = BASE_OUTPUT_DIR,
    period: str = "1D",
) -> Path:
    records = compact_backtest_outputs(input_dir=input_dir, period=period)
    return write_overall_csv_from_records(records, input_dir=input_dir, output_dir=output_dir)


def write_overall_csv_from_records(
    records: list[dict[str, Any]],
    *,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    if not records:
        raise ValueError(f"No compactable factor reports found under {input_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "overall.csv"
    new_table = pd.DataFrame(records)
    table = merge_overall_csv(path, new_table)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def merge_overall_csv(path: Path, new_table: pd.DataFrame) -> pd.DataFrame:
    """Append compact rows; replace existing rows for the same factor_id."""

    if not path.exists():
        return new_table
    existing = pd.read_csv(path)
    if existing.empty:
        return new_table
    if "factor_id" not in existing.columns or "factor_id" not in new_table.columns:
        return pd.concat([existing, new_table], ignore_index=True)
    replaced_ids = set(new_table["factor_id"].dropna().tolist())
    kept = existing[~existing["factor_id"].isin(replaced_ids)]
    return pd.concat([kept, new_table], ignore_index=True)


def compact_backtest_outputs(
    *,
    input_dir: Path = BASE_INPUT_DIR,
    period: str = "1D",
) -> list[dict[str, Any]]:
    batches = discover_batches(input_dir=input_dir)
    records: list[dict[str, Any]] = []
    for batch_dir in batches:
        summary = read_factor_summary(batch_dir)
        for _, factor in summary.iterrows():
            try:
                records.append(compact_factor_row(batch_dir, factor, period=period))
            except FileNotFoundError as exc:
                print(f"skip {batch_dir.name} factor {factor.get('factor_id')}: {exc}")
    return records


def read_factor_summary(batch_dir: Path) -> pd.DataFrame:
    path = batch_dir / "factor_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing batch factor_summary.csv: {path}")
    return pd.read_csv(path)


def compact_factor_row(
    batch_dir: Path,
    factor: pd.Series,
    *,
    period: str,
) -> dict[str, Any]:
    factor_dir = batch_dir / Path(str(factor["report_path"])).parent
    quantile_excess = read_quantile_returns(factor_dir, period=period, kind="excess")
    top_quantile = int(max(quantile_excess.columns))
    ic = read_period_series(factor_dir / "daily_ic_by_period.csv", period=period)
    top_excess = quantile_excess[top_quantile].dropna().sort_index()
    factor_data = read_factor_data(factor_dir / "alphalens_factor_data.csv")
    turnover = compute_top_quantile_turnover(factor_data, top_quantile=top_quantile)

    record: dict[str, Any] = {
        "factor_id": factor.get("factor_id"),
        "total_annualized_excess_return": annualized_return(top_excess),
        "ic": safe_mean(ic),
        "icir": information_coefficient_ratio(ic),
        "ir": annualized_information_ratio(top_excess),
        "max_drawdown": max_drawdown(top_excess),
        "turnover": safe_mean(turnover),
    }
    record.update(annual_return_columns(top_excess))
    return record


def discover_batches(*, input_dir: Path) -> list[Path]:
    if (input_dir / "factors").is_dir() and (input_dir / "factor_summary.csv").exists():
        return [input_dir]
    batches = sorted(
        [
            path
            for path in input_dir.glob("full-factor-pool-*")
            if (path / "factors").is_dir() and (path / "factor_summary.csv").exists()
        ],
        key=lambda path: path.name,
    )
    return batches[-1:] if batches else []


def read_quantile_returns(factor_dir: Path, *, period: str, kind: str) -> pd.DataFrame:
    path = factor_dir / f"quantile_{kind}_returns_{period}.csv"
    frame = read_date_indexed_csv(path)
    frame.columns = [int(column) for column in frame.columns]
    return frame.sort_index().sort_index(axis=1)


def read_period_series(path: Path, *, period: str) -> pd.Series:
    frame = read_date_indexed_csv(path)
    if period not in frame:
        raise FileNotFoundError(f"{path} does not contain period column {period!r}.")
    return frame[period].dropna().sort_index()


def read_factor_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["date", "asset", "factor_quantile"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["factor_quantile"] = pd.to_numeric(frame["factor_quantile"], errors="coerce")
    return frame.dropna(subset=["date", "asset", "factor_quantile"])


def read_date_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame.set_index("date").sort_index()


def compute_top_quantile_turnover(
    factor_data: pd.DataFrame,
    *,
    top_quantile: int,
) -> pd.Series:
    top = factor_data.loc[factor_data["factor_quantile"] == top_quantile, ["date", "asset"]]
    holdings = top.groupby("date", sort=True)["asset"].apply(lambda values: set(values))
    values: dict[pd.Timestamp, float] = {}
    previous: set[str] | None = None
    for date, current in holdings.items():
        if previous:
            values[date] = 1.0 - (len(previous & current) / len(previous))
        previous = current
    return pd.Series(values, name="turnover").sort_index()


def annual_return_columns(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {}
    yearly = returns.groupby(returns.index.year).apply(annualized_return)
    return {str(int(year)): float(value) for year, value in yearly.items()}


def annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    total_return = (1.0 + clean).prod() - 1.0
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / len(clean)) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def annualized_information_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    std = clean.std(ddof=1)
    if clean.empty or pd.isna(std) or std == 0:
        return float("nan")
    return float(clean.mean() / std * (TRADING_DAYS_PER_YEAR**0.5))


def information_coefficient_ratio(ic: pd.Series) -> float:
    clean = ic.dropna()
    std = clean.std(ddof=1)
    if clean.empty or pd.isna(std) or std == 0:
        return float("nan")
    return float(clean.mean() / std)


def safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


if __name__ == "__main__":
    main()
