"""Cross-sectional factor backtest engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogalpha.config import BaselineExperimentConfig
from cogalpha.data import compute_forward_returns, normalize_ohlcv_panel, slice_ohlcv_panel
from cogalpha.execution import execute_alpha_candidate
from cogalpha.fitness import compute_predictive_metrics
from factor_backtest.loader import FactorBacktestInput


@dataclass(frozen=True)
class FactorBacktestResult:
    """Paths and summary for one independent factor backtest."""

    output_dir: Path
    report_path: Path
    audit_path: Path
    counts: dict[str, int]


def run_factor_backtest(
    *,
    factor_input: FactorBacktestInput,
    data_dir: str | Path,
    output_root: str | Path = "outputs/backtests",
    start_date: str | None = None,
    end_date: str | None = None,
    quantiles: int = 5,
    cost_bps: float = 10.0,
    neutralization_data: str | Path | None = None,
) -> FactorBacktestResult:
    """Run a compact cross-sectional factor research backtest."""

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    data_path = Path(data_dir)
    metadata = _read_metadata(data_path)
    config = BaselineExperimentConfig()
    ohlcv_panel = normalize_ohlcv_panel(pd.read_parquet(data_path / "ohlcv_panel.parquet"))
    if start_date is not None or end_date is not None:
        dates = ohlcv_panel.index.get_level_values("date")
        start = start_date or str(dates.min().date())
        end = end_date or str(dates.max().date())
        ohlcv_panel = slice_ohlcv_panel(ohlcv_panel, start=start, end=end)

    forward_returns = compute_forward_returns(
        ohlcv_panel,
        horizon_days=config.horizon_days,
        price_column=config.return_price_column,
        trade_delay_days=config.trade_delay_days,
    )
    raw_factor_values = execute_alpha_candidate(factor_input.candidate, ohlcv_panel)
    factor_values = raw_factor_values
    neutralization = {"status": "skipped"}
    if neutralization_data is not None:
        factor_values = neutralize_factor_values(
            factor_values,
            neutralization_data=neutralization_data,
        )
        neutralization = {"status": "applied", "file": str(neutralization_data)}

    daily_ic = compute_daily_ic(factor_values, forward_returns)
    metrics = compute_predictive_metrics(factor_values, forward_returns)
    quantile_returns, weights = compute_quantile_returns(
        factor_values,
        forward_returns,
        quantiles=quantiles,
    )
    turnover = compute_turnover(weights)
    returns = build_strategy_returns(
        quantile_returns,
        turnover=turnover,
        cost_bps=cost_bps,
    )
    annual_metrics = compute_annual_metrics(
        daily_ic=daily_ic,
        returns=returns,
        factor_values=factor_values,
        forward_returns=forward_returns,
        horizon_days=config.horizon_days,
    )
    summary = build_summary(
        metrics=metrics.model_dump(mode="python"),
        returns=returns,
        daily_ic=daily_ic,
        annual_metrics=annual_metrics,
        horizon_days=config.horizon_days,
    )

    output_dir = _backtest_output_dir(Path(output_root), factor_input.factor_id)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    audit_path = output_dir / "backtest_audit.json"

    quantile_returns.to_csv(output_dir / "quantile_returns.csv", index_label="date")
    daily_ic.to_csv(output_dir / "daily_ic.csv", index_label="date")
    annual_metrics.to_csv(output_dir / "annual_metrics.csv", index=False)
    write_quantile_svg(quantile_returns, plots_dir / "quantile_cumulative_returns.svg")

    report = {
        "factor_id": factor_input.factor_id,
        "factor_name": factor_input.factor_name,
        "candidate_id": factor_input.candidate_id,
        "run_id": factor_input.run_id,
        "pool": factor_input.pool,
        "domain_agent": factor_input.domain_agent,
        "created_at": datetime.now(UTC).isoformat(),
        "data_dir": str(data_path),
        "data_version": metadata.get("data_version", "unversioned"),
        "start_date": str(ohlcv_panel.index.get_level_values("date").min().date()),
        "end_date": str(ohlcv_panel.index.get_level_values("date").max().date()),
        "horizon_days": config.horizon_days,
        "trade_delay_days": config.trade_delay_days,
        "return_price_column": config.return_price_column,
        "cost_bps": cost_bps,
        "quantiles": quantiles,
        "neutralization": neutralization,
        "summary": summary,
        "artifacts": {
            "daily_ic": "daily_ic.csv",
            "quantile_returns": "quantile_returns.csv",
            "annual_metrics": "annual_metrics.csv",
            "quantile_plot": "plots/quantile_cumulative_returns.svg",
            "backtest_audit": "backtest_audit.json",
        },
    }
    _write_json(report_path, report)
    _write_json(
        audit_path,
        build_backtest_audit(
            report=report,
            annual_metrics=annual_metrics,
            factor_input=factor_input,
        ),
    )
    return FactorBacktestResult(
        output_dir=output_dir,
        report_path=report_path,
        audit_path=audit_path,
        counts={
            "daily_rows": int(len(returns)),
            "annual_rows": int(len(annual_metrics)),
            "quantiles": int(quantiles),
        },
    )


def compute_daily_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Return daily IC and RankIC time series."""

    factors, returns = factor_values.align(forward_returns, join="inner", axis=None)
    records: list[dict[str, Any]] = []
    for date, factor_row in factors.iterrows():
        return_row = returns.loc[date]
        valid = factor_row.notna() & return_row.notna()
        x = factor_row.loc[valid].astype(float)
        y = return_row.loc[valid].astype(float)
        if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
            ic = np.nan
            rank_ic = np.nan
        else:
            ic = float(x.corr(y))
            rank_ic = float(x.rank(method="average").corr(y.rank(method="average")))
        records.append(
            {
                "date": pd.Timestamp(date),
                "ic": ic,
                "rank_ic": rank_ic,
                "coverage_count": int(valid.sum()),
                "universe_count": int(return_row.notna().sum()),
                "coverage": _safe_div(float(valid.sum()), float(return_row.notna().sum())),
            }
        )
    return pd.DataFrame.from_records(records).set_index("date").sort_index()


def compute_quantile_returns(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantiles: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return quantile forward returns and daily equal-weight portfolio weights."""

    factors, returns = factor_values.align(forward_returns, join="inner", axis=None)
    quantile_rows: list[dict[str, Any]] = []
    top_weights: dict[pd.Timestamp, pd.Series] = {}
    bottom_weights: dict[pd.Timestamp, pd.Series] = {}
    long_short_weights: dict[pd.Timestamp, pd.Series] = {}

    for date, factor_row in factors.iterrows():
        return_row = returns.loc[date]
        valid = factor_row.notna() & return_row.notna()
        x = factor_row.loc[valid].astype(float)
        y = return_row.loc[valid].astype(float)
        row: dict[str, Any] = {"date": pd.Timestamp(date)}
        if len(x) < quantiles or x.nunique() < quantiles:
            for index in range(1, quantiles + 1):
                row[f"q{index}"] = np.nan
            row["market"] = float(y.mean()) if len(y) else np.nan
            row["top_excess"] = np.nan
            row["long_short_gross"] = np.nan
            quantile_rows.append(row)
            continue

        labels = pd.qcut(x.rank(method="first"), q=quantiles, labels=False) + 1
        for index in range(1, quantiles + 1):
            members = labels[labels == index].index
            row[f"q{index}"] = float(y.loc[members].mean()) if len(members) else np.nan

        top_assets = labels[labels == quantiles].index
        bottom_assets = labels[labels == 1].index
        top_return = row[f"q{quantiles}"]
        bottom_return = row["q1"]
        row["market"] = float(y.mean())
        row["top_excess"] = float(top_return - row["market"])
        row["long_short_gross"] = float(top_return - bottom_return)

        top = pd.Series(1.0 / len(top_assets), index=top_assets, dtype=float)
        bottom = pd.Series(1.0 / len(bottom_assets), index=bottom_assets, dtype=float)
        long_short = top.add(-bottom, fill_value=0.0)
        top_weights[pd.Timestamp(date)] = top
        bottom_weights[pd.Timestamp(date)] = bottom
        long_short_weights[pd.Timestamp(date)] = long_short
        quantile_rows.append(row)

    return (
        pd.DataFrame.from_records(quantile_rows).set_index("date").sort_index(),
        {
            "top": _weights_to_frame(top_weights),
            "bottom": _weights_to_frame(bottom_weights),
            "long_short": _weights_to_frame(long_short_weights),
        },
    )


def build_strategy_returns(
    quantile_returns: pd.DataFrame,
    *,
    turnover: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Merge gross returns, turnover, and transaction-cost-adjusted returns."""

    returns = quantile_returns.copy()
    returns = returns.join(turnover, how="left")
    returns["long_short_turnover"] = returns["long_short_turnover"].fillna(0.0)
    returns["transaction_cost"] = returns["long_short_turnover"] * float(cost_bps) / 10000.0
    returns["long_short_net"] = (
        returns["long_short_gross"] - returns["transaction_cost"]
    )
    return returns


def compute_turnover(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute one-way turnover for top, bottom, and long-short weights."""

    records: list[dict[str, Any]] = []
    dates = weights["long_short"].index
    previous = {name: None for name in weights}
    for date in dates:
        row = {"date": pd.Timestamp(date)}
        for name, frame in weights.items():
            current = frame.loc[date].dropna()
            if previous[name] is None:
                value = 0.0
            else:
                aligned_current, aligned_previous = current.align(previous[name], fill_value=0.0)
                value = 0.5 * float((aligned_current - aligned_previous).abs().sum())
            row[f"{name}_turnover"] = value
            previous[name] = current
        records.append(row)
    if not records:
        return pd.DataFrame(
            columns=["top_turnover", "bottom_turnover", "long_short_turnover"]
        )
    return pd.DataFrame.from_records(records).set_index("date").sort_index()


def compute_annual_metrics(
    *,
    daily_ic: pd.DataFrame,
    returns: pd.DataFrame,
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Return calendar-year stability metrics."""

    records: list[dict[str, Any]] = []
    for year, frame in returns.groupby(returns.index.year):
        ic_year = daily_ic.loc[daily_ic.index.year == year]
        coverage = compute_year_coverage(factor_values, forward_returns, int(year))
        records.append(
            {
                "year": int(year),
                "ic_mean": _finite_mean(ic_year["ic"]),
                "rank_ic_mean": _finite_mean(ic_year["rank_ic"]),
                "top_excess_annual_return": annualize_return(
                    frame["top_excess"],
                    horizon_days=horizon_days,
                ),
                "long_short_gross_annual_return": annualize_return(
                    frame["long_short_gross"],
                    horizon_days=horizon_days,
                ),
                "long_short_net_annual_return": annualize_return(
                    frame["long_short_net"],
                    horizon_days=horizon_days,
                ),
                "max_drawdown": max_drawdown(frame["long_short_net"]),
                "avg_turnover": _finite_mean(frame["long_short_turnover"]),
                "avg_coverage": coverage,
            }
        )
    return pd.DataFrame.from_records(records)


def build_summary(
    *,
    metrics: dict[str, float],
    returns: pd.DataFrame,
    daily_ic: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    horizon_days: int,
) -> dict[str, Any]:
    """Build compact JSON summary metrics."""

    return {
        "ic_mean": float(metrics["ic"]),
        "rank_ic_mean": float(metrics["rank_ic"]),
        "icir": float(metrics["icir"]),
        "rank_icir": float(metrics["rank_icir"]),
        "mi": float(metrics["mi"]),
        "long_short_gross_annual_return": annualize_return(
            returns["long_short_gross"],
            horizon_days=horizon_days,
        ),
        "long_short_net_annual_return": annualize_return(
            returns["long_short_net"],
            horizon_days=horizon_days,
        ),
        "top_excess_annual_return": annualize_return(
            returns["top_excess"],
            horizon_days=horizon_days,
        ),
        "max_drawdown": max_drawdown(returns["long_short_net"]),
        "avg_turnover": _finite_mean(returns["long_short_turnover"]),
        "avg_coverage": _finite_mean(daily_ic["coverage"]),
        "daily_ic_rows": int(daily_ic["ic"].notna().sum()),
        "annual_years": [
            int(year)
            for year in annual_metrics.get("year", pd.Series(dtype=int)).tolist()
        ],
    }


def build_backtest_audit(
    *,
    report: dict[str, Any],
    annual_metrics: pd.DataFrame,
    factor_input: FactorBacktestInput,
) -> dict[str, Any]:
    """Build bounded feedback evidence for factor_memory."""

    summary = dict(report["summary"])
    bottlenecks: list[str] = []
    if summary["long_short_net_annual_return"] <= 0:
        bottlenecks.append("long_short_net_annual_return")
    if summary["rank_ic_mean"] <= 0:
        bottlenecks.append("rank_ic_mean")
    if summary["avg_coverage"] < 0.5:
        bottlenecks.append("coverage")
    if summary["long_short_net_annual_return"] < summary["long_short_gross_annual_return"] * 0.5:
        bottlenecks.append("transaction_cost")
    outcome = "backtest_success" if not bottlenecks else "backtest_failure"
    return {
        "audit_id": f"backtest:{factor_input.factor_id}:{report['created_at']}",
        "created_at": report["created_at"],
        "factor_id": factor_input.factor_id,
        "factor_name": factor_input.factor_name,
        "candidate_id": factor_input.candidate_id,
        "domain_agent": factor_input.domain_agent,
        "run_id": factor_input.run_id,
        "pool": factor_input.pool,
        "data_version": report["data_version"],
        "start_date": report["start_date"],
        "end_date": report["end_date"],
        "outcome": outcome,
        "bottlenecks": bottlenecks,
        "summary": summary,
        "annual_metrics": _records_for_json(annual_metrics),
    }


def neutralize_factor_values(
    factor_values: pd.DataFrame,
    *,
    neutralization_data: str | Path,
) -> pd.DataFrame:
    """Neutralize daily factor values by log market cap and industry dummies."""

    exposures = pd.read_parquet(neutralization_data)
    required = {"date", "asset", "industry", "market_cap"}
    missing = sorted(required - set(exposures.columns))
    if missing:
        raise ValueError(f"neutralization data missing columns: {missing}")
    exposures = exposures.copy()
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")
    exposures["asset"] = exposures["asset"].astype(str)
    exposures["market_cap"] = pd.to_numeric(exposures["market_cap"], errors="coerce")
    exposures = exposures.dropna(subset=["date", "asset", "industry", "market_cap"])
    exposures = exposures.set_index(["date", "asset"]).sort_index()

    neutralized: dict[pd.Timestamp, pd.Series] = {}
    for date, row in factor_values.iterrows():
        values = row.dropna().rename("factor")
        if values.empty:
            neutralized[pd.Timestamp(date)] = row * np.nan
            continue
        try:
            exposure_frame = exposures.loc[pd.Timestamp(date)]
        except KeyError:
            neutralized[pd.Timestamp(date)] = values
            continue
        frame = values.to_frame().join(exposure_frame, how="inner")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 5 or frame["industry"].nunique() < 2:
            neutralized[pd.Timestamp(date)] = values
            continue
        y = frame["factor"].astype(float).to_numpy()
        log_size = np.log(frame["market_cap"].astype(float).clip(lower=1.0)).rename("log_mcap")
        industry = pd.get_dummies(frame["industry"].astype(str), prefix="industry", drop_first=True)
        x_frame = pd.concat([log_size, industry], axis=1).astype(float)
        x = np.column_stack([np.ones(len(x_frame)), x_frame.to_numpy(dtype=float)])
        if x.shape[0] <= x.shape[1]:
            neutralized[pd.Timestamp(date)] = values
            continue
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = pd.Series(y - x @ beta, index=frame.index, dtype=float)
        neutralized[pd.Timestamp(date)] = residual
    return pd.DataFrame(neutralized).T.sort_index().sort_index(axis=1)


def write_quantile_svg(quantile_returns: pd.DataFrame, path: Path) -> None:
    """Write a minimal cumulative quantile-return SVG without plotting dependencies."""

    quantile_columns = [column for column in quantile_returns.columns if column.startswith("q")]
    cumulative = (1.0 + quantile_returns[quantile_columns].fillna(0.0)).cumprod() - 1.0
    width = 900
    height = 420
    margin = 48
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
    values = cumulative.to_numpy(dtype=float)
    if values.size == 0:
        min_value, max_value = -0.01, 0.01
    else:
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if min_value == max_value:
            min_value -= 0.01
            max_value += 0.01
    x_denominator = max(len(cumulative.index) - 1, 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = margin + index * (width - margin * 2) / x_denominator
        y = height - margin - (value - min_value) * (height - margin * 2) / (
            max_value - min_value
        )
        return x, y

    paths: list[str] = []
    for idx, column in enumerate(quantile_columns):
        series = cumulative[column].ffill().fillna(0.0)
        coords = [point(i, float(value)) for i, value in enumerate(series)]
        if not coords:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}"
            for i, (x, y) in enumerate(coords)
        )
        color = colors[idx % len(colors)]
        paths.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        paths.append(
            f'<text x="{width - margin + 8}" y="{margin + idx * 18}" '
            f'font-size="12" fill="{color}">{column}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#111827" stroke-width="1"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="#111827" stroke-width="1"/>'
        '<text x="48" y="24" font-size="16" fill="#111827">'
        'Quantile cumulative returns</text>'
        + "".join(paths)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def annualize_return(values: pd.Series, *, horizon_days: int) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    periods_per_year = 252.0 / float(horizon_days)
    mean_return = float(finite.mean())
    return float((1.0 + mean_return) ** periods_per_year - 1.0)


def max_drawdown(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if finite.empty:
        return 0.0
    equity = (1.0 + finite).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def compute_year_coverage(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    year: int,
) -> float:
    factors, returns = factor_values.align(forward_returns, join="inner", axis=None)
    mask = factors.index.year == year
    if not mask.any():
        return 0.0
    valid = factors.loc[mask].notna() & returns.loc[mask].notna()
    universe = returns.loc[mask].notna()
    return _safe_div(float(valid.sum().sum()), float(universe.sum().sum()))


def _weights_to_frame(weights: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    if not weights:
        return pd.DataFrame()
    return pd.DataFrame(weights).T.sort_index().sort_index(axis=1)


def _finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    return float(finite.mean())


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace([np.inf, -np.inf], np.nan).to_json(orient="records"))


def _read_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _backtest_output_dir(output_root: Path, factor_id: int) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / f"factor-{factor_id}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
