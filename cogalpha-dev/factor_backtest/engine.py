"""Alphalens-backed cross-sectional factor research engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogalpha.config import BaselineExperimentConfig
from cogalpha.data import load_qlib_daily_pv_hdf, normalize_ohlcv_panel, slice_ohlcv_panel
from cogalpha.execution import execute_alpha_candidate
from factor_backtest.alphalens import run_alphalens_factor_analysis
from factor_backtest.loader import FactorBacktestInput


@dataclass(frozen=True)
class FactorBacktestResult:
    """Paths and summary for one independent factor analysis."""

    output_dir: Path
    report_path: Path
    counts: dict[str, int]


def run_factor_backtest(
    *,
    factor_input: FactorBacktestInput,
    data_dir: str | Path,
    output_root: str | Path = "outputs/backtests",
    start_date: str | None = None,
    end_date: str | None = None,
    quantiles: int = 5,
    neutralization_data: str | Path | None = None,
    analysis_periods: tuple[int, ...] | None = None,
) -> FactorBacktestResult:
    """Run Alphalens factor analysis; no trade simulation is performed."""

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    data_path = Path(data_dir)
    metadata = _read_metadata(data_path)
    config = BaselineExperimentConfig()
    ohlcv_panel = _load_full_ohlcv_panel(data_path)
    if start_date is not None or end_date is not None:
        dates = ohlcv_panel.index.get_level_values("date")
        start = start_date or str(dates.min().date())
        end = end_date or str(dates.max().date())
        ohlcv_panel = slice_ohlcv_panel(ohlcv_panel, start=start, end=end)

    raw_factor_values = execute_alpha_candidate(factor_input.candidate, ohlcv_panel)
    factor_direction = int(factor_input.fitness_direction)
    factor_values = raw_factor_values * factor_direction
    neutralization = {"status": "skipped"}
    if neutralization_data is not None:
        factor_values = neutralize_factor_values(
            factor_values,
            neutralization_data=neutralization_data,
        )
        neutralization = {"status": "applied", "file": str(neutralization_data)}

    output_dir = _backtest_output_dir(Path(output_root), factor_input.factor_id)
    plots_dir = output_dir / "plots"
    report_path = output_dir / "report.json"

    analysis = run_alphalens_factor_analysis(
        factor_values=factor_values,
        ohlcv_panel=ohlcv_panel,
        price_column=config.return_price_column,
        horizon_days=config.horizon_days,
        trade_delay_days=config.trade_delay_days,
        quantiles=quantiles,
        plots_dir=plots_dir,
        analysis_periods=analysis_periods,
    )
    summary = build_alphalens_summary(
        daily_ic=analysis.daily_ic,
        quantile_excess_returns=analysis.quantile_excess_returns,
        long_short_returns=analysis.long_short_returns,
        quantiles=quantiles,
    )

    analysis.factor_data.to_csv(output_dir / "alphalens_factor_data.csv")
    analysis.daily_ic.to_csv(output_dir / "daily_ic.csv", index_label="date")
    analysis.quantile_excess_returns.to_csv(
        output_dir / "quantile_excess_returns.csv",
        index_label="date",
    )
    analysis.quantile_raw_returns.to_csv(
        output_dir / "quantile_raw_returns.csv",
        index_label="date",
    )
    analysis.long_short_returns.to_csv(
        output_dir / "long_short_returns.csv",
        index_label="date",
    )
    analysis.daily_ic_by_period.to_csv(
        output_dir / "daily_ic_by_period.csv",
        index_label="date",
    )
    analysis.long_short_returns_by_period.to_csv(
        output_dir / "long_short_returns_by_period.csv",
        index_label="date",
    )
    quantile_excess_by_period_artifacts: dict[str, str] = {}
    quantile_raw_by_period_artifacts: dict[str, str] = {}
    for period_column, returns in analysis.quantile_excess_returns_by_period.items():
        filename = f"quantile_excess_returns_{_artifact_period_name(period_column)}.csv"
        returns.to_csv(output_dir / filename, index_label="date")
        quantile_excess_by_period_artifacts[period_column] = filename
    for period_column, returns in analysis.quantile_raw_returns_by_period.items():
        filename = f"quantile_raw_returns_{_artifact_period_name(period_column)}.csv"
        returns.to_csv(output_dir / filename, index_label="date")
        quantile_raw_by_period_artifacts[period_column] = filename

    report = {
        "factor_id": factor_input.factor_id,
        "factor_name": factor_input.factor_name,
        "candidate_id": factor_input.candidate_id,
        "run_id": factor_input.run_id,
        "pool": factor_input.pool,
        "domain_agent": factor_input.domain_agent,
        "created_at": datetime.now(UTC).isoformat(),
        "engine": "alphalens",
        "primary_view": "top_quantile_excess_returns",
        "auxiliary_view": "top_minus_bottom_long_short_returns",
        "data_dir": str(data_path),
        "data_version": metadata.get("data_version", "unversioned"),
        "start_date": str(ohlcv_panel.index.get_level_values("date").min().date()),
        "end_date": str(ohlcv_panel.index.get_level_values("date").max().date()),
        "horizon_days": config.horizon_days,
        "period_column": analysis.primary_period_column,
        "primary_period_column": analysis.primary_period_column,
        "analysis_periods_requested": list(analysis_periods)
        if analysis_periods is not None
        else None,
        "diagnostic_period_columns": analysis.period_columns,
        "trade_delay_days": config.trade_delay_days,
        "return_price_column": config.return_price_column,
        "factor_direction": factor_direction,
        "quantiles": quantiles,
        "neutralization": neutralization,
        "summary": summary,
        "artifacts": {
            "alphalens_factor_data": "alphalens_factor_data.csv",
            "daily_ic": "daily_ic.csv",
            "quantile_excess_returns": "quantile_excess_returns.csv",
            "quantile_raw_returns": "quantile_raw_returns.csv",
            "long_short_returns": "long_short_returns.csv",
            "daily_ic_by_period": "daily_ic_by_period.csv",
            "long_short_returns_by_period": "long_short_returns_by_period.csv",
            "quantile_excess_returns_by_period": quantile_excess_by_period_artifacts,
            "quantile_raw_returns_by_period": quantile_raw_by_period_artifacts,
            "tear_sheets": [
                str(path.relative_to(output_dir))
                for path in analysis.tear_sheet_paths
            ],
        },
    }
    _write_json(report_path, report)
    return FactorBacktestResult(
        output_dir=output_dir,
        report_path=report_path,
        counts={
            "factor_rows": int(len(analysis.factor_data)),
            "daily_ic_rows": int(len(analysis.daily_ic)),
            "diagnostic_periods": int(len(analysis.period_columns)),
            "quantiles": int(quantiles),
            "tear_sheet_figures": int(len(analysis.tear_sheet_paths)),
        },
    )


def build_alphalens_summary(
    *,
    daily_ic: pd.DataFrame,
    quantile_excess_returns: pd.DataFrame,
    long_short_returns: pd.Series,
    quantiles: int,
) -> dict[str, Any]:
    """Build a small index for batch reports; Alphalens owns the analysis itself."""

    top_quantile = int(quantiles)
    top_excess = quantile_excess_returns[top_quantile]
    return {
        "rank_ic_mean": _finite_mean(daily_ic["rank_ic"]),
        "rank_icir": ratio_mean_to_std(daily_ic["rank_ic"], annualize=False),
        "top_quantile_excess_mean_return": _finite_mean(top_excess),
        "top_quantile_excess_positive_rate": positive_rate(top_excess),
        "long_short_mean_return": _finite_mean(long_short_returns),
        "long_short_positive_rate": positive_rate(long_short_returns),
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


def ratio_mean_to_std(
    values: pd.Series,
    *,
    annualize: bool = True,
) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    std = float(finite.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float(finite.mean()) / std


def positive_rate(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    return float((finite > 0).mean())


def _artifact_period_name(period_column: str) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in str(period_column)
    ).strip("_")


def _load_full_ohlcv_panel(data_dir: Path) -> pd.DataFrame:
    full_path = data_dir / "ohlcv_panel.parquet"
    try:
        return normalize_ohlcv_panel(pd.read_parquet(full_path))
    except OSError:
        split_frames = []
        for split_name in ("train", "valid", "test"):
            split_path = data_dir / f"{split_name}_ohlcv.parquet"
            if split_path.exists():
                try:
                    split_frames.append(pd.read_parquet(split_path))
                except OSError:
                    split_frames = []
                    break
        if not split_frames:
            raw_path = data_dir.parents[1] / "raw" / data_dir.name / "daily_pv.h5"
            return load_qlib_daily_pv_hdf(raw_path)
        return normalize_ohlcv_panel(pd.concat(split_frames, ignore_index=True))


def _finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
    return float(finite.mean())


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
