"""Thin Alphalens adapter for CogAlpha factor_pool artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AlphalensAnalysis:
    """Alphalens-native factor analysis artifacts."""

    factor_data: pd.DataFrame
    daily_ic: pd.DataFrame
    daily_ic_by_period: pd.DataFrame
    quantile_excess_returns: pd.DataFrame
    quantile_excess_returns_by_period: dict[str, pd.DataFrame]
    quantile_raw_returns: pd.DataFrame
    quantile_raw_returns_by_period: dict[str, pd.DataFrame]
    long_short_returns: pd.Series
    long_short_returns_by_period: pd.DataFrame
    tear_sheet_paths: list[Path]
    primary_period_column: str
    period_columns: list[str]


def run_alphalens_factor_analysis(
    *,
    factor_values: pd.DataFrame,
    ohlcv_panel: pd.DataFrame,
    price_column: str,
    horizon_days: int,
    quantiles: int,
    plots_dir: Path,
    analysis_periods: tuple[int, ...] | None = None,
) -> AlphalensAnalysis:
    """Run Alphalens directly on adjusted factor values and prices."""

    al = import_alphalens()
    factor = factor_frame_to_series(factor_values)
    prices = prices_from_ohlcv_panel(ohlcv_panel, price_column=price_column)
    periods = _analysis_periods(horizon_days, analysis_periods=analysis_periods)
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        quantiles=quantiles,
        periods=periods,
        max_loss=0.95,
    )
    primary_period_column = _period_column(factor_data, horizon_days)
    period_columns = _period_columns(factor_data, periods)
    daily_ic_by_period = _daily_ic_by_period_from_alphalens(al, factor_data)
    daily_ic = daily_ic_by_period[primary_period_column].rename("rank_ic").to_frame()
    quantile_excess_returns_by_period = {
        period_column: _quantile_returns_from_alphalens(
            al,
            factor_data,
            period_column,
            demeaned=True,
        )
        for period_column in period_columns
    }
    quantile_raw_returns_by_period = {
        period_column: _quantile_returns_from_alphalens(
            al,
            factor_data,
            period_column,
            demeaned=False,
        )
        for period_column in period_columns
    }
    quantile_excess_returns = quantile_excess_returns_by_period[primary_period_column]
    quantile_raw_returns = quantile_raw_returns_by_period[primary_period_column]
    long_short_returns = (
        quantile_raw_returns[int(quantiles)] - quantile_raw_returns[1]
    ).rename("long_short_return")
    long_short_returns_by_period = pd.DataFrame(
        {
            period_column: (
                returns[int(quantiles)] - returns[1]
            )
            for period_column, returns in quantile_raw_returns_by_period.items()
        }
    ).sort_index()
    tear_sheet_paths = write_alphalens_tear_sheets(
        al=al,
        factor_data=factor_data,
        plots_dir=plots_dir,
    )
    return AlphalensAnalysis(
        factor_data=factor_data,
        daily_ic=daily_ic,
        daily_ic_by_period=daily_ic_by_period,
        quantile_excess_returns=quantile_excess_returns,
        quantile_excess_returns_by_period=quantile_excess_returns_by_period,
        quantile_raw_returns=quantile_raw_returns,
        quantile_raw_returns_by_period=quantile_raw_returns_by_period,
        long_short_returns=long_short_returns,
        long_short_returns_by_period=long_short_returns_by_period,
        tear_sheet_paths=tear_sheet_paths,
        primary_period_column=primary_period_column,
        period_columns=period_columns,
    )


def factor_frame_to_series(factor_values: pd.DataFrame) -> pd.Series:
    """Convert date x asset factor values to Alphalens MultiIndex Series."""

    frame = factor_values.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(str)
    # pandas' new stack implementation no longer accepts dropna; drop explicitly.
    series = frame.stack().dropna().rename("factor")
    series.index = series.index.set_names(["date", "asset"])
    return series.sort_index()


def prices_from_ohlcv_panel(
    ohlcv_panel: pd.DataFrame,
    *,
    price_column: str,
) -> pd.DataFrame:
    """Convert normalized OHLCV panel to Alphalens prices DataFrame."""

    frame = ohlcv_panel.reset_index()
    prices = frame.pivot(index="date", columns="asset", values=price_column)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str)
    return prices.sort_index().sort_index(axis=1)


def write_alphalens_tear_sheets(
    *,
    al: Any,
    factor_data: pd.DataFrame,
    plots_dir: Path,
) -> list[Path]:
    """Let Alphalens create figures, then persist all open matplotlib figures."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    paths: list[Path] = []
    original_show = plt.show

    def save_and_close_open_figures(*args, **kwargs):
        del args, kwargs
        while plt.get_fignums():
            figure_number = plt.get_fignums()[0]
            figure = plt.figure(figure_number)
            if not _figure_is_blank(figure):
                path = plots_dir / f"alphalens_tear_sheet_{len(paths) + 1:02d}.png"
                figure.savefig(path, bbox_inches="tight", dpi=150)
                paths.append(path)
            plt.close(figure)

    try:
        plt.show = save_and_close_open_figures
        al.tears.create_full_tear_sheet(
            factor_data,
            long_short=True,
            group_neutral=False,
            by_group=False,
        )
        save_and_close_open_figures()
    finally:
        plt.show = original_show
        plt.close("all")
    return paths


def _figure_is_blank(figure: Any) -> bool:
    figure.canvas.draw()
    rgba = figure.canvas.buffer_rgba()
    try:
        import numpy as np
    except ImportError:
        return False
    pixels = np.asarray(rgba)
    return bool(float(pixels[:, :, :3].std()) < 1.0)


def import_alphalens() -> Any:
    try:
        import alphalens as al
    except ImportError as exc:
        raise ImportError(
            "Alphalens is required for factor_backtest. Install "
            "`alphalens-reloaded>=0.4.6` in this environment."
        ) from exc
    return al


def _analysis_periods(
    horizon_days: int,
    *,
    analysis_periods: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    periods = [int(horizon_days)]
    if analysis_periods is not None:
        periods.extend(int(period) for period in analysis_periods)
    return tuple(dict.fromkeys(period for period in periods if period > 0))


def _period_column(factor_data: pd.DataFrame, horizon_days: int) -> str:
    candidates = [
        str(horizon_days),
        f"{int(horizon_days)}D",
        f"{int(horizon_days)}d",
        f"{int(horizon_days)} days",
    ]
    columns = [str(column) for column in factor_data.columns]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for column in columns:
        if column.startswith(str(int(horizon_days))) and column not in {
            "factor",
            "factor_quantile",
            "group",
        }:
            return column
    forward_columns = [
        str(column)
        for column in factor_data.columns
        if str(column) not in {"factor", "factor_quantile", "group"}
    ]
    if len(forward_columns) == 1:
        return forward_columns[0]
    raise ValueError(
        f"Could not identify Alphalens forward-return column for {horizon_days} days: "
        f"{columns}"
    )


def _period_columns(factor_data: pd.DataFrame, periods: tuple[int, ...]) -> list[str]:
    return [_period_column(factor_data, period) for period in periods]


def _daily_ic_by_period_from_alphalens(
    al: Any,
    factor_data: pd.DataFrame,
) -> pd.DataFrame:
    ic = al.performance.factor_information_coefficient(factor_data)
    if isinstance(ic, pd.Series):
        frame = ic.to_frame()
    else:
        frame = ic.copy()
    frame.columns = [str(column) for column in frame.columns]
    return frame.sort_index()


def _quantile_returns_from_alphalens(
    al: Any,
    factor_data: pd.DataFrame,
    period_column: str,
    *,
    demeaned: bool,
) -> pd.DataFrame:
    mean_returns, _ = al.performance.mean_return_by_quantile(
        factor_data,
        by_date=True,
        demeaned=demeaned,
    )
    series = mean_returns[period_column]
    if isinstance(series.index, pd.MultiIndex):
        quantile_level = "factor_quantile"
        if quantile_level not in series.index.names:
            quantile_level = series.index.names[0]
        frame = series.unstack(quantile_level)
    else:
        frame = series.to_frame()
    frame.columns = [int(column) for column in frame.columns]
    return frame.sort_index().sort_index(axis=1)
