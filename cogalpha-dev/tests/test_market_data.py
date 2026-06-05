from datetime import date

import pandas as pd
import pytest

from cogalpha.config import BaselineExperimentConfig, SplitConfig
from cogalpha.data import (
    build_baseline_market_data,
    compute_forward_returns,
    load_prepared_baseline_market_data,
    normalize_ohlcv_panel,
)


def test_normalize_ohlcv_panel_accepts_flat_frame_and_sorts():
    raw = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-01"],
            "ticker": ["b", "a"],
            "open": [20.0, 10.0],
            "high": [21.0, 11.0],
            "low": [19.0, 9.0],
            "close": [20.5, 10.5],
            "volume": [200.0, 100.0],
        }
    )

    panel = normalize_ohlcv_panel(raw, date_column="trade_date", asset_column="ticker")

    assert panel.index.names == ["date", "asset"]
    assert list(panel.index) == [
        (pd.Timestamp("2024-01-01"), "a"),
        (pd.Timestamp("2024-01-02"), "b"),
    ]
    assert list(panel.columns) == ["open", "high", "low", "close", "volume"]


def test_compute_forward_returns_aligns_to_current_date():
    panel = _make_single_asset_panel(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        close=[100.0, 110.0, 121.0, 133.1],
    )

    returns = compute_forward_returns(panel, horizon_days=2)

    assert returns.loc[pd.Timestamp("2024-01-01"), "a"] == pytest.approx(0.21)
    assert returns.loc[pd.Timestamp("2024-01-02"), "a"] == pytest.approx(0.21)
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-03"), "a"])
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-04"), "a"])


def test_compute_forward_returns_can_delay_entry_until_next_open():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    panel = _make_single_asset_panel(dates, close=[100.0, 110.0, 121.0, 133.1])

    returns = compute_forward_returns(
        panel,
        horizon_days=1,
        price_column="open",
        trade_delay_days=1,
    )

    assert returns.loc[pd.Timestamp("2024-01-01"), "a"] == pytest.approx((120.0 / 109.0) - 1.0)
    assert returns.loc[pd.Timestamp("2024-01-02"), "a"] == pytest.approx((132.1 / 120.0) - 1.0)
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-03"), "a"])


def test_baseline_market_data_computes_returns_inside_each_split_only():
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    panel = _make_single_asset_panel(
        dates,
        close=[100.0, 110.0, 121.0, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881],
    )
    config = BaselineExperimentConfig(
        horizon_days=1,
        split=SplitConfig(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 1, 3),
            valid_start=date(2024, 1, 4),
            valid_end=date(2024, 1, 6),
            test_start=date(2024, 1, 7),
            test_end=date(2024, 1, 9),
        ),
    )

    dataset = build_baseline_market_data(panel, config)

    assert dataset.train.forward_returns.loc[pd.Timestamp("2024-01-01"), "a"] == pytest.approx(
        (120.0 / 109.0) - 1.0
    )
    assert pd.isna(dataset.train.forward_returns.loc[pd.Timestamp("2024-01-03"), "a"])
    assert pd.isna(dataset.valid.forward_returns.loc[pd.Timestamp("2024-01-05"), "a"])
    assert pd.isna(dataset.test.forward_returns.loc[pd.Timestamp("2024-01-08"), "a"])


def test_load_prepared_baseline_market_data_reads_processed_parquets(tmp_path):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    panel = _make_single_asset_panel(dates, close=[100.0, 101.0, 102.0])
    returns = compute_forward_returns(panel, horizon_days=1)
    panel.reset_index().to_parquet(tmp_path / "ohlcv_panel.parquet", index=False)
    for split_name in ("train", "valid", "test"):
        panel.reset_index().to_parquet(tmp_path / f"{split_name}_ohlcv.parquet", index=False)
        returns.to_parquet(tmp_path / f"{split_name}_forward_returns.parquet")

    dataset = load_prepared_baseline_market_data(tmp_path)

    assert dataset.dataset == "CSI300"
    assert dataset.horizon_days == 10
    assert dataset.train.ohlcv_panel.index.names == ["date", "asset"]
    assert dataset.valid.forward_returns.equals(returns)


def _make_single_asset_panel(dates: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([dates, ["a"]], names=["date", "asset"])
    return pd.DataFrame(
        {
            "open": [value - 1.0 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 2.0 for value in close],
            "close": close,
            "volume": [100.0 + offset for offset, _ in enumerate(close)],
        },
        index=index,
    )
