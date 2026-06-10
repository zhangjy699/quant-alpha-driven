import json

import pandas as pd
import pytest

from factor_backtest import load_factor_from_pool, run_factor_backtest
from factor_backtest.alphalens import prices_from_ohlcv_panel

pytest.importorskip("alphalens")


def test_factor_backtest_loads_factor_pool_id_and_writes_reports(tmp_path):
    factor_pool = _write_factor_pool(tmp_path)
    data_dir = _write_processed_data(tmp_path)

    factor_input = load_factor_from_pool(factor_id=7, factor_pool_root=factor_pool)
    assert factor_input.fitness_direction == -1
    result = run_factor_backtest(
        factor_input=factor_input,
        data_dir=data_dir,
        output_root=tmp_path / "outputs" / "backtests",
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.output_dir.parent == tmp_path / "outputs" / "backtests"
    assert report["factor_id"] == 7
    assert report["factor_direction"] == -1
    assert report["engine"] == "alphalens"
    assert report["primary_view"] == "top_quantile_excess_returns"
    assert report["neutralization"] == {"status": "skipped"}
    assert "rank_ic_mean" in report["summary"]
    assert "top_quantile_excess_mean_return" in report["summary"]
    assert "long_short_mean_return" in report["summary"]
    assert (result.output_dir / "alphalens_factor_data.csv").exists()
    assert (result.output_dir / "daily_ic.csv").exists()
    assert (result.output_dir / "quantile_excess_returns.csv").exists()
    assert (result.output_dir / "quantile_raw_returns.csv").exists()
    assert (result.output_dir / "long_short_returns.csv").exists()
    assert report["artifacts"]["tear_sheets"]


def test_factor_backtest_fails_fast_for_missing_factor_id(tmp_path):
    factor_pool = _write_factor_pool(tmp_path)

    try:
        load_factor_from_pool(factor_id=99, factor_pool_root=factor_pool)
    except ValueError as exc:
        assert "factor_id 99" in str(exc)
    else:
        raise AssertionError("Expected missing factor_id failure.")


def test_factor_backtest_applies_optional_neutralization(tmp_path):
    factor_pool = _write_factor_pool(tmp_path)
    data_dir = _write_processed_data(tmp_path)
    neutralization_path = _write_neutralization_data(tmp_path)

    result = run_factor_backtest(
        factor_input=load_factor_from_pool(factor_id=7, factor_pool_root=factor_pool),
        data_dir=data_dir,
        output_root=tmp_path / "outputs" / "backtests",
        neutralization_data=neutralization_path,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["neutralization"]["status"] == "applied"


def test_alphalens_prices_apply_trade_delay_before_forward_returns():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    index = pd.MultiIndex.from_product([dates, ["a"]], names=["date", "asset"])
    panel = pd.DataFrame(
        {
            "open": [10.0, 11.0, 13.2],
            "high": [10.0, 11.0, 13.2],
            "low": [10.0, 11.0, 13.2],
            "close": [10.0, 11.0, 13.2],
            "volume": [100.0, 100.0, 100.0],
        },
        index=index,
    )

    prices = prices_from_ohlcv_panel(
        panel,
        price_column="open",
        trade_delay_days=1,
    )

    assert prices.loc[pd.Timestamp("2024-01-01"), "a"] == 11.0
    assert prices.loc[pd.Timestamp("2024-01-02"), "a"] == 13.2


def _write_factor_pool(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    factor_path = factor_pool / "qualified/alpha-range-vol/7.json"
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    factor_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "factor_name": "factor_intraday_strength",
                "formula": "close / open - 1",
                "code": (
                    "def factor_intraday_strength(df):\n"
                    "    out = df['close'] / df['open'] - 1.0\n"
                    "    out.name = 'factor_intraday_strength'\n"
                    "    return out\n"
                ),
                "rationale": "Intraday close strength.",
                "required_columns": ["open", "high", "low", "close", "volume"],
                "allowed_libraries": ["np", "pd"],
                "fitness_direction": -1,
                "metrics": {
                    "ic": 0.01,
                    "rank_ic": 0.01,
                    "icir": 0.1,
                    "rank_icir": 0.1,
                    "mi": 0.02,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (factor_pool / "index.json").write_text(
        json.dumps(
            {
                "next_factor_id": 8,
                "counts": {"qualified": 1},
                "factors": [
                    {
                        "factor_id": 7,
                        "file": "qualified/alpha-range-vol/7.json",
                        "pool": "qualified",
                        "domain_agent": "alpha-range-vol",
                        "run_id": "run-1",
                        "candidate_id": "candidate-7",
                        "factor_name": "factor_intraday_strength",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return factor_pool


def _write_processed_data(tmp_path):
    data_dir = tmp_path / "data/processed/company_all_a"
    data_dir.mkdir(parents=True)
    dates = pd.bdate_range("2021-12-01", periods=36)
    assets = [f"SZ00000{i}" for i in range(1, 7)]
    rows = []
    for date_index, date in enumerate(dates):
        for asset_index, asset in enumerate(assets, start=1):
            base = 10.0 + asset_index + date_index * 0.05
            signal = (asset_index - 3.5) * 0.002 + (date_index % 3) * 0.0005
            open_price = base
            close = open_price * (1.0 + signal)
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": 1000 + asset_index * 100,
                }
            )
    pd.DataFrame(rows).to_parquet(data_dir / "ohlcv_panel.parquet", index=False)
    (data_dir / "metadata.json").write_text(
        json.dumps({"data_version": "unit-test-data"}),
        encoding="utf-8",
    )
    return data_dir


def _write_neutralization_data(tmp_path):
    dates = pd.bdate_range("2021-12-01", periods=36)
    assets = [f"SZ00000{i}" for i in range(1, 7)]
    rows = []
    for date in dates:
        for asset_index, asset in enumerate(assets, start=1):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "industry": "tech" if asset_index % 2 else "finance",
                    "market_cap": 1_000_000 + asset_index * 10_000,
                }
            )
    path = tmp_path / "neutralization.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
