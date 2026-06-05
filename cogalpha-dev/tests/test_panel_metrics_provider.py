import pandas as pd
import pytest

from cogalpha.data import compute_forward_returns
from cogalpha.evaluation import PanelBackedMetricsProvider
from cogalpha.schemas import AlphaCandidate, AlphaFunction, GuardStatus


def test_panel_backed_metrics_provider_evaluates_candidate_against_forward_returns():
    panel = _make_two_asset_panel()
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
    )

    metrics = provider.evaluate([_make_gap_candidate()])

    assert "gap" in metrics
    assert metrics["gap"].rank_ic == pytest.approx(1.0)
    assert provider.guard_reports_by_candidate_id["gap"].status == GuardStatus.PASS


def test_panel_backed_metrics_provider_skips_runtime_guard_failures():
    panel = _make_two_asset_panel()
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
    )

    metrics = provider.evaluate([_make_all_nan_candidate()])

    assert metrics == {}
    assert provider.guard_reports_by_candidate_id["bad"].status == GuardStatus.FAIL


def _make_gap_candidate() -> AlphaCandidate:
    code = '''
def factor_gap(df):
    df_copy = df.copy()
    df_copy["factor_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_gap"]
'''
    return AlphaCandidate(
        candidate_id="gap",
        alpha=AlphaFunction(
            name="factor_gap",
            code=code,
            rationale="Measures whether the bar closes above or below the open.",
        ),
    )


def _make_all_nan_candidate() -> AlphaCandidate:
    code = '''
def factor_bad(df):
    df_copy = df.copy()
    df_copy["factor_bad"] = pd.Series(float("nan"), index=df_copy.index)
    return df_copy["factor_bad"]
'''
    return AlphaCandidate(
        candidate_id="bad",
        alpha=AlphaFunction(
            name="factor_bad",
            code=code,
            rationale="Invalid all-NaN candidate.",
        ),
    )


def _make_two_asset_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    index = pd.MultiIndex.from_product([dates, ["a", "b"]], names=["date", "asset"])
    close = [11.0, 9.0, 12.0, 8.0, 13.0, 7.0, 14.0, 6.0]
    return pd.DataFrame(
        {
            "open": [10.0] * len(index),
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [100.0, 100.0, 110.0, 90.0, 120.0, 80.0, 130.0, 70.0],
        },
        index=index,
    )
