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


def test_panel_backed_metrics_provider_reuses_runtime_guard_factor_values(monkeypatch):
    panel = _make_two_asset_panel()
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
    )
    calls = {"count": 0}

    from cogalpha import execution

    original_execute = execution.execute_alpha_candidate

    def counting_execute(candidate, ohlcv_panel):
        calls["count"] += 1
        return original_execute(candidate, ohlcv_panel)

    monkeypatch.setattr(execution, "execute_alpha_candidate", counting_execute)
    monkeypatch.setattr(
        "cogalpha.guards.alpha_runtime.execute_alpha_candidate",
        counting_execute,
    )

    metrics = provider.evaluate([_make_gap_candidate()])

    assert "gap" in metrics
    assert calls["count"] == 1


def test_panel_backed_metrics_provider_normalizes_negative_factor_direction():
    panel = _make_two_asset_panel()
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
    )

    result = provider.evaluate_candidates([_make_inverse_gap_candidate()])[0]

    assert result.fitness_direction == -1
    assert result.metrics is not None
    assert result.raw_metrics is not None
    assert result.metrics.rank_ic == pytest.approx(1.0)
    assert result.raw_metrics.rank_ic == pytest.approx(-1.0)
    assert result.metrics.ic > 0


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


def _make_inverse_gap_candidate() -> AlphaCandidate:
    code = '''
def factor_inverse_gap(df):
    df_copy = df.copy()
    df_copy["factor_inverse_gap"] = df_copy["open"] - df_copy["close"]
    return df_copy["factor_inverse_gap"]
'''
    return AlphaCandidate(
        candidate_id="inverse_gap",
        alpha=AlphaFunction(
            name="factor_inverse_gap",
            code=code,
            rationale="Measures the inverse intraday body direction.",
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
