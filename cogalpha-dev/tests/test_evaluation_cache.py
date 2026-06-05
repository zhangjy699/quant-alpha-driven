import pandas as pd

from cogalpha.data import compute_forward_returns
from cogalpha.evaluation import EvaluationCache, PanelBackedMetricsProvider
from cogalpha.schemas import AlphaCandidate, AlphaFunction


def test_panel_backed_metrics_provider_reuses_cached_metrics(tmp_path):
    panel = _make_panel()
    cache = EvaluationCache(tmp_path / "evaluation_cache.jsonl")
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
        data_version="test-data-v1",
        cache=cache,
    )
    candidate = _make_candidate("gap")

    first = provider.evaluate([candidate])
    second = provider.evaluate([candidate])

    assert first == second
    assert provider.cache_hits_by_candidate_id == {"gap": True}
    assert len(cache.load_all()) == 1


def test_evaluation_cache_key_changes_when_alpha_code_changes(tmp_path):
    panel = _make_panel()
    cache = EvaluationCache(tmp_path / "evaluation_cache.jsonl")
    provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
        data_version="test-data-v1",
        cache=cache,
    )

    provider.evaluate([_make_candidate("gap", offset=0.0)])
    provider.evaluate([_make_candidate("gap", offset=1.0)])

    assert len(cache.load_all()) == 2


def test_evaluation_cache_key_changes_when_split_changes(tmp_path):
    panel = _make_panel()
    cache = EvaluationCache(tmp_path / "evaluation_cache.jsonl")
    candidate = _make_candidate("gap")
    valid_provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
        data_version="test-data-v1",
        split_name="valid",
        cache=cache,
    )
    test_provider = PanelBackedMetricsProvider(
        ohlcv_panel=panel,
        forward_returns=compute_forward_returns(panel, horizon_days=1),
        data_version="test-data-v1",
        split_name="test",
        cache=cache,
    )

    valid_provider.evaluate([candidate])
    test_provider.evaluate([candidate])

    assert len(cache.load_all()) == 2
    assert {record.split_name for record in cache.load_all()} == {"valid", "test"}


def _make_candidate(candidate_id: str, offset: float = 0.0) -> AlphaCandidate:
    code = f'''
def factor_gap(df):
    df_copy = df.copy()
    df_copy["factor_gap"] = df_copy["close"] - df_copy["open"] + {offset}
    return df_copy["factor_gap"]
'''
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name="factor_gap",
            code=code,
            rationale="Measures whether the bar closes above or below the open.",
        ),
    )


def _make_panel() -> pd.DataFrame:
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
