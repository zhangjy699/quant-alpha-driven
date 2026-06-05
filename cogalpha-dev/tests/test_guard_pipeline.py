import pandas as pd

from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.schemas import AlphaCandidate, AlphaFunction, GuardStatus


def make_ohlcv_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["a"]],
        names=["date", "asset"],
    )
    return pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100.0, 110.0],
        },
        index=index,
    )


def make_candidate(code: str, name: str = "factor_test") -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=name,
        alpha=AlphaFunction(name=name, code=code, rationale="Test factor."),
    )


def test_guard_pipeline_runs_runtime_guard_after_static_pass():
    code = '''
def factor_test(df):
    df_copy = df.copy()
    df_copy["factor_test"] = pd.Series(float("nan"), index=df_copy.index)
    return df_copy["factor_test"]
'''

    outcome = DeterministicGuardPipeline(runtime_ohlcv_panel=make_ohlcv_panel()).run(
        make_candidate(code)
    )

    assert [report.guard_name for report in outcome.reports] == [
        "static_alpha_code",
        "runtime_alpha_code",
    ]
    assert outcome.failed
    assert outcome.reports[-1].status == GuardStatus.FAIL
    assert any(issue.code == "all_nan_output" for issue in outcome.reports[-1].issues)


def test_guard_pipeline_stops_runtime_guard_after_static_failure():
    code = '''
def factor_test(df):
    df_copy = df.copy()
    df_copy["factor_test"] = df_copy["close"].shift(-1)
    return df_copy["factor_test"]
'''

    outcome = DeterministicGuardPipeline(runtime_ohlcv_panel=make_ohlcv_panel()).run(
        make_candidate(code)
    )

    assert [report.guard_name for report in outcome.reports] == ["static_alpha_code"]
    assert outcome.failed
