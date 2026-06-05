import pandas as pd

from cogalpha.execution import AlphaExecutionError, execute_alpha_candidate
from cogalpha.guards import run_runtime_alpha_code_guard
from cogalpha.schemas import AlphaCandidate, AlphaFunction, GuardStatus


def make_ohlcv_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["a", "b"]],
        names=["date", "asset"],
    )
    return pd.DataFrame(
        {
            "open": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
            "high": [11.0, 21.0, 12.0, 20.0, 13.0, 19.0],
            "low": [9.0, 19.0, 10.0, 18.0, 11.0, 17.0],
            "close": [10.5, 19.5, 11.5, 18.5, 12.5, 17.5],
            "volume": [100.0, 200.0, 110.0, 190.0, 120.0, 180.0],
        },
        index=index,
    )


def make_candidate(code: str, name: str = "factor_close_open_gap") -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=name,
        alpha=AlphaFunction(
            name=name,
            code=code,
            rationale="Measures intraday body direction.",
        ),
    )


def test_execute_alpha_candidate_returns_date_by_asset_panel():
    code = '''
def factor_close_open_gap(df):
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''

    result = execute_alpha_candidate(make_candidate(code), make_ohlcv_panel())

    assert list(result.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert list(result.columns) == ["a", "b"]
    assert result.loc[pd.Timestamp("2024-01-01"), "a"] == 0.5
    assert result.loc[pd.Timestamp("2024-01-01"), "b"] == -0.5


def test_execute_alpha_candidate_rejects_index_mismatch():
    code = '''
def factor_close_open_gap(df):
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_close_open_gap"].iloc[1:]
'''

    try:
        execute_alpha_candidate(make_candidate(code), make_ohlcv_panel())
    except AlphaExecutionError as exc:
        assert "output index" in str(exc)
    else:
        raise AssertionError("Expected AlphaExecutionError")


def test_runtime_guard_rejects_all_nan_output():
    code = '''
def factor_all_nan(df):
    df_copy = df.copy()
    df_copy["factor_all_nan"] = pd.Series(float("nan"), index=df_copy.index)
    return df_copy["factor_all_nan"]
'''

    report = run_runtime_alpha_code_guard(
        make_candidate(code, name="factor_all_nan"),
        make_ohlcv_panel(),
    )

    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "all_nan_output" for issue in report.issues)
    assert report.metadata["nan_fraction"] == 1.0
