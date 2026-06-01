"""Fixed-sample preflight artifacts for real-data evaluation readiness."""

from __future__ import annotations

from cogalpha.schemas import AlphaCandidate, AlphaFunction


def fixed_preflight_candidates() -> list[AlphaCandidate]:
    """Return deterministic Alpha Candidates used before formal workflow runs."""

    return [
        AlphaCandidate(
            candidate_id="preflight_intraday_body",
            alpha=AlphaFunction(
                name="factor_preflight_intraday_body",
                code='''
def factor_preflight_intraday_body(df):
    df_copy = df.copy()
    df_copy["factor_preflight_intraday_body"] = (
        df_copy["close"] - df_copy["open"]
    ) / (df_copy["open"].abs() + 1e-12)
    return df_copy["factor_preflight_intraday_body"]
''',
                formula="(close - open) / abs(open)",
                rationale="Measures same-day body direction as a simple OHLCV smoke factor.",
            ),
        ),
        AlphaCandidate(
            candidate_id="preflight_close_location",
            alpha=AlphaFunction(
                name="factor_preflight_close_location",
                code='''
def factor_preflight_close_location(df):
    df_copy = df.copy()
    denominator = (df_copy["high"] - df_copy["low"]).abs() + 1e-12
    df_copy["factor_preflight_close_location"] = (
        (df_copy["close"] - df_copy["low"]) / denominator
    ) - 0.5
    return df_copy["factor_preflight_close_location"]
''',
                formula="(close - low) / (high - low) - 0.5",
                rationale="Measures where the close falls within the daily range.",
            ),
        ),
        AlphaCandidate(
            candidate_id="preflight_volume_shock",
            alpha=AlphaFunction(
                name="factor_preflight_volume_shock",
                code='''
def factor_preflight_volume_shock(df):
    df_copy = df.copy()
    baseline = df_copy["volume"].rolling(20, min_periods=5).mean()
    df_copy["factor_preflight_volume_shock"] = (
        df_copy["volume"] / (baseline.abs() + 1e-12)
    ) - 1.0
    return df_copy["factor_preflight_volume_shock"]
''',
                formula="volume / rolling_mean(volume, 20) - 1",
                rationale="Measures abnormal volume using only past and present observations.",
            ),
        ),
    ]
