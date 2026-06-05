from cogalpha.schemas import AlphaCandidate, AlphaFunction, CandidateStage


def test_alpha_candidate_schema_accepts_function_artifact():
    candidate = AlphaCandidate(
        candidate_id="alpha-1",
        alpha=AlphaFunction(
            name="factor_close_open_gap",
            code="def factor_close_open_gap(df):\n    return df['close'] - df['open']\n",
            rationale="Measures intraday body direction.",
        ),
    )

    assert candidate.stage == CandidateStage.GENERATED
    assert candidate.alpha.required_columns == ["open", "high", "low", "close", "volume"]

