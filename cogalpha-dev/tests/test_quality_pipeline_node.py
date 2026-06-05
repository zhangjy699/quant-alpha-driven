from cogalpha.config import MVPLoopConfig
from cogalpha.nodes import QualityPipelineNode
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    QualityDecision,
    QualityVerdict,
    SkillKind,
    SkillRef,
)


def make_candidate(code: str, name: str = "factor_close_open_gap") -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id="candidate-1",
        alpha=AlphaFunction(
            name=name,
            code=code,
            rationale="Measures intraday body direction.",
        ),
    )


def make_decision(skill_name: str, verdict: QualityVerdict, repaired=None) -> QualityDecision:
    return QualityDecision(
        skill=SkillRef(
            name=skill_name,
            path=f"skills/{skill_name}/SKILL.md",
            kind=SkillKind.QUALITY_CHECKER,
        ),
        verdict=verdict,
        practical_soundness="Looks testable.",
        feedback="No issues.",
        repaired_candidate=repaired,
    )


class AcceptingQualityInvoker:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append(skill_name)
        return make_decision(skill_name, QualityVerdict.ACCEPT)


class RepairingQualityInvoker:
    def __init__(self, repaired_candidate) -> None:
        self.calls = []
        self.repaired_candidate = repaired_candidate

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append(skill_name)
        if skill_name == "alpha-code-repair":
            return make_decision(skill_name, QualityVerdict.REPAIR, self.repaired_candidate)
        return make_decision(skill_name, QualityVerdict.ACCEPT)


class FailingQualityInvoker:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append(skill_name)
        raise ValueError("invalid quality JSON")


def test_quality_pipeline_accepts_guard_clean_candidate():
    code = '''
def factor_close_open_gap(df):
    """Close-open gap."""
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''
    state = CogAlphaState(candidates=[make_candidate(code)]).model_dump(mode="python")
    invoker = AcceptingQualityInvoker()
    node = QualityPipelineNode(invoker=invoker, config=MVPLoopConfig())

    result = CogAlphaState.model_validate(node(state))

    assert len(result.candidates) == 1
    assert result.candidates[0].stage == CandidateStage.ACCEPTED_BY_QUALITY
    assert invoker.calls == ["alpha-code-quality", "alpha-judge"]


def test_quality_pipeline_repairs_guard_failure_then_accepts():
    bad_code = '''
def factor_close_open_gap(df):
    """Invalid future gap."""
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"].shift(-1) - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''
    repaired_code = '''
def factor_close_open_gap(df):
    """Close-open gap."""
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''
    repaired = make_candidate(repaired_code)
    state = CogAlphaState(candidates=[make_candidate(bad_code)]).model_dump(mode="python")
    invoker = RepairingQualityInvoker(repaired)
    node = QualityPipelineNode(invoker=invoker, config=MVPLoopConfig(max_repair_attempts=1))

    result = CogAlphaState.model_validate(node(state))

    assert len(result.candidates) == 1
    assert result.candidates[0].stage == CandidateStage.ACCEPTED_BY_QUALITY
    assert invoker.calls == ["alpha-code-repair", "alpha-code-quality", "alpha-judge"]
    assert result.node_history[-1].guard_reports[0].status == "fail"
    assert result.node_history[-1].guard_reports[1].status == "pass"


def test_quality_pipeline_zero_repair_budget_rejects_guard_failure_without_repair():
    bad_code = '''
def factor_close_open_gap(df):
    """Invalid future gap."""
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"].shift(-1) - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''
    repaired = make_candidate(bad_code)
    state = CogAlphaState(candidates=[make_candidate(bad_code)]).model_dump(mode="python")
    invoker = RepairingQualityInvoker(repaired)
    node = QualityPipelineNode(invoker=invoker, config=MVPLoopConfig(max_repair_attempts=0))

    result = CogAlphaState.model_validate(node(state))

    assert result.candidates == []
    assert result.rejected_pool[0].stage == CandidateStage.REJECTED_BY_QUALITY
    assert invoker.calls == []


def test_quality_pipeline_rejects_one_candidate_when_quality_skill_fails():
    code = '''
def factor_close_open_gap(df):
    """Close-open gap."""
    df_copy = df.copy()
    df_copy["factor_close_open_gap"] = df_copy["close"] - df_copy["open"]
    return df_copy["factor_close_open_gap"]
'''
    state = CogAlphaState(candidates=[make_candidate(code)]).model_dump(mode="python")
    invoker = FailingQualityInvoker()
    node = QualityPipelineNode(invoker=invoker, config=MVPLoopConfig())

    result = CogAlphaState.model_validate(node(state))

    assert result.candidates == []
    assert result.rejected_pool[0].stage == CandidateStage.REJECTED_BY_QUALITY
    assert result.node_history[-1].metadata == {"accepted": 0, "rejected": 1}
    assert result.node_history[-1].guard_reports[-1].guard_name == "quality_pipeline_exception"
    assert result.node_history[-1].guard_reports[-1].issues[0].code == "quality_skill_error"
