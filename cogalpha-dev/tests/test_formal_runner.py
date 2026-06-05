import os

import pytest

from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateEvaluationResult,
    CandidateStage,
    CogAlphaState,
    DAGNodeResult,
    FitnessMetrics,
)
from cogalpha.verification.trace_verifier import verify_cogalpha_trace
from scripts.run_formal_mvp import (
    _assert_formal_run_complete,
    _build_formal_run_report,
    _configure_llm_provider,
    _load_key_file,
)


def test_key_file_key_alias_populates_cogalpha_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("COGALPHA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_MODEL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    key_file = tmp_path / "KEY.md"
    key_file.write_text("key=secret-value\n", encoding="utf-8")

    _load_key_file(str(key_file))

    assert os.environ["COGALPHA_LLM_API_KEY"] == "secret-value"
    assert "COGALPHA_LLM_MODEL" not in os.environ


def test_key_file_does_not_override_explicit_llm_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("COGALPHA_LLM_API_KEY", "existing-key")
    monkeypatch.setenv("COGALPHA_LLM_MODEL", "existing-model")
    key_file = tmp_path / "KEY.md"
    key_file.write_text("key=other-key\nmodel=other-model\n", encoding="utf-8")

    _load_key_file(str(key_file))

    assert os.environ["COGALPHA_LLM_API_KEY"] == "existing-key"
    assert os.environ["COGALPHA_LLM_MODEL"] == "existing-model"


def test_key_file_custom_base_url_requires_explicit_model(tmp_path, monkeypatch):
    monkeypatch.delenv("COGALPHA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_MODEL", raising=False)
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    key_file = tmp_path / "KEY.md"
    key_file.write_text("key=secret-value\nbase_url=https://example.invalid/v1\n", encoding="utf-8")

    _load_key_file(str(key_file))

    assert os.environ["COGALPHA_LLM_API_KEY"] == "secret-value"
    assert os.environ.get("COGALPHA_LLM_MODEL") is None


def test_deepseek_provider_defaults_to_v4_pro_high(monkeypatch):
    monkeypatch.delenv("COGALPHA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_MODEL", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_THINKING", raising=False)
    monkeypatch.delenv("COGALPHA_LLM_MAX_TOKENS", raising=False)
    args = __import__("argparse").Namespace(
        provider="deepseek",
        model=None,
        base_url=None,
        reasoning_effort=None,
        thinking=None,
        max_tokens=8192,
    )

    _configure_llm_provider(args)

    assert os.environ["COGALPHA_LLM_BASE_URL"] == "https://api.deepseek.com"
    assert os.environ["COGALPHA_LLM_MODEL"] == "deepseek-v4-pro"
    assert os.environ["COGALPHA_LLM_REASONING_EFFORT"] == "high"
    assert os.environ["COGALPHA_LLM_THINKING"] == "enabled"
    assert os.environ["COGALPHA_LLM_MAX_TOKENS"] == "8192"


def test_formal_run_invariant_accepts_fitness_terminal_state():
    candidate = _candidate("alpha-1", CandidateStage.ELITE)
    state = CogAlphaState(
        candidates=[],
        qualified_pool=[candidate],
        elite_pool=[candidate],
        node_history=[
            DAGNodeResult(node_name="domain_agents"),
            DAGNodeResult(
                node_name="fitness_gate",
                evaluation_results=[
                    CandidateEvaluationResult(
                        candidate_id="alpha-1",
                        metrics=FitnessMetrics(
                            ic=0.02,
                            rank_ic=0.02,
                            icir=0.2,
                            rank_icir=0.2,
                            mi=0.03,
                        ),
                    )
                ],
            ),
        ],
    )

    _assert_formal_run_complete(state)


def test_formal_run_complete_state_can_still_fail_trace_verification():
    candidate = _candidate("alpha-1", CandidateStage.ELITE)
    state = CogAlphaState(
        candidates=[],
        qualified_pool=[candidate],
        elite_pool=[candidate],
        node_history=[
            DAGNodeResult(node_name="domain_agents"),
            DAGNodeResult(
                node_name="fitness_gate",
                evaluation_results=[
                    CandidateEvaluationResult(
                        candidate_id="alpha-1",
                        metrics=FitnessMetrics(
                            ic=0.02,
                            rank_ic=0.02,
                            icir=0.2,
                            rank_icir=0.2,
                            mi=0.03,
                        ),
                    )
                ],
            ),
        ],
    )

    _assert_formal_run_complete(state)
    trace_report = verify_cogalpha_trace(state, [])

    assert not trace_report.passed
    assert "missing_tool_result" in {finding.code for finding in trace_report.findings}


def test_formal_run_invariant_rejects_trailing_unevaluated_candidates():
    state = CogAlphaState(
        candidates=[_candidate("alpha-1", CandidateStage.ACCEPTED_BY_QUALITY)],
        node_history=[DAGNodeResult(node_name="thinking_evolution")],
    )

    with pytest.raises(RuntimeError, match="expected final fitness_gate"):
        _assert_formal_run_complete(state)


def test_formal_run_invariant_rejects_final_pool_without_evaluation():
    candidate = _candidate("alpha-1", CandidateStage.ELITE)
    state = CogAlphaState(
        candidates=[],
        qualified_pool=[candidate],
        node_history=[DAGNodeResult(node_name="fitness_gate")],
    )

    with pytest.raises(RuntimeError, match="without evaluation results"):
        _assert_formal_run_complete(state)


def test_formal_run_invariant_allows_quality_rejected_without_evaluation():
    state = CogAlphaState(
        candidates=[],
        rejected_pool=[_candidate("alpha-1", CandidateStage.REJECTED_BY_QUALITY)],
        node_history=[DAGNodeResult(node_name="fitness_gate")],
    )

    _assert_formal_run_complete(state)


def test_formal_run_report_holds_when_no_candidate_qualifies(tmp_path):
    summary = {
        "run_id": "smoke",
        "split": "valid",
        "node_history": ["domain_agents", "quality_pipeline", "fitness_gate"],
        "skill_errors": 0,
        "remaining_candidates": 0,
        "qualified": 0,
        "elite": 0,
        "rejected": 1,
    }

    report = _build_formal_run_report(
        summary=summary,
        data_version="data-v1",
        manifest_path=tmp_path / "run_manifest.json",
    )

    assert report.decision == "hold"
    assert report.layers[1].status == "go"
    assert report.layers[2].status == "hold"


def _candidate(candidate_id: str, stage: CandidateStage) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id.replace('-', '_')}",
            code="def factor_alpha_1(df):\n    return df['close'] - df['open']\n",
            rationale="Test alpha.",
        ),
        stage=stage,
    )
