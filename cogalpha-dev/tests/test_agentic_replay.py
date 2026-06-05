import json

from cogalpha.reporting import replay_agentic_run
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    DAGNodeResult,
)
from cogalpha.tracing import TraceEventKind, TraceLedger


def test_replay_agentic_run_reads_final_state_and_trace_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_state = CogAlphaState(
        candidates=[],
        qualified_pool=[_candidate("alpha-1", CandidateStage.QUALIFIED)],
        node_history=[
            DAGNodeResult(node_name="domain_agents"),
            DAGNodeResult(node_name="quality_pipeline"),
            DAGNodeResult(node_name="fitness_gate"),
        ],
    )
    (run_dir / "final_state.json").write_text(
        final_state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    ledger = TraceLedger(run_id="replay-test", path=run_dir / "trace.jsonl")
    ledger.record(TraceEventKind.RUN_STARTED)
    ledger.record(
        TraceEventKind.TOOL_CALL_FINISHED,
        payload={"tool_name": "domain_agents", "status": "ok"},
    )

    report = replay_agentic_run(run_dir)

    assert not report.passed
    assert report.trace_event_count == 2
    assert "missing_tool_result" in {finding.code for finding in report.findings}


def test_replay_agentic_run_rejects_static_summary_without_trace_events(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_state = CogAlphaState(
        candidates=[],
        qualified_pool=[_candidate("alpha-1", CandidateStage.QUALIFIED)],
        node_history=[DAGNodeResult(node_name="fitness_gate")],
    )
    (run_dir / "final_state.json").write_text(
        final_state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"node_history": ["fitness_gate"]}))

    report = replay_agentic_run(run_dir)

    assert not report.passed
    assert report.trace_event_count == 0
    assert "missing_tool_result" in {finding.code for finding in report.findings}


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
