import json
from collections.abc import Sequence

from cogalpha.config import MVPLoopConfig
from cogalpha.harness import ToolCall, ToolSpec
from cogalpha.harness.agentic import AgenticDecisionClient
from cogalpha.harness.loop import AgentDecision, AgentLoopState
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaCandidateBatch,
    AlphaFunction,
    CandidateStage,
    FitnessMetrics,
    QualityDecision,
    QualityVerdict,
    SkillKind,
    SkillRef,
)
from scripts.run_agentic_mvp import run_agentic_formal_mvp


def test_agentic_formal_runner_writes_trace_and_verification_artifacts(tmp_path):
    output = run_agentic_formal_mvp(
        run_id="agentic-test",
        output_root=tmp_path,
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
        decision_client=FakeDecisionClient(
            [
                AgentDecision(tool_calls=[ToolCall(name="domain_agents.generate")]),
                AgentDecision(tool_calls=[ToolCall(name="quality_pipeline.review")]),
                AgentDecision(tool_calls=[ToolCall(name="fitness_gate.evaluate")]),
                AgentDecision(content="stop", stop_reason="generation_limit"),
            ]
        ),
        skill_invoker=FakeSkillInvoker(),
        metrics_provider=AcceptingMetricsProvider(),
        data_version="fixture-data-v1",
        split="valid",
        agent_limit=1,
    )

    run_dir = output.run_dir
    expected_artifacts = {
        "final_state.json",
        "summary.json",
        "run_manifest.json",
        "evaluation_run_report.json",
        "skill_invocations.jsonl",
        "skill_selection.jsonl",
        "skill_utility.json",
        "trace.jsonl",
        "trace_manifest.json",
        "trace_verification.json",
    }

    assert {path.name for path in run_dir.iterdir()} >= expected_artifacts
    assert output.trace_verification.passed

    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert trace_lines
    assert any('"kind": "tool_call_finished"' in line for line in trace_lines)

    verification = json.loads((run_dir / "trace_verification.json").read_text())
    assert verification["passed"] is True
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["qualified"] == 1
    assert summary["remaining_candidates"] == 0


def test_agentic_formal_runner_persists_skill_utility_and_selection_from_trace(
    tmp_path,
):
    output = run_agentic_formal_mvp(
        run_id="agentic-skill-evidence-test",
        output_root=tmp_path,
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
        decision_client=FakeDecisionClient(
            [
                AgentDecision(tool_calls=[ToolCall(name="domain_agents.generate")]),
                AgentDecision(tool_calls=[ToolCall(name="quality_pipeline.review")]),
                AgentDecision(tool_calls=[ToolCall(name="fitness_gate.evaluate")]),
                AgentDecision(content="stop", stop_reason="generation_limit"),
            ]
        ),
        skill_invoker=FakeSkillInvoker(),
        metrics_provider=AcceptingMetricsProvider(),
        data_version="fixture-data-v1",
        split="valid",
        agent_limit=1,
    )

    run_dir = output.run_dir
    utility_path = run_dir / "skill_utility.json"
    selection_path = run_dir / "skill_selection.jsonl"
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert utility_path.exists()
    assert selection_path.exists()

    linked_events = [
        event
        for event in trace_events
        if event["kind"]
        in {
            "skill_invocation_finished",
            "candidate_stage_changed",
            "fitness_evaluation_recorded",
        }
        and event["payload"].get("skill_name")
        and event["payload"].get("evidence_id")
    ]
    assert {event["kind"] for event in linked_events} >= {
        "skill_invocation_finished",
        "candidate_stage_changed",
        "fitness_evaluation_recorded",
    }
    linked_keys_by_kind = {
        kind: {
            (event["payload"]["skill_name"], event["payload"]["evidence_id"])
            for event in linked_events
            if event["kind"] == kind
        }
        for kind in {
            "skill_invocation_finished",
            "candidate_stage_changed",
            "fitness_evaluation_recorded",
        }
    }
    common_keys = set.intersection(*linked_keys_by_kind.values())
    assert common_keys

    utility_records = json.loads(utility_path.read_text(encoding="utf-8"))
    assert any(
        record["utility"] > 0.0
        and record["usage_count"] > 0
        and record["evidence_ids"]
        for record in utility_records
    )

    selection_records = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
    ]
    assert selection_records
    assert all(
        record["selected_skill"] in record["eligible_skills"]
        for record in selection_records
    )
    assert any(record["evidence_ids"] for record in selection_records)


class FakeDecisionClient(AgenticDecisionClient):
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions

    def decide(
        self,
        state: AgentLoopState,
        *,
        tools: list[ToolSpec],
    ) -> AgentDecision:
        assert state.context
        assert tools
        return self.decisions.pop(0)


class FakeSkillInvoker:
    def invoke(self, skill_name, runtime_payload, output_schema):
        assert runtime_payload
        if output_schema is AlphaCandidateBatch:
            return AlphaCandidateBatch(candidates=[_candidate(f"{skill_name}-candidate")])

        if output_schema is QualityDecision:
            return QualityDecision(
                skill=SkillRef(
                    name=skill_name,
                    path=f"skills/{skill_name}/SKILL.md",
                    kind=SkillKind.QUALITY_CHECKER,
                ),
                verdict=QualityVerdict.ACCEPT,
                practical_soundness="The candidate is coherent enough for fitness.",
                feedback="No repair needed.",
            )

        raise AssertionError(f"Unexpected output schema: {output_schema}")


class AcceptingMetricsProvider:
    cache_hits_by_candidate_id = {}
    errors_by_candidate_id = {}

    def evaluate(self, candidates: Sequence[AlphaCandidate]):
        return {
            candidate.candidate_id: FitnessMetrics(
                ic=0.03,
                rank_ic=0.03,
                icir=0.3,
                rank_icir=0.3,
                mi=0.04,
            )
            for candidate in candidates
        }


def _candidate(candidate_id: str) -> AlphaCandidate:
    function_name = f"factor_{candidate_id.replace('-', '_')}"
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=function_name,
            code=(
                f"def {function_name}(df):\n"
                "    df_copy = df.copy()\n"
                f"    df_copy['{function_name}'] = df_copy['close'] - df_copy['open']\n"
                f"    return df_copy['{function_name}']\n"
            ),
            rationale=f"{candidate_id} rationale.",
        ),
        stage=CandidateStage.GENERATED,
    )
