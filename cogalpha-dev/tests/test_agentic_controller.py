from collections.abc import Sequence

import pytest

from cogalpha.config import MVPLoopConfig
from cogalpha.harness import ToolCall, ToolRegistry, ToolSpec
from cogalpha.harness.agentic import (
    AgenticController,
    AgenticControllerConfig,
    AgenticDecisionClient,
    AgenticDecisionValidationError,
)
from cogalpha.harness.cogalpha_tools import (
    COGALPHA_STATE_KEY,
    CogAlphaRuntime,
    build_cogalpha_tools,
)
from cogalpha.harness.loop import AgentDecision, AgentLoopState, run_agent_loop
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaCandidateBatch,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    FitnessMetrics,
    QualityDecision,
    QualityVerdict,
    SkillKind,
    SkillRef,
)
from cogalpha.tracing import TraceEventKind


class FakeDecisionClient(AgenticDecisionClient):
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions
        self.states: list[AgentLoopState] = []

    def decide(
        self,
        state: AgentLoopState,
        *,
        tools: list[ToolSpec],
    ) -> AgentDecision:
        self.states.append(state)
        assert tools
        return self.decisions.pop(0)


class MemoryLedger:
    def __init__(self) -> None:
        self.events: list[tuple[TraceEventKind, dict]] = []

    def record(self, kind: TraceEventKind | str, *, payload: dict | None = None):
        self.events.append((TraceEventKind(kind), payload or {}))


def test_agentic_controller_accepts_registered_tool_and_records_trace():
    registry = _registry_with_review_tool()
    ledger = MemoryLedger()
    client = FakeDecisionClient(
        [
            AgentDecision(
                content="review the generated candidate",
                tool_calls=[ToolCall(name="quality_pipeline.review", arguments={"limit": 1})],
            )
        ]
    )
    controller = AgenticController(
        client=client,
        tools=registry,
        trace_ledger=ledger,
    )

    decision = controller.decide(AgentLoopState(messages=[], context={"budget": 1}))

    assert decision.tool_calls == [
        ToolCall(name="quality_pipeline.review", arguments={"limit": 1})
    ]
    assert ledger.events == [
        (
            TraceEventKind.AGENT_DECISION,
            {
                "content": "review the generated candidate",
                "tool_calls": [
                    {"name": "quality_pipeline.review", "arguments": {"limit": 1}}
                ],
            },
        )
    ]


def test_agentic_controller_rejects_unknown_tool_before_dispatch():
    registry = _registry_with_review_tool()
    client = FakeDecisionClient(
        [AgentDecision(tool_calls=[ToolCall(name="unknown.tool", arguments={})])]
    )
    controller = AgenticController(client=client, tools=registry)

    with pytest.raises(AgenticDecisionValidationError, match="Unknown tool: unknown.tool"):
        controller.decide(AgentLoopState(messages=[], context={}))


def test_agentic_controller_retries_invalid_decisions_until_limit():
    registry = _registry_with_review_tool()
    client = FakeDecisionClient(
        [
            AgentDecision(tool_calls=[ToolCall(name="unknown.tool", arguments={})]),
            AgentDecision(tool_calls=[ToolCall(name="unknown.tool", arguments={})]),
        ]
    )
    controller = AgenticController(
        client=client,
        tools=registry,
        config=AgenticControllerConfig(max_invalid_attempts=2),
    )

    with pytest.raises(AgenticDecisionValidationError, match="max_invalid_attempts=2"):
        controller.decide(AgentLoopState(messages=[], context={}))

    assert len(client.states) == 2


def test_agentic_controller_requires_stop_reason():
    registry = _registry_with_review_tool()
    client = FakeDecisionClient([AgentDecision(content="done")])
    controller = AgenticController(client=client, tools=registry)

    with pytest.raises(AgenticDecisionValidationError, match="stop_reason is required"):
        controller.decide(AgentLoopState(messages=[], context={}))


def test_agentic_controller_drives_real_cogalpha_tools_and_traces_stop_decision():
    from cogalpha.harness import AgenticController as ExportedAgenticController

    runtime = CogAlphaRuntime(
        invoker=FakeCogAlphaInvoker(),
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
        metrics_provider=AcceptingMetricsProvider(),
    )
    tools = build_cogalpha_tools(runtime)
    ledger = MemoryLedger()
    controller = ExportedAgenticController(
        client=FakeDecisionClient(
            [
                AgentDecision(tool_calls=[ToolCall(name="domain_agents.generate")]),
                AgentDecision(tool_calls=[ToolCall(name="quality_pipeline.review")]),
                AgentDecision(tool_calls=[ToolCall(name="fitness_gate.evaluate")]),
                AgentDecision(content="stop", stop_reason="generation_limit"),
            ]
        ),
        tools=tools,
        trace_ledger=ledger,
    )
    context = {COGALPHA_STATE_KEY: CogAlphaState().model_dump(mode="python")}

    run_agent_loop(
        adapter=controller,
        tools=tools,
        messages=[],
        context=context,
        max_turns=4,
    )

    state = CogAlphaState.model_validate(context[COGALPHA_STATE_KEY])
    assert [entry.node_name for entry in state.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert state.candidates == []
    assert {candidate.stage for candidate in state.qualified_pool} == {CandidateStage.ELITE}
    assert [kind for kind, _payload in ledger.events].count(TraceEventKind.AGENT_DECISION) == 3
    assert [kind for kind, _payload in ledger.events].count(TraceEventKind.STOP_DECISION) == 1
    assert ledger.events[-1] == (
        TraceEventKind.STOP_DECISION,
        {"content": "stop", "reason": "generation_limit"},
    )


def _registry_with_review_tool() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="quality_pipeline.review",
            description="Review alpha candidates.",
            input_schema={"type": "object"},
        ),
        lambda _call, _context: {"accepted": []},
    )
    return registry


def _make_candidate(candidate_id: str) -> AlphaCandidate:
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
    )


class FakeCogAlphaInvoker:
    def invoke(self, skill_name, runtime_payload, output_schema):
        assert runtime_payload
        if output_schema is AlphaCandidateBatch:
            return AlphaCandidateBatch(candidates=[_make_candidate(f"{skill_name}-candidate")])

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

        if output_schema is AlphaCandidate:
            return _make_candidate(f"{skill_name}-child")

        raise AssertionError(f"Unexpected output schema: {output_schema}")


class AcceptingMetricsProvider:
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
