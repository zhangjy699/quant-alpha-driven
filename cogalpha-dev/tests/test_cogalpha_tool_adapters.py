from collections.abc import Sequence

from cogalpha.config import MVPLoopConfig
from cogalpha.harness import ToolCall
from cogalpha.harness.cogalpha_tools import (
    COGALPHA_STATE_KEY,
    CogAlphaRuntime,
    build_cogalpha_tools,
)
from cogalpha.registry import DOMAIN_AGENT_SPECS
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


def make_candidate(candidate_id: str) -> AlphaCandidate:
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


class FakeInvoker:
    def invoke(self, skill_name, runtime_payload, output_schema):
        assert runtime_payload
        if output_schema is AlphaCandidateBatch:
            return AlphaCandidateBatch(candidates=[make_candidate(f"{skill_name}-candidate")])

        if output_schema is QualityDecision:
            return QualityDecision(
                skill=SkillRef(
                    name=skill_name,
                    path=f"skills/{skill_name}/SKILL.md",
                    kind=SkillKind.QUALITY_CHECKER,
                ),
                verdict=QualityVerdict.ACCEPT,
                practical_soundness="Coherent enough for harness tests.",
                feedback="No repair needed.",
            )

        if output_schema is AlphaCandidate:
            return make_candidate(f"{skill_name}-child")

        raise AssertionError(f"Unexpected output schema: {output_schema}")


class FakeMetricsProvider:
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


def test_cogalpha_tools_run_domain_quality_and_fitness_nodes():
    registry = build_cogalpha_tools(
        CogAlphaRuntime(
            invoker=FakeInvoker(),
            config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
            metrics_provider=FakeMetricsProvider(),
        )
    )
    context = {COGALPHA_STATE_KEY: CogAlphaState().model_dump(mode="python")}

    for tool_name in [
        "domain_agents.generate",
        "quality_pipeline.review",
        "fitness_gate.evaluate",
    ]:
        result = registry.dispatch(ToolCall(name=tool_name, arguments={}), context=context)
        assert result.success is True

    state = CogAlphaState.model_validate(context[COGALPHA_STATE_KEY])

    assert [entry.node_name for entry in state.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert len(state.node_history[0].candidates) == len(DOMAIN_AGENT_SPECS)
    assert len(state.elite_pool) == len(DOMAIN_AGENT_SPECS)
    assert {candidate.stage for candidate in state.qualified_pool} == {CandidateStage.ELITE}
    assert state.candidates == []


def test_cogalpha_tool_registry_contains_expected_tool_names():
    registry = build_cogalpha_tools(
        CogAlphaRuntime(
            invoker=FakeInvoker(),
            config=MVPLoopConfig(),
            metrics_provider=FakeMetricsProvider(),
        )
    )

    assert set(registry.specs) == {
        "domain_agents.generate",
        "quality_pipeline.review",
        "fitness_gate.evaluate",
        "thinking_evolution.generate_children",
    }
