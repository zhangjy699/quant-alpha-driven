from collections.abc import Sequence

from cogalpha.config import MVPLoopConfig
from cogalpha.graph import build_mvp_graph
from cogalpha.harness.mvp import run_mvp_harness
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


class FakeMVPInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.child_counter = 0

    def invoke(self, skill_name, runtime_payload, output_schema):
        assert runtime_payload
        self.calls.append(skill_name)

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
                practical_soundness="The candidate is coherent enough for test fitness.",
                feedback="No repair needed.",
            )

        if output_schema is AlphaCandidate:
            self.child_counter += 1
            return make_candidate(f"child-{self.child_counter}")

        raise AssertionError(f"Unexpected output schema: {output_schema}")


class FakeMetricsProvider:
    def __init__(self) -> None:
        self.evaluated_batches: list[list[str]] = []

    def evaluate(self, candidates: Sequence[AlphaCandidate]):
        self.evaluated_batches.append([candidate.candidate_id for candidate in candidates])
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


class RejectingMetricsProvider:
    def __init__(self) -> None:
        self.evaluated_batches: list[list[str]] = []

    def evaluate(self, candidates: Sequence[AlphaCandidate]):
        self.evaluated_batches.append([candidate.candidate_id for candidate in candidates])
        return {
            candidate.candidate_id: FitnessMetrics(
                ic=-0.01,
                rank_ic=-0.01,
                icir=-0.1,
                rank_icir=-0.1,
                mi=0.0,
            )
            for candidate in candidates
        }


def test_mvp_harness_matches_legacy_graph_for_two_generations():
    initial_state = CogAlphaState().model_dump(mode="python")
    config = MVPLoopConfig(max_generations=2, parent_pool_size=2)
    harness_metrics = FakeMetricsProvider()
    legacy_metrics = FakeMetricsProvider()

    harness_result = CogAlphaState.model_validate(
        run_mvp_harness(
            initial_state,
            invoker=FakeMVPInvoker(),
            config=config,
            metrics_provider=harness_metrics,
        )
    )
    legacy_result = CogAlphaState.model_validate(
        build_mvp_graph(
            invoker=FakeMVPInvoker(),
            config=config,
            metrics_provider=legacy_metrics,
        ).invoke(initial_state)
    )

    assert [entry.node_name for entry in harness_result.node_history] == [
        entry.node_name for entry in legacy_result.node_history
    ]
    assert harness_result.generation == legacy_result.generation == 1
    assert len(harness_metrics.evaluated_batches) == len(legacy_metrics.evaluated_batches) == 2
    assert len(harness_metrics.evaluated_batches[1]) == 4
    assert harness_result.candidates == legacy_result.candidates == []


def test_mvp_harness_stops_without_qualified_candidates():
    metrics_provider = RejectingMetricsProvider()
    result = CogAlphaState.model_validate(
        run_mvp_harness(
            CogAlphaState().model_dump(mode="python"),
            invoker=FakeMVPInvoker(),
            config=MVPLoopConfig(max_generations=3, parent_pool_size=2),
            metrics_provider=metrics_provider,
        )
    )

    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert result.generation == 0
    assert len(metrics_provider.evaluated_batches) == 1
    assert len(result.elite_pool) == 0
    assert {candidate.stage for candidate in result.rejected_pool} == {
        CandidateStage.REJECTED_BY_FITNESS
    }
    assert result.candidates == []
    assert len(DOMAIN_AGENT_SPECS) > 0
