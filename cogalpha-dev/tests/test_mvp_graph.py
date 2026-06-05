import builtins
from collections.abc import Sequence

from cogalpha.config import MVPLoopConfig
from cogalpha.graph import build_mvp_graph
from cogalpha.guards.pipeline import DeterministicGuardOutcome
from cogalpha.registry import DOMAIN_AGENT_SPECS
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaCandidateBatch,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    FitnessMetrics,
    GuardReport,
    GuardStatus,
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


class RejectingGuardPipeline:
    def __init__(self) -> None:
        self.checked: list[str] = []

    def run(self, candidate: AlphaCandidate) -> DeterministicGuardOutcome:
        self.checked.append(candidate.candidate_id)
        return DeterministicGuardOutcome(
            reports=[
                GuardReport(
                    guard_name="test_runtime_guard",
                    status=GuardStatus.FAIL,
                    issues=[],
                )
            ]
        )


def test_mvp_graph_runs_with_skill_invoker_and_metrics_provider():
    invoker = FakeMVPInvoker()
    metrics_provider = FakeMetricsProvider()
    config = MVPLoopConfig(max_generations=1, parent_pool_size=2)
    graph = build_mvp_graph(
        invoker=invoker,
        config=config,
        metrics_provider=metrics_provider,
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert result.generation == 0
    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert len(metrics_provider.evaluated_batches) == 1
    assert len(metrics_provider.evaluated_batches[0]) == len(DOMAIN_AGENT_SPECS)
    assert len(result.elite_pool) == len(DOMAIN_AGENT_SPECS)
    assert {candidate.stage for candidate in result.qualified_pool} == {CandidateStage.ELITE}
    assert result.candidates == []


def test_invoker_backed_mvp_graph_does_not_require_langgraph(monkeypatch):
    real_import = builtins.__import__

    def blocked_langgraph_import(name, *args, **kwargs):
        if name == "langgraph.graph":
            raise ModuleNotFoundError("blocked langgraph")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_langgraph_import)

    graph = build_mvp_graph(
        invoker=FakeMVPInvoker(),
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
        metrics_provider=FakeMetricsProvider(),
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]


def test_invoker_backed_mvp_graph_honors_agent_specs():
    metrics_provider = FakeMetricsProvider()
    graph = build_mvp_graph(
        invoker=FakeMVPInvoker(),
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2),
        metrics_provider=metrics_provider,
        agent_specs=DOMAIN_AGENT_SPECS[:1],
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert len(result.node_history[0].candidates) == 1
    assert len(metrics_provider.evaluated_batches[0]) == 1


def test_invoker_backed_mvp_graph_honors_guard_pipeline():
    guard_pipeline = RejectingGuardPipeline()
    graph = build_mvp_graph(
        invoker=FakeMVPInvoker(),
        config=MVPLoopConfig(max_generations=1, parent_pool_size=2, max_repair_attempts=0),
        metrics_provider=FakeMetricsProvider(),
        guard_pipeline=guard_pipeline,
        agent_specs=DOMAIN_AGENT_SPECS[:1],
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert guard_pipeline.checked == ["alpha-market-cycle-candidate"]
    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert result.qualified_pool == []
    assert result.elite_pool == []


def test_mvp_graph_evaluates_evolved_children_before_ending():
    invoker = FakeMVPInvoker()
    metrics_provider = FakeMetricsProvider()
    config = MVPLoopConfig(max_generations=2, parent_pool_size=2)
    graph = build_mvp_graph(
        invoker=invoker,
        config=config,
        metrics_provider=metrics_provider,
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
        "thinking_evolution",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert result.generation == 1
    assert len(metrics_provider.evaluated_batches) == 2
    assert len(metrics_provider.evaluated_batches[1]) == 4
    assert result.candidates == []


def test_mvp_graph_max_generations_counts_evaluated_generation_indices():
    invoker = FakeMVPInvoker()
    metrics_provider = FakeMetricsProvider()
    config = MVPLoopConfig(max_generations=3, parent_pool_size=2)
    graph = build_mvp_graph(
        invoker=invoker,
        config=config,
        metrics_provider=metrics_provider,
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
        "thinking_evolution",
        "quality_pipeline",
        "fitness_gate",
        "thinking_evolution",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert result.generation == 2
    assert len(metrics_provider.evaluated_batches) == 3
    assert result.node_history[-1].node_name == "fitness_gate"
    assert result.candidates == []


def test_mvp_graph_early_stops_without_qualified_candidates():
    invoker = FakeMVPInvoker()
    metrics_provider = RejectingMetricsProvider()
    config = MVPLoopConfig(max_generations=3, parent_pool_size=2)
    graph = build_mvp_graph(
        invoker=invoker,
        config=config,
        metrics_provider=metrics_provider,
    )

    result = CogAlphaState.model_validate(graph.invoke(CogAlphaState().model_dump(mode="python")))

    assert [entry.node_name for entry in result.node_history] == [
        "domain_agents",
        "quality_pipeline",
        "fitness_gate",
    ]
    assert result.generation == 0
    assert len(metrics_provider.evaluated_batches) == 1
    assert result.candidates == []
