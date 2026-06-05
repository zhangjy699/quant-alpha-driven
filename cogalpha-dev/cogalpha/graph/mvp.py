"""LangGraph skeleton for the small-scale CogAlpha MVP loop."""

from __future__ import annotations

from typing import Any

from cogalpha.config import MVPLoopConfig
from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.harness.cogalpha_tools import CogAlphaRuntime
from cogalpha.harness.mvp import MVPHarnessGraph
from cogalpha.nodes import DomainAgentNode, EvolutionNode, FitnessGateNode, QualityPipelineNode
from cogalpha.nodes.fitness import CandidateMetricsProvider
from cogalpha.registry import DOMAIN_AGENT_SPECS, DomainAgentSpec
from cogalpha.schemas import CogAlphaState, DAGNodeResult
from cogalpha.skill_invocation import SkillInvoker


def generation_node(state: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for the 21 domain-agent skill nodes."""

    parsed = CogAlphaState.model_validate(state)
    parsed.node_history.append(
        DAGNodeResult(
            node_name="generation",
            metadata={"status": "placeholder", "next": "invoke domain agent skills"},
        )
    )
    return parsed.model_dump(mode="python")


def build_generation_node(
    invoker: SkillInvoker,
    config: MVPLoopConfig,
    agent_specs: tuple[DomainAgentSpec, ...] = DOMAIN_AGENT_SPECS,
) -> DomainAgentNode:
    """Build the generation node used by the MVP graph."""

    return DomainAgentNode(invoker=invoker, config=config, agent_specs=agent_specs)


def build_quality_node(
    invoker: SkillInvoker,
    config: MVPLoopConfig,
    guard_pipeline: DeterministicGuardPipeline | None = None,
) -> QualityPipelineNode:
    """Build the quality node used by the MVP graph."""

    if guard_pipeline is None:
        return QualityPipelineNode(invoker=invoker, config=config)
    return QualityPipelineNode(invoker=invoker, config=config, guard_pipeline=guard_pipeline)


def build_fitness_node(
    config: MVPLoopConfig,
    metrics_provider: CandidateMetricsProvider | None = None,
) -> FitnessGateNode:
    """Build the fitness gate node used by the MVP graph."""

    return FitnessGateNode(config=config, metrics_provider=metrics_provider)


def build_evolution_node(invoker: SkillInvoker, config: MVPLoopConfig) -> EvolutionNode:
    """Build the thinking-evolution node used by the MVP graph."""

    return EvolutionNode(invoker=invoker, config=config)


def quality_node(state: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for quality checker skills and deterministic guards."""

    parsed = CogAlphaState.model_validate(state)
    parsed.node_history.append(
        DAGNodeResult(
            node_name="quality",
            metadata={"status": "placeholder", "next": "run guards and quality skills"},
        )
    )
    return parsed.model_dump(mode="python")


def fitness_node(state: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for the paper-defined fitness gate."""

    parsed = CogAlphaState.model_validate(state)
    parsed.node_history.append(
        DAGNodeResult(
            node_name="fitness",
            metadata={"status": "placeholder", "next": "compute IC/RankIC/ICIR/RankICIR/MI"},
        )
    )
    return parsed.model_dump(mode="python")


def evolution_node(state: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for mutation and crossover skill nodes."""

    parsed = CogAlphaState.model_validate(state)
    parsed.generation += 1
    parsed.node_history.append(
        DAGNodeResult(
            node_name="evolution",
            metadata={"status": "placeholder", "next": "invoke mutation and crossover skills"},
        )
    )
    return parsed.model_dump(mode="python")


def build_mvp_graph(
    invoker: SkillInvoker | None = None,
    config: MVPLoopConfig | None = None,
    metrics_provider: CandidateMetricsProvider | None = None,
    guard_pipeline: DeterministicGuardPipeline | None = None,
    agent_specs: tuple[DomainAgentSpec, ...] = DOMAIN_AGENT_SPECS,
):
    """Build the MVP graph.

    The import is intentionally local so schema and guard tests can run before LangGraph is
    installed in a fresh development environment.
    """

    loop_config = config or MVPLoopConfig()
    if invoker is not None:
        return MVPHarnessGraph(
            CogAlphaRuntime(
                invoker=invoker,
                config=loop_config,
                metrics_provider=metrics_provider,
                guard_pipeline=guard_pipeline,
                agent_specs=agent_specs,
            )
        )

    try:
        from langgraph.graph import END, StateGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LangGraph is required to build the MVP graph. Install project dependencies first."
        ) from exc

    graph = StateGraph(dict)
    generation = generation_node
    quality = quality_node
    fitness = fitness_node
    evolution = evolution_node

    graph.add_node("generation", generation)
    graph.add_node("quality", quality)
    graph.add_node("fitness", fitness)
    graph.add_node("evolution", evolution)

    graph.set_entry_point("generation")
    graph.add_edge("generation", "quality")
    graph.add_edge("quality", "fitness")
    graph.add_conditional_edges(
        "fitness",
        lambda state: _route_after_fitness(state, loop_config),
        {"evolution": "evolution", "end": END},
    )
    graph.add_edge("evolution", "quality")
    return graph.compile()


def _route_after_fitness(state: dict[str, Any], config: MVPLoopConfig) -> str:
    parsed = CogAlphaState.model_validate(state)
    if parsed.generation >= config.max_generations - 1:
        return "end"
    if not parsed.qualified_pool:
        return "end"
    return "evolution"
