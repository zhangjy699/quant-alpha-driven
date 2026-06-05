from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cogalpha.config import MVPLoopConfig
from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.harness.tools import ToolRegistry, ToolSpec
from cogalpha.nodes import DomainAgentNode, EvolutionNode, FitnessGateNode, QualityPipelineNode
from cogalpha.registry import DOMAIN_AGENT_SPECS, DomainAgentSpec
from cogalpha.schemas import CogAlphaState
from cogalpha.skill_nodes import StructuredArtifactInvoker

COGALPHA_STATE_KEY = "cogalpha_state"


@dataclass(frozen=True)
class CogAlphaRuntime:
    invoker: StructuredArtifactInvoker
    config: MVPLoopConfig
    metrics_provider: Any | None = None
    guard_pipeline: DeterministicGuardPipeline | None = None
    agent_specs: tuple[DomainAgentSpec, ...] = DOMAIN_AGENT_SPECS


def build_cogalpha_tools(runtime: CogAlphaRuntime) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="domain_agents.generate",
            description="Generate candidates from the configured domain agent skills.",
            input_schema={"type": "object"},
        ),
        lambda _call, context: _run_node(
            DomainAgentNode(
                invoker=runtime.invoker,
                config=runtime.config,
                agent_specs=runtime.agent_specs,
            ),
            context,
        ),
    )
    registry.register(
        ToolSpec(
            name="quality_pipeline.review",
            description="Run deterministic guards and quality checker skills.",
            input_schema={"type": "object"},
        ),
        lambda _call, context: _run_node(_quality_node(runtime), context),
    )
    registry.register(
        ToolSpec(
            name="fitness_gate.evaluate",
            description="Evaluate candidates and update elite and qualified pools.",
            input_schema={"type": "object"},
        ),
        lambda _call, context: _run_node(
            FitnessGateNode(runtime.config, runtime.metrics_provider),
            context,
        ),
    )
    registry.register(
        ToolSpec(
            name="thinking_evolution.generate_children",
            description="Generate child candidates from the qualified parent pool.",
            input_schema={"type": "object"},
        ),
        lambda _call, context: _run_node(EvolutionNode(runtime.invoker, runtime.config), context),
    )
    return registry


def _quality_node(runtime: CogAlphaRuntime) -> QualityPipelineNode:
    if runtime.guard_pipeline is None:
        return QualityPipelineNode(runtime.invoker, runtime.config)
    return QualityPipelineNode(
        runtime.invoker,
        runtime.config,
        guard_pipeline=runtime.guard_pipeline,
    )


def _run_node(
    node: Callable[[dict[str, Any]], dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    state_payload = context.get(COGALPHA_STATE_KEY)
    if state_payload is None:
        raise KeyError(f"Missing context key: {COGALPHA_STATE_KEY}")

    state = CogAlphaState.model_validate(state_payload)
    updated = CogAlphaState.model_validate(node(state.model_dump(mode="python")))
    serialized = updated.model_dump(mode="python")
    context[COGALPHA_STATE_KEY] = serialized
    return serialized
