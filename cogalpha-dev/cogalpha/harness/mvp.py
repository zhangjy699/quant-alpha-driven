from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cogalpha.config import MVPLoopConfig
from cogalpha.harness.cogalpha_tools import (
    COGALPHA_STATE_KEY,
    CogAlphaRuntime,
    build_cogalpha_tools,
)
from cogalpha.harness.loop import AgentDecision, AgentLoopState, run_agent_loop
from cogalpha.harness.tools import ToolCall
from cogalpha.schemas import CogAlphaState


@dataclass(frozen=True)
class MVPDecisionAdapter:
    config: MVPLoopConfig

    def decide(self, state: AgentLoopState) -> AgentDecision:
        cogalpha_state = CogAlphaState.model_validate(state.context[COGALPHA_STATE_KEY])
        next_tool = _next_mvp_tool(cogalpha_state, self.config)
        if next_tool is None:
            return AgentDecision()
        return AgentDecision(tool_calls=[ToolCall(name=next_tool, arguments={})])


@dataclass(frozen=True)
class MVPHarnessGraph:
    runtime: CogAlphaRuntime

    def invoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        return run_mvp_harness(initial_state, runtime=self.runtime)


def run_mvp_harness(
    initial_state: dict[str, Any],
    *,
    invoker: Any | None = None,
    config: MVPLoopConfig | None = None,
    metrics_provider: Any | None = None,
    runtime: CogAlphaRuntime | None = None,
) -> dict[str, Any]:
    if runtime is None:
        if invoker is None:
            raise ValueError("invoker is required when runtime is not provided")
        runtime = CogAlphaRuntime(
            invoker=invoker,
            config=config or MVPLoopConfig(),
            metrics_provider=metrics_provider,
        )

    starting_state = CogAlphaState.model_validate(initial_state).model_dump(mode="python")
    context = {COGALPHA_STATE_KEY: starting_state}
    run_agent_loop(
        adapter=MVPDecisionAdapter(runtime.config),
        tools=build_cogalpha_tools(runtime),
        messages=[],
        context=context,
        max_turns=max(1, runtime.config.max_generations * 3 + 1),
    )
    return CogAlphaState.model_validate(context[COGALPHA_STATE_KEY]).model_dump(mode="python")


def _next_mvp_tool(state: CogAlphaState, config: MVPLoopConfig) -> str | None:
    if not state.node_history:
        return "domain_agents.generate"

    last_node = state.node_history[-1].node_name
    if last_node == "domain_agents":
        return "quality_pipeline.review"
    if last_node == "quality_pipeline":
        return "fitness_gate.evaluate"
    if last_node == "fitness_gate":
        if state.generation >= config.max_generations - 1:
            return None
        if not state.qualified_pool:
            return None
        return "thinking_evolution.generate_children"
    if last_node == "thinking_evolution":
        return "quality_pipeline.review"
    return None
