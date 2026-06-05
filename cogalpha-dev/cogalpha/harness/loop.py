from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cogalpha.harness.tools import ToolCall, ToolRegistry, ToolResult


@dataclass(frozen=True)
class AgentMessage:
    role: str
    content: str = ""
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class AgentDecision:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None


@dataclass
class AgentLoopState:
    messages: list[AgentMessage]
    context: dict[str, Any]
    turns: int = 0


class DecisionAdapter(Protocol):
    def decide(self, state: AgentLoopState) -> AgentDecision:
        ...


def run_agent_loop(
    *,
    adapter: DecisionAdapter,
    tools: ToolRegistry,
    messages: list[AgentMessage],
    context: dict[str, Any],
    max_turns: int = 16,
    fail_fast_tools: bool = False,
) -> AgentLoopState:
    state = AgentLoopState(messages=list(messages), context=context)

    for _ in range(max_turns):
        decision = adapter.decide(state)
        state.turns += 1

        if not decision.tool_calls:
            if decision.content:
                state.messages.append(AgentMessage(role="assistant", content=decision.content))
            return state

        results = tools.dispatch_all(
            decision.tool_calls,
            context=state.context,
            fail_fast=fail_fast_tools,
        )
        state.messages.extend(
            AgentMessage(
                role="tool",
                content=result.error or "" if not result.success else "",
                tool_result=result,
            )
            for result in results
        )

    raise RuntimeError(f"Agent loop exceeded max_turns={max_turns}")
