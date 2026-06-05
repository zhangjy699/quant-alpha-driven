from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cogalpha.harness.loop import AgentDecision, AgentLoopState
from cogalpha.harness.tools import ToolCall, ToolRegistry, ToolSpec
from cogalpha.tracing import TraceEventKind, TraceLedger


class AgenticDecisionValidationError(ValueError):
    """Raised when a model decision is not safe to dispatch."""


class AgenticDecisionClient(Protocol):
    def decide(
        self,
        state: AgentLoopState,
        *,
        tools: list[ToolSpec],
    ) -> AgentDecision:
        ...


@dataclass(frozen=True)
class AgenticControllerConfig:
    max_invalid_attempts: int = 1

    def __post_init__(self) -> None:
        if self.max_invalid_attempts < 1:
            msg = "max_invalid_attempts must be at least 1"
            raise ValueError(msg)


@dataclass
class AgenticController:
    client: AgenticDecisionClient
    tools: ToolRegistry
    config: AgenticControllerConfig = AgenticControllerConfig()
    trace_ledger: TraceLedger | None = None

    def decide(self, state: AgentLoopState) -> AgentDecision:
        last_error: AgenticDecisionValidationError | None = None
        for _attempt in range(self.config.max_invalid_attempts):
            decision = self.client.decide(state, tools=list(self.tools.specs.values()))
            try:
                self._validate(decision)
            except AgenticDecisionValidationError as exc:
                last_error = exc
                continue

            self._record_decision(decision)
            return decision

        detail = str(last_error) if last_error is not None else "invalid decision"
        msg = (
            f"Agentic decision validation failed after "
            f"max_invalid_attempts={self.config.max_invalid_attempts}: {detail}"
        )
        raise AgenticDecisionValidationError(msg)

    def _validate(self, decision: AgentDecision) -> None:
        if decision.tool_calls:
            for call in decision.tool_calls:
                if call.name not in self.tools.specs:
                    msg = f"Unknown tool: {call.name}"
                    raise AgenticDecisionValidationError(msg)
            return

        if not decision.stop_reason or not decision.stop_reason.strip():
            msg = "stop_reason is required when no tool_calls are returned"
            raise AgenticDecisionValidationError(msg)

    def _record_decision(self, decision: AgentDecision) -> None:
        if self.trace_ledger is None:
            return

        if decision.tool_calls:
            self.trace_ledger.record(
                TraceEventKind.AGENT_DECISION,
                payload={
                    "content": decision.content,
                    "tool_calls": [_tool_call_payload(call) for call in decision.tool_calls],
                },
            )
            return

        self.trace_ledger.record(
            TraceEventKind.STOP_DECISION,
            payload={
                "content": decision.content,
                "reason": decision.stop_reason,
            },
        )


def _tool_call_payload(call: ToolCall) -> dict[str, object]:
    return {"name": call.name, "arguments": dict(call.arguments)}
