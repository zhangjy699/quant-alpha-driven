import pytest

from cogalpha.harness import ToolCall, ToolRegistry, ToolSpec
from cogalpha.harness.loop import (
    AgentDecision,
    AgentLoopState,
    AgentMessage,
    run_agent_loop,
)


class ScriptedDecisionAdapter:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions
        self.states: list[AgentLoopState] = []

    def decide(self, state: AgentLoopState) -> AgentDecision:
        self.states.append(state)
        if not self.decisions:
            return AgentDecision()
        return self.decisions.pop(0)


def test_agent_loop_dispatches_tool_results_until_stop():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="math.add", description="Add one.", input_schema={"type": "object"}),
        lambda call, _context: call.arguments["value"] + 1,
    )
    adapter = ScriptedDecisionAdapter(
        [
            AgentDecision(tool_calls=[ToolCall(name="math.add", arguments={"value": 2})]),
            AgentDecision(),
        ]
    )

    state = run_agent_loop(
        adapter=adapter,
        tools=registry,
        messages=[AgentMessage(role="user", content="add one")],
        context={},
    )

    assert state.turns == 2
    assert [message.role for message in state.messages] == ["user", "tool"]
    assert state.messages[-1].tool_result is not None
    assert state.messages[-1].tool_result.output == 3
    assert adapter.states[-1].messages[-1].tool_result.output == 3


def test_agent_loop_stops_when_decision_has_no_tool_calls():
    state = run_agent_loop(
        adapter=ScriptedDecisionAdapter([AgentDecision(content="done")]),
        tools=ToolRegistry(),
        messages=[AgentMessage(role="user", content="nothing to do")],
        context={"kept": True},
    )

    assert state.turns == 1
    assert [message.role for message in state.messages] == ["user", "assistant"]
    assert state.messages[-1].content == "done"
    assert state.context == {"kept": True}


def test_agent_loop_raises_when_max_turns_exceeded():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="loop.forever", description="Never stop.", input_schema={"type": "object"}),
        lambda _call, _context: "again",
    )
    adapter = ScriptedDecisionAdapter(
        [
            AgentDecision(tool_calls=[ToolCall(name="loop.forever", arguments={})]),
            AgentDecision(tool_calls=[ToolCall(name="loop.forever", arguments={})]),
        ]
    )

    with pytest.raises(RuntimeError, match="Agent loop exceeded max_turns=1"):
        run_agent_loop(
            adapter=adapter,
            tools=registry,
            messages=[AgentMessage(role="user", content="continue")],
            context={},
            max_turns=1,
        )
