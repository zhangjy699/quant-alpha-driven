import pytest

from cogalpha.harness.agentic import AgenticController, AgenticDecisionValidationError
from cogalpha.harness.loop import AgentLoopState
from cogalpha.harness.tools import ToolRegistry, ToolSpec
from cogalpha.llm.client import AgenticDecisionParseError, JSONToolDecisionClient


class FakeJSONCompletionClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete_json(
        self,
        context: str,
        schema_name: str,
        metadata: dict | None = None,
    ) -> str:
        self.calls.append(
            {
                "context": context,
                "schema_name": schema_name,
                "metadata": metadata,
            }
        )
        return self.response


def test_json_tool_decision_client_converts_strict_json_to_agent_decision():
    fake_json_client = FakeJSONCompletionClient(
        """
        {
          "content": "review generated candidates",
          "tool_calls": [
            {"name": "quality_pipeline.review", "arguments": {"limit": 1}}
          ],
          "stop_reason": null
        }
        """
    )
    client = JSONToolDecisionClient(fake_json_client)
    tools = [_review_tool_spec()]

    decision = client.decide(AgentLoopState(messages=[], context={"generation": 0}), tools=tools)

    assert decision.content == "review generated candidates"
    assert decision.tool_calls[0].name == "quality_pipeline.review"
    assert decision.tool_calls[0].arguments == {"limit": 1}
    assert decision.stop_reason is None
    assert fake_json_client.calls[0]["schema_name"] == "AgentDecision"
    assert fake_json_client.calls[0]["metadata"] == {
        "decision_schema": ["content", "tool_calls", "stop_reason"],
        "available_tools": ["quality_pipeline.review"],
    }


def test_json_tool_decision_client_rejects_malformed_json():
    client = JSONToolDecisionClient(FakeJSONCompletionClient("{not-json"))

    with pytest.raises(AgenticDecisionParseError, match="Malformed JSON decision"):
        client.decide(AgentLoopState(messages=[], context={}), tools=[_review_tool_spec()])


def test_json_tool_decision_client_rejects_missing_required_fields():
    client = JSONToolDecisionClient(
        FakeJSONCompletionClient('{"content": "missing tool calls"}')
    )

    with pytest.raises(AgenticDecisionParseError, match="Missing required decision field"):
        client.decide(AgentLoopState(messages=[], context={}), tools=[_review_tool_spec()])


def test_json_tool_decision_unknown_tool_flows_to_controller_validation():
    registry = ToolRegistry()
    registry.register(_review_tool_spec(), lambda _call, _context: {"accepted": []})
    client = JSONToolDecisionClient(
        FakeJSONCompletionClient(
            """
            {
              "content": "dispatch unknown tool",
              "tool_calls": [{"name": "missing.tool", "arguments": {}}],
              "stop_reason": null
            }
            """
        )
    )
    controller = AgenticController(client=client, tools=registry)

    with pytest.raises(AgenticDecisionValidationError, match="Unknown tool: missing.tool"):
        controller.decide(AgentLoopState(messages=[], context={}))


def _review_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="quality_pipeline.review",
        description="Review alpha candidates.",
        input_schema={"type": "object"},
    )
