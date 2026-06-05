import pytest

from cogalpha.harness import ToolCall, ToolRegistry, ToolSpec


def test_tool_registry_dispatches_registered_tool_with_shared_context():
    registry = ToolRegistry()

    def remember(call: ToolCall, context: dict):
        context["seen"] = call.arguments["value"]
        return {"remembered": context["seen"]}

    registry.register(
        ToolSpec(
            name="memory.remember",
            description="Store a value in shared context.",
            input_schema={"type": "object"},
        ),
        remember,
    )

    context = {}
    result = registry.dispatch(
        ToolCall(name="memory.remember", arguments={"value": "alpha"}),
        context=context,
    )

    assert result.name == "memory.remember"
    assert result.success is True
    assert result.output == {"remembered": "alpha"}
    assert result.error is None
    assert context == {"seen": "alpha"}


def test_tool_registry_returns_structured_error_for_unknown_tool():
    registry = ToolRegistry()

    result = registry.dispatch(ToolCall(name="missing.tool", arguments={}), context={})

    assert result.name == "missing.tool"
    assert result.success is False
    assert result.output is None
    assert result.error == "Unknown tool: missing.tool"


def test_tool_registry_fail_fast_reraises_handler_errors():
    registry = ToolRegistry()

    def broken(_call: ToolCall, _context: dict):
        raise ValueError("bad input")

    registry.register(
        ToolSpec(
            name="broken.tool",
            description="Raise a predictable error.",
            input_schema={"type": "object"},
        ),
        broken,
    )

    with pytest.raises(ValueError, match="bad input"):
        registry.dispatch(ToolCall(name="broken.tool", arguments={}), context={}, fail_fast=True)


def test_tool_registry_dispatch_all_preserves_order():
    registry = ToolRegistry()

    def echo(call: ToolCall, _context: dict):
        return call.arguments["value"]

    registry.register(
        ToolSpec(
            name="echo",
            description="Echo a value.",
            input_schema={"type": "object"},
        ),
        echo,
    )

    results = registry.dispatch_all(
        [
            ToolCall(name="echo", arguments={"value": 1}),
            ToolCall(name="echo", arguments={"value": 2}),
        ],
        context={},
    )

    assert [result.output for result in results] == [1, 2]
