from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    success: bool
    output: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)


ToolHandler: TypeAlias = Callable[[ToolCall, dict[str, Any]], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    @property
    def specs(self) -> Mapping[str, ToolSpec]:
        return MappingProxyType(self._specs)

    def register(self, spec: ToolSpec, handler: ToolHandler) -> ToolRegistry:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler
        return self

    def dispatch(
        self,
        call: ToolCall,
        *,
        context: dict[str, Any],
        fail_fast: bool = False,
    ) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(
                name=call.name,
                success=False,
                error=f"Unknown tool: {call.name}",
            )

        try:
            return ToolResult(name=call.name, success=True, output=handler(call, context))
        except Exception as exc:
            if fail_fast:
                raise
            return ToolResult(name=call.name, success=False, error=str(exc))

    def dispatch_all(
        self,
        calls: Iterable[ToolCall],
        *,
        context: dict[str, Any],
        fail_fast: bool = False,
    ) -> list[ToolResult]:
        return [
            self.dispatch(call, context=context, fail_fast=fail_fast)
            for call in calls
        ]
