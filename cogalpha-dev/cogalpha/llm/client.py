"""Provider-agnostic JSON completion client contract."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from cogalpha.harness.loop import AgentDecision, AgentLoopState
from cogalpha.harness.tools import ToolCall, ToolSpec


class JSONCompletionClient(Protocol):
    """LLM interface used by Skill Nodes."""

    def complete_json(
        self,
        context: str,
        schema_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return a strict JSON string compatible with the requested schema."""


class AgenticDecisionParseError(ValueError):
    """Raised when a JSON decision cannot be converted into an AgentDecision."""


@dataclass(frozen=True)
class JSONToolDecisionClient:
    """Convert strict provider JSON completions into agentic tool decisions."""

    client: JSONCompletionClient

    def decide(
        self,
        state: AgentLoopState,
        *,
        tools: list[ToolSpec],
    ) -> AgentDecision:
        raw_json = self.client.complete_json(
            _build_tool_decision_context(state, tools),
            "AgentDecision",
            metadata={
                "decision_schema": ["content", "tool_calls", "stop_reason"],
                "available_tools": [tool.name for tool in tools],
            },
        )
        payload = _parse_decision_json(raw_json)
        return AgentDecision(
            content=payload["content"],
            tool_calls=[
                ToolCall(
                    name=call["name"],
                    arguments=call["arguments"],
                )
                for call in payload["tool_calls"]
            ],
            stop_reason=payload["stop_reason"],
        )


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions adapter.

    This adapter intentionally targets the common `/chat/completions` shape so the
    project can point at OpenAI-compatible providers without changing graph nodes.
    """

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.8
    timeout_seconds: int = 120
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None
    response_format: str | None = "json_object"

    @classmethod
    def from_env(cls) -> OpenAICompatibleClient:
        """Create a client from environment variables."""

        api_key = (
            os.environ.get("COGALPHA_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        base_url = os.environ.get("COGALPHA_LLM_BASE_URL") or os.environ.get(
            "DEEPSEEK_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        model = (
            os.environ.get("COGALPHA_LLM_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or os.environ.get("CHAT_MODEL", "")
        )
        max_tokens = _parse_optional_int(os.environ.get("COGALPHA_LLM_MAX_TOKENS"))
        reasoning_effort = os.environ.get("COGALPHA_LLM_REASONING_EFFORT") or os.environ.get(
            "DEEPSEEK_REASONING_EFFORT"
        )
        thinking = os.environ.get("COGALPHA_LLM_THINKING") or os.environ.get("DEEPSEEK_THINKING")
        response_format = os.environ.get("COGALPHA_LLM_RESPONSE_FORMAT", "json_object")
        if response_format.lower() in {"", "none", "off", "false"}:
            response_format = None
        if not api_key:
            raise RuntimeError("Missing COGALPHA_LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
        if not model:
            raise RuntimeError("Missing COGALPHA_LLM_MODEL or CHAT_MODEL")
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            response_format=response_format,
        )

    def complete_json(
        self,
        context: str,
        schema_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        system_prompt = (
            "You are executing a CogAlpha Standard Skill. "
            "Return strict JSON only. Do not include markdown fences or commentary. "
            f"The JSON must conform to the requested Runtime Schema: {schema_name}."
        )
        if metadata:
            policy = json.dumps(metadata, sort_keys=True)
            system_prompt += f"\nSkill invocation metadata: {policy}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
        }
        if self.response_format:
            payload["response_format"] = {"type": self.response_format}
        if self.thinking != "enabled":
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        return data["choices"][0]["message"]["content"]


def _parse_optional_int(raw_value: str | None) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def _build_tool_decision_context(state: AgentLoopState, tools: list[ToolSpec]) -> str:
    tool_payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": dict(tool.input_schema),
        }
        for tool in tools
    ]
    state_payload = {
        "turns": state.turns,
        "context": state.context,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "tool_result": None
                if message.tool_result is None
                else {
                    "name": message.tool_result.name,
                    "success": message.tool_result.success,
                    "output": message.tool_result.output,
                    "error": message.tool_result.error,
                },
            }
            for message in state.messages
        ],
    }
    return (
        "Choose the next CogAlpha tool call or stop decision. Return strict JSON only "
        "with required fields: content, tool_calls, stop_reason. Use an empty tool_calls "
        "array only when stop_reason is a non-empty string.\n\n"
        f"Available tools:\n{json.dumps(tool_payload, sort_keys=True)}\n\n"
        f"Current state:\n{json.dumps(state_payload, sort_keys=True, default=str)}"
    )


def _parse_decision_json(raw_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        msg = f"Malformed JSON decision: {exc.msg}"
        raise AgenticDecisionParseError(msg) from exc

    if not isinstance(payload, Mapping):
        msg = "Agentic decision must be a JSON object"
        raise AgenticDecisionParseError(msg)

    for field_name in ("content", "tool_calls", "stop_reason"):
        if field_name not in payload:
            msg = f"Missing required decision field: {field_name}"
            raise AgenticDecisionParseError(msg)

    if not isinstance(payload["content"], str):
        msg = "Decision field content must be a string"
        raise AgenticDecisionParseError(msg)
    if payload["stop_reason"] is not None and not isinstance(payload["stop_reason"], str):
        msg = "Decision field stop_reason must be a string or null"
        raise AgenticDecisionParseError(msg)
    if not isinstance(payload["tool_calls"], list):
        msg = "Decision field tool_calls must be a list"
        raise AgenticDecisionParseError(msg)

    tool_calls = [_parse_tool_call(item, index) for index, item in enumerate(payload["tool_calls"])]
    return {
        "content": payload["content"],
        "tool_calls": tool_calls,
        "stop_reason": payload["stop_reason"],
    }


def _parse_tool_call(raw_call: object, index: int) -> dict[str, Any]:
    if not isinstance(raw_call, Mapping):
        msg = f"tool_calls[{index}] must be an object"
        raise AgenticDecisionParseError(msg)
    if "name" not in raw_call:
        msg = f"tool_calls[{index}] missing required field: name"
        raise AgenticDecisionParseError(msg)
    if "arguments" not in raw_call:
        msg = f"tool_calls[{index}] missing required field: arguments"
        raise AgenticDecisionParseError(msg)
    if not isinstance(raw_call["name"], str) or not raw_call["name"].strip():
        msg = f"tool_calls[{index}].name must be a non-empty string"
        raise AgenticDecisionParseError(msg)
    if not isinstance(raw_call["arguments"], Mapping):
        msg = f"tool_calls[{index}].arguments must be an object"
        raise AgenticDecisionParseError(msg)
    return {"name": raw_call["name"], "arguments": dict(raw_call["arguments"])}
