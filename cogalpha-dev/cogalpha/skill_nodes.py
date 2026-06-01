"""Skill Node invocation helpers for Structured Artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from cogalpha.schemas import AlphaCandidate, AlphaCandidateBatch, QualityDecision

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredArtifactInvoker(Protocol):
    """Invoker shape required by Skill Nodes."""

    def invoke(
        self,
        skill_name: str,
        runtime_payload: str,
        output_schema: type[SchemaT],
    ) -> SchemaT:
        """Invoke a Standard Skill and return a Structured Artifact."""


@dataclass(frozen=True)
class SkillNodeRuntime:
    """Invoke Standard Skills using Runtime Schema objects as the Interface."""

    invoker: StructuredArtifactInvoker

    def candidate_batch(self, skill_name: str, request: BaseModel) -> AlphaCandidateBatch:
        """Invoke a Domain Agent Skill."""

        return self._invoke_request(skill_name, request, AlphaCandidateBatch)

    def quality_decision(self, skill_name: str, request: BaseModel) -> QualityDecision:
        """Invoke a Quality Checker Skill."""

        return self._invoke_request(skill_name, request, QualityDecision)

    def alpha_candidate(self, skill_name: str, request: BaseModel) -> AlphaCandidate:
        """Invoke an Evolution Operator Skill."""

        return self._invoke_request(skill_name, request, AlphaCandidate)

    def _invoke_request(
        self,
        skill_name: str,
        request: BaseModel,
        output_schema: type[SchemaT],
    ) -> SchemaT:
        return self.invoker.invoke(
            skill_name=skill_name,
            runtime_payload=runtime_payload_json(request),
            output_schema=output_schema,
        )


def runtime_payload_json(request: BaseModel) -> str:
    """Serialize only the public Runtime Schema surface sent to a skill."""

    payload = request.model_dump(mode="json", exclude_none=True)
    return json.dumps(_strip_private_payload_fields(payload), indent=2, sort_keys=True)


def _strip_private_payload_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private_payload_fields(child)
            for key, child in value.items()
            if key not in {"created_at", "metadata"}
        }
    if isinstance(value, list):
        return [_strip_private_payload_fields(child) for child in value]
    return value
