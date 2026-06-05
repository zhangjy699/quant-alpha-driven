"""Skill invocation orchestration shared by Skill Nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cogalpha.llm import JSONCompletionClient
from cogalpha.skill_loader import SkillLoaderError, StandardSkillLoader

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class SkillInvocationContext:
    """Assembled skill invocation plus frontmatter-derived policy metadata."""

    skill_name: str
    prompt: str
    schema_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SkillInvoker:
    """Assemble standard skill context, call an LLM, and validate the artifact."""

    loader: StandardSkillLoader
    client: JSONCompletionClient
    inline_references: bool = False
    schema_retry_attempts: int = 1
    retrieval_cache_root: Path | str | None = None

    def invoke(
        self,
        skill_name: str,
        runtime_payload: str,
        output_schema: type[SchemaT],
    ) -> SchemaT:
        context = self.prepare_context(skill_name, runtime_payload, output_schema)
        raw_json = self.client.complete_json(
            context.prompt,
            context.schema_name,
            metadata=context.metadata,
        )
        try:
            return output_schema.model_validate_json(raw_json)
        except ValidationError as exc:
            if self.schema_retry_attempts <= 0:
                raise
            retry_prompt = _build_schema_retry_prompt(
                context.prompt,
                raw_json=raw_json,
                validation_error=str(exc),
            )
            raw_json = self.client.complete_json(
                retry_prompt,
                context.schema_name,
                metadata={**context.metadata, "retry_reason": "schema_validation_error"},
            )
            return output_schema.model_validate_json(raw_json)

    def prepare_context(
        self,
        skill_name: str,
        runtime_payload: str,
        output_schema: type[SchemaT],
    ) -> SkillInvocationContext:
        metadata = self.loader.discover().get(skill_name)
        if metadata is None:
            raise SkillLoaderError(f"Unknown skill: {skill_name}")
        if metadata.disable_model_invocation:
            raise PermissionError(
                f"Skill {skill_name!r} has disable-model-invocation enabled"
            )

        schema_text = json.dumps(output_schema.model_json_schema(), indent=2)
        prompt = self.loader.assemble_context(
            skill_name=skill_name,
            runtime_payload=runtime_payload,
            output_schema=schema_text,
            inline_references=self.inline_references,
        )
        retrieval_context = self._load_retrieval_context(skill_name)
        if retrieval_context:
            prompt = (
                f"{prompt}\n\n"
                "# Retrieved Factor Memory\n"
                f"{retrieval_context}"
            )
        return SkillInvocationContext(
            skill_name=skill_name,
            prompt=prompt,
            schema_name=output_schema.__name__,
            metadata={
                "allowed_tools": list(metadata.allowed_tools),
                "model": metadata.model,
                "effort": metadata.effort,
                "context": metadata.context,
                "argument_hint": metadata.argument_hint,
                "argument_names": list(metadata.argument_names),
                "paths": list(metadata.paths),
            },
        )

    def _load_retrieval_context(self, skill_name: str) -> str | None:
        if self.retrieval_cache_root is None:
            return None

        cache_path = Path(self.retrieval_cache_root) / f"{skill_name}.md"
        if not cache_path.exists() or not cache_path.is_file():
            return None

        text = cache_path.read_text(encoding="utf-8").strip()
        return text or None


def _build_schema_retry_prompt(
    prompt: str,
    *,
    raw_json: str,
    validation_error: str,
) -> str:
    return (
        f"{prompt}\n\n"
        "# Previous Invalid JSON Artifact\n"
        "Your previous response failed Runtime Schema validation. Return a corrected strict JSON "
        "object only, with no markdown fences or commentary.\n\n"
        "## Validation Error\n"
        f"{validation_error}\n\n"
        "## Previous Response\n"
        f"{raw_json}"
    )
