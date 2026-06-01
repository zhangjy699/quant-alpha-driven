"""Instrumentation helpers for formal CogAlpha workflow runs."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from cogalpha.skill_nodes import StructuredArtifactInvoker

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class InvocationRecorder:
    """Append skill invocation records to JSONL."""

    path: Path

    def append(self, record: dict) -> None:
        """Append one invocation record."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class RecordingInvoker:
    """StructuredArtifactInvoker wrapper that records latency and errors."""

    inner: StructuredArtifactInvoker
    recorder: InvocationRecorder
    context_variant: str
    calls: list[dict] = field(default_factory=list)

    def invoke(
        self,
        skill_name: str,
        runtime_payload: str,
        output_schema: type[SchemaT],
    ) -> SchemaT:
        """Invoke a skill and record public instrumentation."""

        started = time.perf_counter()
        record = {
            "created_at": datetime.now(UTC).isoformat(),
            "skill_name": skill_name,
            "schema_name": output_schema.__name__,
            "context_variant": self.context_variant,
            "runtime_payload_sha256": hashlib.sha256(
                runtime_payload.encode("utf-8")
            ).hexdigest(),
            "status": "ok",
        }
        try:
            result = self.inner.invoke(skill_name, runtime_payload, output_schema)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            self.recorder.append(record)
            self.calls.append(record)
        return result
