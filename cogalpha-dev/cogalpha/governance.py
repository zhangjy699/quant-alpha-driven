"""Governance artifacts for evidence, experiments, and promotion decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvidenceDecision(StrEnum):
    """Decision status for evidence-backed changes."""

    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"
    COLLECT_MORE_EVIDENCE = "collect_more_evidence"


class ChangeType(StrEnum):
    """Layer where a candidate change applies."""

    ARTIFACT_CONTRACT = "artifact_contract"
    GATE_POLICY = "gate_policy"
    EVALUATOR_POLICY = "evaluator_policy"
    CANDIDATE_STRATEGY = "candidate_strategy"
    TOPOLOGY_CHANGE = "topology_change"
    DATA_POLICY = "data_policy"
    PROMPT_PATCH = "prompt_patch"


class EvidenceRecord(BaseModel):
    """Observation-to-decision artifact for harness changes."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    observation: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)
    attribution_layer: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)
    proposed_change: str = Field(..., min_length=1)
    change_type: ChangeType
    required_ablation: str = Field(..., min_length=1)
    artifacts: list[str] = Field(default_factory=list)
    decision: EvidenceDecision = EvidenceDecision.HOLD
    reviewer: str | None = None
    rollback: str | None = None


class PromotionDecision(BaseModel):
    """Supervised promotion or rollback decision for default behavior."""

    model_config = ConfigDict(extra="forbid")

    promotion_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidate_change: str = Field(..., min_length=1)
    evidence_ids: list[str] = Field(..., min_length=1)
    reviewer: str = Field(..., min_length=1)
    approver: str = Field(..., min_length=1)
    decision: EvidenceDecision
    effective_version: str | None = None
    rollback_target: str = Field(..., min_length=1)
    blocked_conditions: list[str] = Field(default_factory=list)


def append_jsonl_record(path: str | Path, record: BaseModel) -> None:
    """Append one governance record as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def load_jsonl_records(path: str | Path, schema: type[BaseModel]) -> list[BaseModel]:
    """Load governance JSONL records with schema validation."""

    input_path = Path(path)
    if not input_path.exists():
        return []
    records: list[BaseModel] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(schema.model_validate(json.loads(line)))
    return records
