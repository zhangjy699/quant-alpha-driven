"""Trace-grounded skill utility and governed update records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogalpha.schemas import CandidateStage
from cogalpha.tracing import CogAlphaTraceEvent, TraceEventKind


class SkillUtilityRecord(BaseModel):
    """Utility accounting for one standard skill from observed traces."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(..., min_length=1)
    utility: float = Field(default=0.0, ge=-1.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    evidence_ids: list[str] = Field(default_factory=list)
    last_evidence_id: str | None = None


class SkillSelectionRecord(BaseModel):
    """Traceable record explaining why a skill was selected."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(..., min_length=1)
    selected_skill: str = Field(..., min_length=1)
    eligible_skills: list[str] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def selected_skill_must_be_eligible(self) -> SkillSelectionRecord:
        if self.selected_skill not in self.eligible_skills:
            msg = "selected_skill must be present in eligible_skills"
            raise ValueError(msg)
        return self


class SkillUpdateProposal(BaseModel):
    """Governed proposal for changing a skill document."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(..., min_length=1)
    skill_name: str = Field(..., min_length=1)
    proposed_change: str = Field(..., min_length=1)
    status: Literal["draft", "hold", "reject", "promote"] = "draft"
    evidence_id: str | None = None
    reviewer: str | None = None
    rollback: str | None = None

    @model_validator(mode="after")
    def promote_requires_governance_fields(self) -> SkillUpdateProposal:
        if self.status == "promote" and not (
            self.evidence_id and self.reviewer and self.rollback
        ):
            msg = "promote proposals require evidence_id, reviewer, and rollback"
            raise ValueError(msg)
        return self


def update_skill_utility_from_trace(
    record: SkillUtilityRecord,
    events: list[CogAlphaTraceEvent],
) -> SkillUtilityRecord:
    """Return an updated utility record from trace outcomes for this skill."""

    relevant = [_event for _event in events if _payload_skill(_event) == record.skill_name]
    usage_count = record.usage_count + _count_skill_invocations(relevant)
    evidence_ids = _merge_evidence_ids(record.evidence_ids, relevant)
    outcome_delta = _best_outcome_delta(relevant)
    utility = _bounded(record.utility + outcome_delta)

    return record.model_copy(
        update={
            "utility": utility,
            "usage_count": usage_count,
            "evidence_ids": evidence_ids,
            "last_evidence_id": evidence_ids[-1] if evidence_ids else record.last_evidence_id,
        }
    )


def write_skill_utility_records(
    path: str | Path,
    records: list[SkillUtilityRecord],
) -> None:
    """Write skill utility records as a stable JSON artifact."""

    output_path = Path(path)
    output_path.write_text(
        json.dumps(
            [record.model_dump(mode="json") for record in records],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def read_skill_utility_records(path: str | Path) -> list[SkillUtilityRecord]:
    """Read skill utility records from a JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SkillUtilityRecord.model_validate(record) for record in payload]


def write_skill_selection_records(
    path: str | Path,
    records: list[SkillSelectionRecord],
) -> None:
    """Write skill selection records as newline-delimited JSON."""

    output_path = Path(path)
    output_path.write_text(
        "".join(
            f"{json.dumps(record.model_dump(mode='json'), sort_keys=True)}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def read_skill_selection_records(path: str | Path) -> list[SkillSelectionRecord]:
    """Read skill selection records from newline-delimited JSON."""

    return [
        SkillSelectionRecord.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _payload_skill(event: CogAlphaTraceEvent) -> str | None:
    skill_name = event.payload.get("skill_name") or event.payload.get("skill")
    return str(skill_name) if skill_name else None


def _count_skill_invocations(events: list[CogAlphaTraceEvent]) -> int:
    return sum(
        event.kind
        in {
            TraceEventKind.SKILL_INVOCATION_FINISHED,
            TraceEventKind.SKILL_INVOCATION_STARTED,
        }
        for event in events
    )


def _merge_evidence_ids(
    existing: list[str],
    events: list[CogAlphaTraceEvent],
) -> list[str]:
    merged = list(existing)
    seen = set(merged)
    for event in events:
        evidence_id = event.payload.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id and evidence_id not in seen:
            merged.append(evidence_id)
            seen.add(evidence_id)
    return merged


def _best_outcome_delta(events: list[CogAlphaTraceEvent]) -> float:
    deltas = [_event_delta(event) for event in events]
    if not deltas:
        return 0.0
    return max(deltas, key=abs)


def _event_delta(event: CogAlphaTraceEvent) -> float:
    if event.kind == TraceEventKind.FITNESS_EVALUATION_RECORDED:
        stage = _coerce_stage(event.payload.get("stage"))
        if stage in {CandidateStage.QUALIFIED, CandidateStage.ELITE}:
            if event.payload.get("status") not in {"ok", "success"}:
                return 0.0
            if not _has_complete_metrics(event):
                return 0.0
        if stage == CandidateStage.REJECTED_BY_FITNESS:
            return -0.10
    return _stage_delta(event.payload.get("stage"))


def _coerce_stage(stage: object) -> CandidateStage | None:
    try:
        return CandidateStage(stage)
    except ValueError:
        return None


def _has_complete_metrics(event: CogAlphaTraceEvent) -> bool:
    metrics = event.payload.get("metrics")
    return isinstance(metrics, dict) and {
        "ic",
        "rank_ic",
        "icir",
        "rank_icir",
        "mi",
    }.issubset(metrics)


def _stage_delta(stage: object) -> float:
    candidate_stage = _coerce_stage(stage)
    if candidate_stage is None:
        return 0.0

    if candidate_stage == CandidateStage.ELITE:
        return 0.35
    if candidate_stage == CandidateStage.QUALIFIED:
        return 0.20
    if candidate_stage in {
        CandidateStage.REJECTED_BY_FITNESS,
        CandidateStage.REJECTED_BY_QUALITY,
    }:
        return -0.10
    return 0.0


def _bounded(value: float) -> float:
    return min(1.0, max(-1.0, value))
