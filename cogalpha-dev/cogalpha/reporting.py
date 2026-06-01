"""Layered preflight and evaluation reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StopGoDecision(StrEnum):
    """Decision for whether a run may proceed."""

    GO = "go"
    HOLD = "hold"
    STOP = "stop"


class ReportLayer(BaseModel):
    """One layer in a rubric-aligned run report."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    status: StopGoDecision
    summary: str = Field(..., min_length=1)
    artifacts: list[str] = Field(default_factory=list)


class EvaluationRunReport(BaseModel):
    """Layered report for preflight or formal evaluation readiness."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    purpose: str = Field(..., min_length=1)
    data_version: str
    manifest_path: str
    layers: list[ReportLayer] = Field(default_factory=list)
    decision: StopGoDecision
    blockers: list[str] = Field(default_factory=list)


def write_evaluation_run_report(path: str | Path, report: EvaluationRunReport) -> None:
    """Write a layered report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
