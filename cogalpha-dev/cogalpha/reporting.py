"""Layered preflight and evaluation reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cogalpha.schemas import CogAlphaState
from cogalpha.tracing import read_trace_events
from cogalpha.verification.trace_verifier import (
    TraceVerificationFinding,
    TraceVerificationReport,
    verify_cogalpha_trace,
)


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


class AgenticReplayReport(BaseModel):
    """Replay result for trace-first agentic run artifacts."""

    model_config = ConfigDict(extra="forbid")

    final_state_path: str
    trace_path: str
    trace_event_count: int
    passed: bool
    findings: list[TraceVerificationFinding] = Field(default_factory=list)


def replay_agentic_run(run_dir: str | Path) -> AgenticReplayReport:
    """Load final state and trace JSONL, then run semantic trace verification."""

    run_path = Path(run_dir)
    final_state_path = run_path / "final_state.json"
    trace_path = run_path / "trace.jsonl"
    final_state = CogAlphaState.model_validate_json(
        final_state_path.read_text(encoding="utf-8")
    )
    trace_events = read_trace_events(trace_path)
    verification = verify_cogalpha_trace(final_state, trace_events)
    return AgenticReplayReport(
        final_state_path=str(final_state_path),
        trace_path=str(trace_path),
        trace_event_count=len(trace_events),
        passed=verification.passed,
        findings=verification.findings,
    )


def build_agentic_run_report(
    *,
    summary: dict,
    data_version: str,
    manifest_path: str | Path,
    trace_verification: TraceVerificationReport,
) -> EvaluationRunReport:
    """Build a formal run report gated by trace verification evidence."""

    blockers: list[str] = []
    workflow_status = StopGoDecision.GO
    if summary["skill_errors"] or summary["remaining_candidates"]:
        workflow_status = StopGoDecision.STOP
        blockers.append("Workflow ended with skill errors or unevaluated candidates.")
    if not trace_verification.passed:
        workflow_status = StopGoDecision.STOP
        finding_codes = ", ".join(finding.code for finding in trace_verification.findings)
        blockers.append(f"Trace verification failed: {finding_codes}.")

    effect_status = StopGoDecision.GO if summary["qualified"] else StopGoDecision.HOLD
    if not summary["qualified"]:
        blockers.append(
            "No candidate qualified on validation; do not treat run as performance evidence."
        )

    promotion_status = StopGoDecision.HOLD
    blockers.append("Promotion requires fixed comparison, review, and rollback pointer.")

    decision = (
        StopGoDecision.STOP if workflow_status == StopGoDecision.STOP else StopGoDecision.HOLD
    )
    return EvaluationRunReport(
        report_id=f"{summary['run_id']}-report",
        purpose="Formal agentic MVP workflow run report",
        data_version=data_version,
        manifest_path=str(manifest_path),
        layers=[
            ReportLayer(
                name="data_contract",
                status=StopGoDecision.GO,
                summary=(
                    "Prepared market-data split uses configured next-open forward returns and recorded "
                    "data_version."
                ),
                artifacts=[str(manifest_path)],
            ),
            ReportLayer(
                name="workflow_execution",
                status=workflow_status,
                summary=(
                    f"Nodes executed: {summary['node_history']}; skill_errors="
                    f"{summary['skill_errors']}; remaining_candidates="
                    f"{summary['remaining_candidates']}; trace_passed="
                    f"{trace_verification.passed}."
                ),
                artifacts=[
                    "summary.json",
                    "final_state.json",
                    "skill_invocations.jsonl",
                    "trace.jsonl",
                    "trace_verification.json",
                ],
            ),
            ReportLayer(
                name="effect_evaluation",
                status=effect_status,
                summary=(
                    f"qualified={summary['qualified']}, elite={summary['elite']}, "
                    f"rejected={summary['rejected']} on split={summary['split']}."
                ),
                artifacts=["evaluation_cache.jsonl", "final_state.json"],
            ),
            ReportLayer(
                name="promotion_governance",
                status=promotion_status,
                summary="Single run cannot promote prompt, topology, gate, or default behavior.",
                artifacts=["run_manifest.json"],
            ),
        ],
        decision=decision,
        blockers=blockers,
    )


def write_evaluation_run_report(path: str | Path, report: EvaluationRunReport) -> None:
    """Write a layered report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
