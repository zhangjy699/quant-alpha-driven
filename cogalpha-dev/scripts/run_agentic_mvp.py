"""Run the CogAlpha MVP workflow through the trace-first agentic controller."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cogalpha.config import MVPLoopConfig
from cogalpha.data import load_prepared_baseline_market_data
from cogalpha.evaluation import EvaluationCache, PanelBackedMetricsProvider
from cogalpha.guards import DeterministicGuardPipeline
from cogalpha.harness.agentic import AgenticController, AgenticDecisionClient
from cogalpha.harness.cogalpha_tools import (
    COGALPHA_STATE_KEY,
    CogAlphaRuntime,
    build_cogalpha_tools,
)
from cogalpha.harness.loop import run_agent_loop
from cogalpha.harness.tools import ToolCall, ToolRegistry, ToolResult
from cogalpha.instrumentation import InvocationRecorder, RecordingInvoker
from cogalpha.llm import OpenAICompatibleClient
from cogalpha.llm.client import JSONToolDecisionClient
from cogalpha.manifest import build_run_manifest, write_run_manifest
from cogalpha.registry import DOMAIN_AGENT_SPECS, PROJECT_ROOT, all_skill_refs
from cogalpha.reporting import build_agentic_run_report, write_evaluation_run_report
from cogalpha.schemas import CandidateStage, CogAlphaState, DAGNodeResult
from cogalpha.skill_invocation import SkillInvoker
from cogalpha.skill_library import (
    SkillSelectionRecord,
    SkillUtilityRecord,
    update_skill_utility_from_trace,
    write_skill_selection_records,
    write_skill_utility_records,
)
from cogalpha.skill_loader import StandardSkillLoader
from cogalpha.tracing import TraceEventKind, TraceLedger, read_trace_events
from cogalpha.verification.trace_verifier import (
    TraceVerificationReport,
    verify_cogalpha_trace,
)
from scripts.run_formal_mvp import (
    _assert_formal_run_complete,
    _configure_llm_provider,
    _load_key_file,
    _summarize_run,
)


@dataclass(frozen=True)
class AgenticRunOutput:
    run_dir: Path
    final_state: CogAlphaState
    summary: dict[str, Any]
    trace_verification: TraceVerificationReport


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal agentic CogAlpha MVP experiment.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", default="data/processed/csi300")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--agent-limit", type=int, default=3)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--alphas-per-agent", type=int, default=1)
    parser.add_argument("--parent-pool-size", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--inline-references", action="store_true")
    parser.add_argument("--key-file", default="KEY.md")
    parser.add_argument("--provider", choices=["deepseek", "openai", "custom"], default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    _load_key_file(args.key_file)
    _configure_llm_provider(args)
    run_id = args.run_id or datetime.now(UTC).strftime("agentic-mvp-%Y%m%d-%H%M%S")
    metadata = json.loads((Path(args.data_dir) / "metadata.json").read_text(encoding="utf-8"))
    market_data = load_prepared_baseline_market_data(args.data_dir)
    data_split = market_data.split(args.split)
    run_dir = Path(args.output_root) / run_id
    cache = EvaluationCache(run_dir / "evaluation_cache.jsonl")
    metrics_provider = PanelBackedMetricsProvider.from_split(
        data_split,
        data_version=metadata["data_version"],
        cache=cache,
    )
    guard_pipeline = DeterministicGuardPipeline(runtime_ohlcv_panel=data_split.ohlcv_panel)
    client = OpenAICompatibleClient.from_env()
    skill_invoker = SkillInvoker(
        loader=StandardSkillLoader(PROJECT_ROOT / "skills"),
        client=client,
        inline_references=args.inline_references,
    )
    decision_client = JSONToolDecisionClient(client)
    config = MVPLoopConfig(
        alphas_per_domain_agent=args.alphas_per_agent,
        max_generations=args.max_generations,
        max_repair_attempts=args.max_repair_attempts,
        parent_pool_size=args.parent_pool_size,
    )

    output = run_agentic_formal_mvp(
        run_id=run_id,
        output_root=args.output_root,
        config=config,
        decision_client=decision_client,
        skill_invoker=skill_invoker,
        metrics_provider=metrics_provider,
        guard_pipeline=guard_pipeline,
        data_version=metadata["data_version"],
        split=args.split,
        agent_limit=args.agent_limit,
        inline_references=args.inline_references,
        model_settings={
            "model": client.model,
            "base_url": client.base_url,
            "temperature": str(client.temperature),
            "reasoning_effort": str(client.reasoning_effort),
            "thinking": str(client.thinking),
            "max_tokens": str(client.max_tokens),
            "inline_references": str(args.inline_references),
        },
    )
    print(json.dumps(output.summary, indent=2, sort_keys=True))


def run_agentic_formal_mvp(
    *,
    run_id: str,
    output_root: str | Path,
    config: MVPLoopConfig,
    decision_client: AgenticDecisionClient,
    skill_invoker: Any,
    metrics_provider: Any,
    data_version: str,
    split: str,
    agent_limit: int,
    guard_pipeline: DeterministicGuardPipeline | None = None,
    inline_references: bool = False,
    model_settings: dict[str, str] | None = None,
) -> AgenticRunOutput:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _reset_output_file(run_dir / "trace.jsonl")
    _reset_output_file(run_dir / "skill_invocations.jsonl")

    recording_invoker = RecordingInvoker(
        inner=skill_invoker,
        recorder=InvocationRecorder(run_dir / "skill_invocations.jsonl"),
        context_variant="inline_references" if inline_references else "minimal",
    )
    (run_dir / "actual_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    trace_ledger = TraceLedger(run_id=run_id, path=run_dir / "trace.jsonl")
    trace_ledger.record(
        TraceEventKind.RUN_STARTED,
        payload={"split": split, "data_version": data_version},
    )
    agent_specs = tuple(DOMAIN_AGENT_SPECS[:agent_limit])
    runtime = CogAlphaRuntime(
        invoker=recording_invoker,
        config=config,
        metrics_provider=metrics_provider,
        guard_pipeline=guard_pipeline,
        agent_specs=agent_specs,
    )
    tools = _TracingToolRegistry(
        inner=build_cogalpha_tools(runtime),
        trace_ledger=trace_ledger,
        invoker=recording_invoker,
    )
    controller = AgenticController(
        client=decision_client,
        tools=tools,
        trace_ledger=trace_ledger,
    )
    initial_state = CogAlphaState(metadata={"run_id": run_id, "split": split})
    context = {COGALPHA_STATE_KEY: initial_state.model_dump(mode="python")}

    run_agent_loop(
        adapter=controller,
        tools=tools,
        messages=[],
        context=context,
        max_turns=max(4, config.max_generations * 4 + 4),
    )

    final_state = CogAlphaState.model_validate(context[COGALPHA_STATE_KEY])
    (run_dir / "final_state.json").write_text(
        final_state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _assert_formal_run_complete(final_state)
    summary = _summarize_run(
        final_state,
        recording_invoker,
        metrics_provider,
        argparse.Namespace(
            split=split,
            agent_limit=agent_limit,
            max_generations=config.max_generations,
        ),
        {"data_version": data_version},
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        manifest_id=run_id,
        purpose="Formal agentic MVP controller run",
        data_version=data_version,
        config_paths=["configs/baseline.yaml", "configs/mvp.yaml"],
        skill_paths=[ref.path for ref in all_skill_refs()],
        code_paths=[
            "scripts/run_agentic_mvp.py",
            "cogalpha/harness/agentic.py",
            "cogalpha/harness/cogalpha_tools.py",
            "cogalpha/tracing.py",
            "cogalpha/verification/trace_verifier.py",
            "cogalpha/reporting.py",
        ],
        fixed_inputs=[
            split,
            *[spec.skill_name for spec in agent_specs],
        ],
        model_settings=model_settings or {},
        notes="Trace-first agentic run. Deterministic MVP mode remains compatibility only.",
    )
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    trace_ledger.record(
        TraceEventKind.RUN_FINISHED,
        payload={
            "qualified": len(final_state.qualified_pool),
            "elite": len(final_state.elite_pool),
            "rejected": len(final_state.rejected_pool),
        },
    )
    trace_events = read_trace_events(run_dir / "trace.jsonl")
    skill_selection_records = _build_skill_selection_records(
        run_id=run_id,
        events=trace_events,
        eligible_skills=[spec.skill_name for spec in agent_specs],
    )
    write_skill_selection_records(run_dir / "skill_selection.jsonl", skill_selection_records)
    skill_utility_records = _build_skill_utility_records(
        trace_events,
        skill_selection_records,
    )
    write_skill_utility_records(run_dir / "skill_utility.json", skill_utility_records)
    trace_verification = verify_cogalpha_trace(final_state, trace_events)
    _write_trace_verification(run_dir / "trace_verification.json", trace_verification)
    _write_trace_manifest(
        run_dir / "trace_manifest.json",
        run_id=run_id,
        trace_events=len(trace_events),
    )
    report = build_agentic_run_report(
        summary=summary,
        data_version=data_version,
        manifest_path=run_dir / "run_manifest.json",
        trace_verification=trace_verification,
    )
    write_evaluation_run_report(run_dir / "evaluation_run_report.json", report)
    return AgenticRunOutput(
        run_dir=run_dir,
        final_state=final_state,
        summary=summary,
        trace_verification=trace_verification,
    )


class _TracingToolRegistry:
    def __init__(
        self,
        *,
        inner: ToolRegistry,
        trace_ledger: TraceLedger,
        invoker: RecordingInvoker,
    ) -> None:
        self._inner = inner
        self._trace_ledger = trace_ledger
        self._invoker = invoker
        self._skill_evidence_by_candidate_id: dict[str, tuple[str, str]] = {}

    @property
    def specs(self):
        return self._inner.specs

    def dispatch_all(
        self,
        calls: list[ToolCall],
        *,
        context: dict[str, Any],
        fail_fast: bool = False,
    ) -> list[ToolResult]:
        return [
            self.dispatch(call, context=context, fail_fast=fail_fast)
            for call in calls
        ]

    def dispatch(
        self,
        call: ToolCall,
        *,
        context: dict[str, Any],
        fail_fast: bool = False,
    ) -> ToolResult:
        self._trace_ledger.record(
            TraceEventKind.TOOL_CALL_STARTED,
            payload={"tool_name": call.name, "arguments": dict(call.arguments)},
        )
        invocation_start = len(self._invoker.calls)
        result = self._inner.dispatch(call, context=context, fail_fast=fail_fast)
        new_records = self._invoker.calls[invocation_start:]
        for invocation_index, record in enumerate(new_records, start=invocation_start + 1):
            skill_name = record.get("skill_name")
            evidence_id = (
                f"{self._trace_ledger.run_id}:skill:{invocation_index}:{skill_name}"
                if isinstance(skill_name, str) and skill_name
                else None
            )
            if evidence_id is not None:
                record["evidence_id"] = evidence_id
            self._trace_ledger.record(
                TraceEventKind.SKILL_INVOCATION_FINISHED,
                payload={
                    "skill_name": skill_name,
                    "schema_name": record.get("schema_name"),
                    "status": record.get("status"),
                    "evidence_id": evidence_id,
                },
            )
        if result.success:
            _record_state_trace_events(
                self._trace_ledger,
                result.output,
                self._skill_evidence_by_candidate_id,
                new_records,
            )
        self._trace_ledger.record(
            TraceEventKind.TOOL_CALL_FINISHED,
            payload={
                "tool_name": call.name,
                "status": "ok" if result.success else "error",
                "error": result.error,
            },
        )
        return result


def _record_state_trace_events(
    ledger: TraceLedger,
    output: Any,
    skill_evidence_by_candidate_id: dict[str, tuple[str, str]],
    new_invocation_records: list[dict],
) -> None:
    state = CogAlphaState.model_validate(output)
    if not state.node_history:
        return

    latest = state.node_history[-1]
    fallback_evidence = _latest_skill_evidence(new_invocation_records)
    for candidate in latest.candidates:
        evidence = _candidate_skill_evidence(
            candidate_id=candidate.candidate_id,
            candidate_skill=candidate.lineage.agent_skill,
            fallback_evidence=fallback_evidence,
            skill_evidence_by_candidate_id=skill_evidence_by_candidate_id,
        )
        if evidence is not None:
            skill_evidence_by_candidate_id[candidate.candidate_id] = evidence
        skill_name, evidence_id = evidence if evidence is not None else (None, None)
        ledger.record(
            TraceEventKind.CANDIDATE_STAGE_CHANGED,
            payload={
                "candidate_id": candidate.candidate_id,
                "stage": candidate.stage,
                "node_name": latest.node_name,
                "skill_name": skill_name,
                "evidence_id": evidence_id,
            },
        )
    _record_quality_trace_events(ledger, latest, state)
    fitness_stage_by_candidate_id = {
        decision.candidate_id: decision.stage for decision in latest.fitness_decisions
    }
    for evaluation in latest.evaluation_results:
        payload = {
            "candidate_id": evaluation.candidate_id,
            "status": "ok" if evaluation.metrics is not None else "error",
            "node_name": latest.node_name,
            "cache_hit": evaluation.cache_hit,
        }
        stage = fitness_stage_by_candidate_id.get(evaluation.candidate_id)
        if stage is not None:
            payload["stage"] = stage
        evidence = skill_evidence_by_candidate_id.get(evaluation.candidate_id)
        if evidence is not None:
            payload["skill_name"], payload["evidence_id"] = evidence
        if evaluation.metrics is not None:
            payload["metrics"] = evaluation.metrics.model_dump(mode="json")
        if evaluation.error:
            payload["error"] = evaluation.error
        ledger.record(TraceEventKind.FITNESS_EVALUATION_RECORDED, payload=payload)


def _latest_skill_evidence(records: list[dict]) -> tuple[str, str] | None:
    for record in reversed(records):
        skill_name = record.get("skill_name")
        evidence_id = record.get("evidence_id")
        if isinstance(skill_name, str) and isinstance(evidence_id, str):
            return skill_name, evidence_id
    return None


def _candidate_skill_evidence(
    *,
    candidate_id: str,
    candidate_skill: str | None,
    fallback_evidence: tuple[str, str] | None,
    skill_evidence_by_candidate_id: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    if candidate_id in skill_evidence_by_candidate_id:
        return skill_evidence_by_candidate_id[candidate_id]
    if candidate_skill and fallback_evidence and fallback_evidence[0] == candidate_skill:
        return fallback_evidence
    if fallback_evidence and candidate_id.startswith(f"{fallback_evidence[0]}-"):
        return fallback_evidence
    return None


def _build_skill_selection_records(
    *,
    run_id: str,
    events: list,
    eligible_skills: list[str],
) -> list[SkillSelectionRecord]:
    records: list[SkillSelectionRecord] = []
    for event in events:
        if event.kind != TraceEventKind.SKILL_INVOCATION_FINISHED:
            continue
        skill_name = event.payload.get("skill_name")
        evidence_id = event.payload.get("evidence_id")
        if not isinstance(skill_name, str) or not skill_name:
            continue
        if skill_name not in eligible_skills:
            continue
        records.append(
            SkillSelectionRecord(
                decision_id=f"{run_id}:selection:{len(records) + 1}:{skill_name}",
                selected_skill=skill_name,
                eligible_skills=eligible_skills,
                reason="runtime tool dispatch selected this skill for the agentic run",
                evidence_ids=[evidence_id] if isinstance(evidence_id, str) else [],
            )
        )
    return records


def _build_skill_utility_records(
    events: list,
    selection_records: list[SkillSelectionRecord],
) -> list[SkillUtilityRecord]:
    selected_skills = sorted({record.selected_skill for record in selection_records})
    return [
        update_skill_utility_from_trace(
            SkillUtilityRecord(skill_name=skill_name),
            events,
        )
        for skill_name in selected_skills
    ]


def _record_quality_trace_events(
    ledger: TraceLedger,
    latest: DAGNodeResult,
    state: CogAlphaState,
) -> None:
    if latest.node_name != "quality_pipeline":
        return

    quality_candidate_ids = [
        candidate.candidate_id
        for candidate in [
            *latest.candidates,
            *state.rejected_pool,
        ]
        if candidate.stage
        in {
            CandidateStage.ACCEPTED_BY_QUALITY,
            CandidateStage.REJECTED_BY_QUALITY,
        }
    ]
    for candidate_id in quality_candidate_ids:
        ledger.record(
            TraceEventKind.GUARD_REPORT_RECORDED,
            payload={
                "candidate_id": candidate_id,
                "status": "pass",
                "node_name": latest.node_name,
            },
        )


def _reset_output_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _write_trace_manifest(path: Path, *, run_id: str, trace_events: int) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "trace_path": "trace.jsonl",
                "trace_events": trace_events,
                "verification_path": "trace_verification.json",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_trace_verification(path: Path, report: TraceVerificationReport) -> None:
    payload = report.model_dump(mode="json")
    payload["passed"] = report.passed
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
