"""Incremental factor-memory updates from the shared factor pool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cogalpha.factor_pool import METRIC_FIELDS, POOL_NAMES, DEFAULT_FACTOR_POOL_ROOT
from cogalpha.registry import PROJECT_ROOT
from cogalpha.skill_invocation import SkillInvoker
from cogalpha.skill_loader import StandardSkillLoader

DEFAULT_FACTOR_MEMORY_ROOT = Path("outputs/factor_memory")
MAX_PATTERNS_PER_KIND = 20
RETRIEVAL_PATTERNS_PER_KIND = 2
RETRIEVAL_REGIME_HYPOTHESES = 1
MEMORY_SUMMARIZER_SKILL = "alpha-memory-summarizer"


class FactorMemoryPattern(BaseModel):
    """One compressed factor-memory lesson."""

    model_config = ConfigDict(extra="forbid")

    lesson: str = Field(..., min_length=1)
    evidence_factor_ids: list[int] = Field(default_factory=list)


class FactorMemoryRegimeHypothesis(BaseModel):
    """A weak market-style hypothesis derived from bounded validation evidence."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(..., min_length=1)
    confidence: Literal["low", "medium"] = "low"
    evidence_factor_ids: list[int] = Field(default_factory=list)
    risk: str = Field(..., min_length=1)


class FactorMemoryEvidence(BaseModel):
    """Bounded evidence passed to the memory summarizer skill."""

    model_config = ConfigDict(extra="forbid")

    factor_id: int = Field(..., ge=0)
    pool: str = Field(..., min_length=1)
    domain_agent: str = Field(..., min_length=1)
    factor_name: str = Field(..., min_length=1)
    formula: str | None = None
    rationale: str = Field(..., min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_bottlenecks: list[str] = Field(default_factory=list)
    split: str | None = None
    validation_outcome: str | None = None
    validation_metrics: dict[str, float] = Field(default_factory=dict)


class FactorMemoryCompactionRequest(BaseModel):
    """Strict runtime payload for LLM-assisted memory compaction."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = MEMORY_SUMMARIZER_SKILL
    domain_agent: str = Field(..., min_length=1)
    new_evidence: list[FactorMemoryEvidence] = Field(default_factory=list)
    success_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    failure_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    avoid_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    regime_hypotheses: list[FactorMemoryRegimeHypothesis] = Field(default_factory=list)
    metric_bottlenecks: dict[str, int] = Field(default_factory=dict)
    max_patterns_per_kind: int = Field(default=MAX_PATTERNS_PER_KIND, ge=1)


class FactorMemoryCompactionResult(BaseModel):
    """Schema-valid memory patterns returned by the summarizer skill."""

    model_config = ConfigDict(extra="forbid")

    success_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    failure_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    avoid_patterns: list[FactorMemoryPattern] = Field(default_factory=list)
    regime_hypotheses: list[FactorMemoryRegimeHypothesis] = Field(default_factory=list)


@dataclass(frozen=True)
class FactorMemorySummarizer:
    """Invoke the standard memory summarizer skill with bounded evidence."""

    invoker: SkillInvoker
    skill_name: str = MEMORY_SUMMARIZER_SKILL

    def summarize(
        self,
        request: FactorMemoryCompactionRequest,
    ) -> FactorMemoryCompactionResult:
        return self.invoker.invoke(
            self.skill_name,
            runtime_payload=request.model_dump_json(indent=2),
            output_schema=FactorMemoryCompactionResult,
        )


@dataclass(frozen=True)
class FactorMemoryUpdateResult:
    """Summary of one incremental memory update."""

    factor_pool_root: Path
    memory_root: Path
    processed_factor_ids: list[int]
    domain_updates: dict[str, int]
    state_path: Path


@dataclass(frozen=True)
class FactorMemoryValidationUpdateResult:
    """Summary of one validation-audit memory update."""

    audit_path: Path
    memory_root: Path
    processed_audit_keys: list[str]
    domain_updates: dict[str, int]
    state_path: Path


def update_factor_memory(
    *,
    factor_pool_root: str | Path = DEFAULT_FACTOR_POOL_ROOT,
    memory_root: str | Path = DEFAULT_FACTOR_MEMORY_ROOT,
    summarizer: FactorMemorySummarizer | None = None,
    max_patterns_per_kind: int = MAX_PATTERNS_PER_KIND,
) -> FactorMemoryUpdateResult:
    """Update compressed domain memories from newly exported factor-pool entries."""

    pool_root = Path(factor_pool_root)
    memory_path = Path(memory_root)
    index = _read_json(pool_root / "index.json")
    state_path = memory_path / "state.json"
    state = _load_memory_state(state_path)
    last_processed_factor_id = int(state["last_processed_factor_id"])

    new_entries = [
        entry
        for entry in index.get("factors", [])
        if int(entry["factor_id"]) >= last_processed_factor_id
    ]
    new_entries.sort(key=lambda entry: int(entry["factor_id"]))

    processed_factor_ids: list[int] = []
    domain_updates: dict[str, int] = {}
    evidence_by_domain: dict[str, list[FactorMemoryEvidence]] = {}
    for entry in new_entries:
        factor_id = int(entry["factor_id"])
        factor = _read_json(pool_root / entry["file"])
        domain = str(entry["domain_agent"])
        memory = _load_domain_memory(memory_path, domain)
        updated = _update_domain_memory(
            memory,
            entry=entry,
            factor=factor,
            max_patterns_per_kind=max_patterns_per_kind,
        )
        evidence_by_domain.setdefault(domain, []).append(
            _build_evidence(entry=entry, factor=factor)
        )
        _write_domain_memory(memory_path, domain, updated)
        processed_factor_ids.append(factor_id)
        domain_updates[domain] = domain_updates.get(domain, 0) + 1

    if summarizer is not None:
        for domain, evidence in evidence_by_domain.items():
            memory = _load_domain_memory(memory_path, domain)
            compacted = _summarize_domain_memory(
                memory,
                evidence=evidence,
                summarizer=summarizer,
                max_patterns_per_kind=max_patterns_per_kind,
            )
            _write_domain_memory(memory_path, domain, compacted)

    if processed_factor_ids:
        state["last_processed_factor_id"] = max(processed_factor_ids) + 1
        state["updated_at"] = _utc_now()
        state["processed_factor_count"] = int(state.get("processed_factor_count", 0)) + len(
            processed_factor_ids
        )
        state["domain_updates"] = _merge_domain_updates(
            dict(state.get("domain_updates", {})),
            domain_updates,
        )
        _write_json(state_path, state)

    _write_retrieval_caches(memory_path)
    return FactorMemoryUpdateResult(
        factor_pool_root=pool_root,
        memory_root=memory_path,
        processed_factor_ids=processed_factor_ids,
        domain_updates=domain_updates,
        state_path=state_path,
    )


def update_factor_memory_from_validation_audit(
    *,
    audit_path: str | Path,
    memory_root: str | Path = DEFAULT_FACTOR_MEMORY_ROOT,
    summarizer: FactorMemorySummarizer | None = None,
    max_patterns_per_kind: int = MAX_PATTERNS_PER_KIND,
) -> FactorMemoryValidationUpdateResult:
    """Update compressed memory with validation-set success/failure lessons."""

    audit_file = Path(audit_path)
    memory_path = Path(memory_root)
    report = _read_json(audit_file)
    report_split = str(report["split"])
    if report_split == "test":
        raise ValueError("Test split audit reports must not update generation memory.")

    state_path = memory_path / "state.json"
    state = _load_memory_state(state_path)
    processed_keys = set(state.get("processed_validation_audit_keys", []))
    newly_processed: list[str] = []
    domain_updates: dict[str, int] = {}
    evidence_by_domain: dict[str, list[FactorMemoryEvidence]] = {}

    for record in report.get("records", []):
        factor_id = int(record["factor_id"])
        audit_key = _validation_audit_key(str(report["run_id"]), report_split, factor_id)
        if audit_key in processed_keys:
            continue
        domain = str(record["domain_agent"])
        memory = _load_domain_memory(memory_path, domain)
        _update_domain_memory_from_validation_record(
            memory,
            split=report_split,
            record=dict(record),
            max_patterns_per_kind=max_patterns_per_kind,
        )
        evidence_by_domain.setdefault(domain, []).append(
            _build_validation_evidence(record=dict(record), split=report_split)
        )
        _write_domain_memory(memory_path, domain, memory)
        processed_keys.add(audit_key)
        newly_processed.append(audit_key)
        domain_updates[domain] = domain_updates.get(domain, 0) + 1

    if summarizer is not None:
        for domain, evidence in evidence_by_domain.items():
            memory = _load_domain_memory(memory_path, domain)
            compacted = _summarize_domain_memory(
                memory,
                evidence=evidence,
                summarizer=summarizer,
                max_patterns_per_kind=max_patterns_per_kind,
            )
            _write_domain_memory(memory_path, domain, compacted)

    if newly_processed:
        state["processed_validation_audit_keys"] = sorted(processed_keys)
        state["updated_at"] = _utc_now()
        state["validation_audit_count"] = (
            int(state.get("validation_audit_count", 0)) + len(newly_processed)
        )
        state["validation_domain_updates"] = _merge_domain_updates(
            dict(state.get("validation_domain_updates", {})),
            domain_updates,
        )
        _write_json(state_path, state)

    _write_retrieval_caches(memory_path)
    return FactorMemoryValidationUpdateResult(
        audit_path=audit_file,
        memory_root=memory_path,
        processed_audit_keys=newly_processed,
        domain_updates=domain_updates,
        state_path=state_path,
    )


def build_factor_memory_summarizer(
    client,
    *,
    inline_references: bool = False,
    skills_root: str | Path = PROJECT_ROOT / "skills",
) -> FactorMemorySummarizer:
    """Build the standard LLM-backed memory summarizer."""

    return FactorMemorySummarizer(
        SkillInvoker(
            loader=StandardSkillLoader(skills_root),
            client=client,
            inline_references=inline_references,
        )
    )


def build_prior_lessons(
    domain_memory: dict[str, Any],
    *,
    patterns_per_kind: int = RETRIEVAL_PATTERNS_PER_KIND,
    regime_hypotheses: int = RETRIEVAL_REGIME_HYPOTHESES,
) -> str:
    """Build a compact prompt block for one domain agent."""

    lines = [f"# Prior Lessons for {domain_memory['skill_name']}"]
    sections = [
        ("Effective", domain_memory.get("success_patterns", [])),
        ("Ineffective", domain_memory.get("failure_patterns", [])),
        ("Avoid", domain_memory.get("avoid_patterns", [])),
    ]
    for title, patterns in sections:
        selected = patterns[:patterns_per_kind]
        if not selected:
            continue
        lines.append("")
        lines.append(f"## {title}")
        for pattern in selected:
            evidence = ", ".join(str(item) for item in pattern.get("evidence_factor_ids", []))
            suffix = f" Evidence factor IDs: {evidence}." if evidence else ""
            lines.append(f"- {pattern['lesson']}{suffix}")
    hypotheses = domain_memory.get("regime_hypotheses", [])[:regime_hypotheses]
    if hypotheses:
        lines.append("")
        lines.append("## Regime Hypotheses")
        for hypothesis in hypotheses:
            evidence = ", ".join(
                str(item) for item in hypothesis.get("evidence_factor_ids", [])
            )
            suffix = f" Evidence factor IDs: {evidence}." if evidence else ""
            lines.append(
                f"- Hypothesis: {hypothesis['hypothesis']} "
                f"Confidence: {hypothesis.get('confidence', 'low')}. "
                f"Risk: {hypothesis['risk']} "
                "Use as inspiration only; keep mechanisms diverse."
                f"{suffix}"
            )
    return "\n".join(lines).strip() + "\n"


def _update_domain_memory(
    memory: dict[str, Any],
    *,
    entry: dict[str, Any],
    factor: dict[str, Any],
    max_patterns_per_kind: int,
) -> dict[str, Any]:
    factor_id = int(entry["factor_id"])
    if factor_id in set(memory.get("processed_factor_ids", [])):
        return memory

    pool = str(entry["pool"])
    metrics = factor.get("metrics") or {}
    bottlenecks = _metric_bottlenecks(metrics)
    memory["updated_at"] = _utc_now()
    memory["processed_factor_ids"].append(factor_id)
    memory["pool_counts"][pool] = int(memory["pool_counts"].get(pool, 0)) + 1

    if pool in {"elite", "qualified"}:
        _append_pattern(
            memory["success_patterns"],
            _success_lesson(entry, factor, pool),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )
    elif pool == "rejected":
        for metric_name in bottlenecks:
            memory["metric_bottlenecks"][metric_name] = (
                int(memory["metric_bottlenecks"].get(metric_name, 0)) + 1
            )
        _append_pattern(
            memory["failure_patterns"],
            _failure_lesson(entry, factor, bottlenecks),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )
        _append_pattern(
            memory["avoid_patterns"],
            _avoid_lesson(factor, bottlenecks),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )

    return memory


def _update_domain_memory_from_validation_record(
    memory: dict[str, Any],
    *,
    split: str,
    record: dict[str, Any],
    max_patterns_per_kind: int,
) -> None:
    factor_id = int(record["factor_id"])
    outcome = str(record["validation_outcome"])
    memory["updated_at"] = _utc_now()
    memory["validation_counts"][outcome] = (
        int(memory["validation_counts"].get(outcome, 0)) + 1
    )

    bottlenecks = list(record.get("metric_bottlenecks") or [])
    for metric_name in bottlenecks:
        memory["validation_metric_bottlenecks"][metric_name] = (
            int(memory["validation_metric_bottlenecks"].get(metric_name, 0)) + 1
        )

    if outcome == "validation_success":
        _append_pattern(
            memory["success_patterns"],
            _validation_success_lesson(record, split),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )
    elif outcome == "validation_failure":
        _append_pattern(
            memory["failure_patterns"],
            _validation_failure_lesson(record, split),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )
        _append_pattern(
            memory["avoid_patterns"],
            _validation_avoid_lesson(record, split),
            factor_id,
            max_patterns=max_patterns_per_kind,
        )


def _summarize_domain_memory(
    memory: dict[str, Any],
    *,
    evidence: list[FactorMemoryEvidence],
    summarizer: FactorMemorySummarizer,
    max_patterns_per_kind: int,
) -> dict[str, Any]:
    request = FactorMemoryCompactionRequest(
        domain_agent=memory["skill_name"],
        new_evidence=evidence,
        success_patterns=[
            FactorMemoryPattern.model_validate(pattern)
            for pattern in memory.get("success_patterns", [])
        ],
        failure_patterns=[
            FactorMemoryPattern.model_validate(pattern)
            for pattern in memory.get("failure_patterns", [])
        ],
        avoid_patterns=[
            FactorMemoryPattern.model_validate(pattern)
            for pattern in memory.get("avoid_patterns", [])
        ],
        regime_hypotheses=[
            FactorMemoryRegimeHypothesis.model_validate(hypothesis)
            for hypothesis in memory.get("regime_hypotheses", [])
        ],
        metric_bottlenecks=dict(memory.get("metric_bottlenecks", {})),
        max_patterns_per_kind=max_patterns_per_kind,
    )
    try:
        result = summarizer.summarize(request)
    except (ValidationError, ValueError, RuntimeError, PermissionError):
        return memory

    allowed_factor_ids = _allowed_request_factor_ids(request)
    memory["success_patterns"] = _validated_patterns(
        result.success_patterns,
        allowed_factor_ids=allowed_factor_ids,
        max_patterns=max_patterns_per_kind,
        fallback=memory.get("success_patterns", []),
    )
    memory["failure_patterns"] = _validated_patterns(
        result.failure_patterns,
        allowed_factor_ids=allowed_factor_ids,
        max_patterns=max_patterns_per_kind,
        fallback=memory.get("failure_patterns", []),
    )
    memory["avoid_patterns"] = _validated_patterns(
        result.avoid_patterns,
        allowed_factor_ids=allowed_factor_ids,
        max_patterns=max_patterns_per_kind,
        fallback=memory.get("avoid_patterns", []),
    )
    memory["regime_hypotheses"] = _validated_regime_hypotheses(
        result.regime_hypotheses,
        allowed_factor_ids=allowed_factor_ids,
        fallback=memory.get("regime_hypotheses", []),
    )
    memory["updated_at"] = _utc_now()
    return memory


def _allowed_request_factor_ids(request: FactorMemoryCompactionRequest) -> set[int]:
    allowed = {item.factor_id for item in request.new_evidence}
    for patterns in (
        request.success_patterns,
        request.failure_patterns,
        request.avoid_patterns,
    ):
        for pattern in patterns:
            allowed.update(pattern.evidence_factor_ids)
    for hypothesis in request.regime_hypotheses:
        allowed.update(hypothesis.evidence_factor_ids)
    return allowed


def _validated_patterns(
    patterns: list[FactorMemoryPattern],
    *,
    allowed_factor_ids: set[int],
    max_patterns: int,
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for pattern in patterns:
        evidence = [
            factor_id
            for factor_id in pattern.evidence_factor_ids
            if factor_id in allowed_factor_ids
        ]
        if not evidence:
            continue
        validated.append(
            {
                "lesson": pattern.lesson,
                "evidence_factor_ids": evidence,
            }
        )
        if len(validated) >= max_patterns:
            break
    if validated:
        return validated
    return list(fallback)[:max_patterns]


def _validated_regime_hypotheses(
    hypotheses: list[FactorMemoryRegimeHypothesis],
    *,
    allowed_factor_ids: set[int],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        evidence = [
            factor_id
            for factor_id in hypothesis.evidence_factor_ids
            if factor_id in allowed_factor_ids
        ]
        if not evidence:
            continue
        validated.append(
            {
                "hypothesis": hypothesis.hypothesis,
                "confidence": hypothesis.confidence,
                "evidence_factor_ids": evidence,
                "risk": hypothesis.risk,
            }
        )
        if len(validated) >= 2:
            break
    if validated:
        return validated
    return list(fallback)[:2]


def _build_evidence(entry: dict[str, Any], factor: dict[str, Any]) -> FactorMemoryEvidence:
    metrics = {
        field: float(factor["metrics"][field])
        for field in METRIC_FIELDS
        if factor.get("metrics") and factor["metrics"].get(field) is not None
    }
    return FactorMemoryEvidence(
        factor_id=int(entry["factor_id"]),
        pool=str(entry["pool"]),
        domain_agent=str(entry["domain_agent"]),
        factor_name=str(factor.get("factor_name") or entry["factor_name"]),
        formula=factor.get("formula"),
        rationale=str(factor.get("rationale") or "No rationale provided."),
        metrics=metrics,
        metric_bottlenecks=_metric_bottlenecks(metrics),
    )


def _build_validation_evidence(
    *,
    record: dict[str, Any],
    split: str,
) -> FactorMemoryEvidence:
    validation_metrics = {
        field: float(record["validation_metrics"][field])
        for field in METRIC_FIELDS
        if record.get("validation_metrics")
        and record["validation_metrics"].get(field) is not None
    }
    return FactorMemoryEvidence(
        factor_id=int(record["factor_id"]),
        pool=str(record["pool"]),
        domain_agent=str(record["domain_agent"]),
        factor_name=str(record["factor_name"]),
        rationale=f"{record['factor_name']} validation audit outcome.",
        metrics=dict(record.get("train_metrics") or {}),
        metric_bottlenecks=list(record.get("metric_bottlenecks") or []),
        split=split,
        validation_outcome=str(record["validation_outcome"]),
        validation_metrics=validation_metrics,
    )


def _success_lesson(entry: dict[str, Any], factor: dict[str, Any], pool: str) -> str:
    factor_name = str(factor.get("factor_name") or entry["factor_name"])
    mechanism = _short_text(factor.get("rationale") or factor.get("formula") or factor_name)
    return (
        f"{factor_name} reached {pool}; preserve the core mechanism: "
        f"{mechanism}"
    )


def _failure_lesson(
    entry: dict[str, Any],
    factor: dict[str, Any],
    bottlenecks: list[str],
) -> str:
    factor_name = str(factor.get("factor_name") or entry["factor_name"])
    weak = ", ".join(bottlenecks) if bottlenecks else "the full five-metric gate"
    mechanism = _short_text(factor.get("rationale") or factor.get("formula") or factor_name)
    return (
        f"{factor_name} was rejected; weakest metrics were {weak}. "
        f"Mechanism to improve or avoid: {mechanism}"
    )


def _avoid_lesson(factor: dict[str, Any], bottlenecks: list[str]) -> str:
    weak = ", ".join(bottlenecks) if bottlenecks else "weak validation metrics"
    formula = factor.get("formula") or factor.get("factor_name")
    return f"Avoid repeating {formula!r} without addressing {weak}."


def _validation_success_lesson(record: dict[str, Any], split: str) -> str:
    factor_name = str(record["factor_name"])
    return (
        f"{factor_name} passed {split} validation minima after qualifying; "
        "treat this mechanism as more likely to generalize out of sample."
    )


def _validation_failure_lesson(record: dict[str, Any], split: str) -> str:
    factor_name = str(record["factor_name"])
    weak = _weak_metric_text(record)
    return (
        f"{factor_name} qualified in the source run but failed {split} "
        f"validation minima; weak validation metrics were {weak}."
    )


def _validation_avoid_lesson(record: dict[str, Any], split: str) -> str:
    factor_name = str(record["factor_name"])
    weak = _weak_metric_text(record)
    return (
        f"Avoid relying on {factor_name} without improving out-of-sample "
        f"{split} robustness for {weak}."
    )


def _weak_metric_text(record: dict[str, Any]) -> str:
    bottlenecks = list(record.get("metric_bottlenecks") or [])
    if bottlenecks:
        return ", ".join(str(metric) for metric in bottlenecks)
    return "the five-metric validation gate"


def _append_pattern(
    patterns: list[dict[str, Any]],
    lesson: str,
    factor_id: int,
    *,
    max_patterns: int,
) -> None:
    for pattern in patterns:
        if pattern["lesson"] == lesson:
            evidence = pattern.setdefault("evidence_factor_ids", [])
            if factor_id not in evidence:
                evidence.append(factor_id)
            return
    patterns.insert(0, {"lesson": lesson, "evidence_factor_ids": [factor_id]})
    del patterns[max_patterns:]


def _metric_bottlenecks(metrics: dict[str, Any]) -> list[str]:
    numeric_metrics = {
        field: float(metrics[field])
        for field in METRIC_FIELDS
        if field in metrics and metrics[field] is not None
    }
    if not numeric_metrics:
        return []
    non_positive = [field for field, value in numeric_metrics.items() if value <= 0]
    if non_positive:
        return non_positive
    ordered = sorted(numeric_metrics, key=numeric_metrics.get)
    return ordered[:2]


def _load_memory_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_processed_factor_id": 0,
            "processed_factor_count": 0,
            "updated_at": None,
            "domain_updates": {},
            "processed_validation_audit_keys": [],
            "validation_audit_count": 0,
            "validation_domain_updates": {},
        }
    state = _read_json(path)
    state.setdefault("processed_validation_audit_keys", [])
    state.setdefault("validation_audit_count", 0)
    state.setdefault("validation_domain_updates", {})
    return state


def _load_domain_memory(memory_root: Path, skill_name: str) -> dict[str, Any]:
    path = _domain_memory_path(memory_root, skill_name)
    if path.exists():
        memory = _read_json(path)
        _ensure_domain_memory_defaults(memory)
        return memory
    return _new_domain_memory(skill_name)


def _new_domain_memory(skill_name: str) -> dict[str, Any]:
    return {
        "skill_name": skill_name,
        "updated_at": None,
        "processed_factor_ids": [],
        "pool_counts": {pool: 0 for pool in POOL_NAMES},
        "success_patterns": [],
        "failure_patterns": [],
        "metric_bottlenecks": {},
        "avoid_patterns": [],
        "regime_hypotheses": [],
        "validation_counts": {
            "validation_success": 0,
            "validation_failure": 0,
            "validation_error": 0,
        },
        "validation_metric_bottlenecks": {},
    }


def _ensure_domain_memory_defaults(memory: dict[str, Any]) -> None:
    memory.setdefault("processed_factor_ids", [])
    memory.setdefault("pool_counts", {pool: 0 for pool in POOL_NAMES})
    memory.setdefault("success_patterns", [])
    memory.setdefault("failure_patterns", [])
    memory.setdefault("metric_bottlenecks", {})
    memory.setdefault("avoid_patterns", [])
    memory.setdefault("regime_hypotheses", [])
    memory.setdefault(
        "validation_counts",
        {
            "validation_success": 0,
            "validation_failure": 0,
            "validation_error": 0,
        },
    )
    memory.setdefault("validation_metric_bottlenecks", {})


def _write_domain_memory(memory_root: Path, skill_name: str, memory: dict[str, Any]) -> None:
    _write_json(_domain_memory_path(memory_root, skill_name), memory)


def _write_retrieval_caches(memory_root: Path) -> None:
    domain_dir = memory_root / "domain_memory"
    if not domain_dir.exists():
        return
    cache_dir = memory_root / "retrieval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(domain_dir.glob("*.json")):
        memory = _read_json(path)
        (cache_dir / f"{memory['skill_name']}.md").write_text(
            build_prior_lessons(memory),
            encoding="utf-8",
        )


def _domain_memory_path(memory_root: Path, skill_name: str) -> Path:
    return memory_root / "domain_memory" / f"{skill_name}.json"


def _merge_domain_updates(
    existing: dict[str, Any],
    updates: dict[str, int],
) -> dict[str, int]:
    merged = {key: int(value) for key, value in existing.items()}
    for domain, count in updates.items():
        merged[domain] = merged.get(domain, 0) + count
    return merged


def _validation_audit_key(run_id: str, split: str, factor_id: int) -> str:
    return f"{run_id}:{split}:{factor_id}"


def _short_text(value: object, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
