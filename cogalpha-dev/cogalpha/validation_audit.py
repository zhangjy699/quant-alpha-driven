"""Validation-set audit for qualified formal MVP factors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cogalpha.config import MVPLoopConfig
from cogalpha.data import DataSplitName, load_prepared_baseline_market_data
from cogalpha.evaluation import EvaluationCache, PanelBackedMetricsProvider
from cogalpha.factor_pool import DEFAULT_FACTOR_POOL_ROOT, METRIC_FIELDS
from cogalpha.schemas import (
    AlphaCandidate,
    CandidateEvaluationResult,
    CogAlphaState,
    FitnessMetrics,
    GuardStatus,
)

@dataclass(frozen=True)
class ValidationAuditResult:
    """Paths and counts for one validation audit run."""

    report_path: Path
    cache_path: Path
    counts: dict[str, int]


def validate_qualified_factors(
    *,
    run_dir: str | Path,
    data_dir: str | Path = "data/processed/csi300",
    split: DataSplitName = "valid",
    factor_pool_root: str | Path = DEFAULT_FACTOR_POOL_ROOT,
    metrics_provider: Any | None = None,
) -> ValidationAuditResult:
    """Audit qualified and elite factors on a split using existing five metrics."""

    run_path = Path(run_dir)
    state = CogAlphaState.model_validate_json(
        (run_path / "final_state.json").read_text(encoding="utf-8")
    )
    run_id = str(state.metadata.get("run_id") or run_path.name)
    loop_config = MVPLoopConfig()
    thresholds = loop_config.experiment.fitness_gate.qualified_minima
    provider = metrics_provider or _build_metrics_provider(
        run_path=run_path,
        data_dir=data_dir,
        split=split,
    )
    factor_entries = _factor_entries_by_candidate(
        factor_pool_root=Path(factor_pool_root),
        run_id=run_id,
    )
    candidates = _qualified_and_elite_candidates(state)
    missing = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id not in factor_entries
    ]
    if missing:
        raise ValueError(
            "Factor pool is missing exported factor IDs for candidate(s): "
            + ", ".join(sorted(missing))
        )

    results = {
        result.candidate_id: result
        for result in provider.evaluate_candidates(candidates)
    }
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        result = results.get(candidate.candidate_id)
        if result is None:
            result = CandidateEvaluationResult(
                candidate_id=candidate.candidate_id,
                error="missing validation evaluation result",
            )
        for entry in factor_entries[candidate.candidate_id]:
            records.append(
                _build_record(
                    candidate=candidate,
                    entry=entry,
                    result=result,
                    thresholds=thresholds,
                )
            )

    report = {
        "run_id": run_id,
        "split": split,
        "data_version": str(getattr(provider, "data_version", "unversioned")),
        "created_at": datetime.now(UTC).isoformat(),
        "validation_rule": "qualified_minima",
        "thresholds": _metrics_to_dict(thresholds),
        "records": records,
        "counts": _audit_counts(records),
    }
    report_path = run_path / f"validation_audit_{split}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ValidationAuditResult(
        report_path=report_path,
        cache_path=run_path / f"validation_evaluation_cache_{split}.jsonl",
        counts=report["counts"],
    )


def _build_metrics_provider(
    *,
    run_path: Path,
    data_dir: str | Path,
    split: DataSplitName,
) -> PanelBackedMetricsProvider:
    metadata = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    market_data = load_prepared_baseline_market_data(data_dir)
    cache = EvaluationCache(run_path / f"validation_evaluation_cache_{split}.jsonl")
    return PanelBackedMetricsProvider.from_split(
        market_data.split(split),
        data_version=metadata["data_version"],
        cache=cache,
    )


def _qualified_and_elite_candidates(state: CogAlphaState) -> list[AlphaCandidate]:
    candidates: list[AlphaCandidate] = []
    seen: set[str] = set()
    for candidate in [*state.elite_pool, *state.qualified_pool]:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        candidates.append(candidate)
    return candidates


def _factor_entries_by_candidate(
    *,
    factor_pool_root: Path,
    run_id: str,
) -> dict[str, list[dict[str, Any]]]:
    index_path = factor_pool_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries: dict[str, list[dict[str, Any]]] = {}
    for entry in index.get("factors", []):
        if str(entry.get("run_id")) != run_id:
            continue
        candidate_id = str(entry["candidate_id"])
        entries.setdefault(candidate_id, []).append(entry)
    return entries


def _build_record(
    *,
    candidate: AlphaCandidate,
    entry: dict[str, Any],
    result: CandidateEvaluationResult,
    thresholds: FitnessMetrics,
) -> dict[str, Any]:
    metrics = result.metrics
    if metrics is None:
        outcome = "validation_error"
        bottlenecks: list[str] = []
    else:
        bottlenecks = _threshold_bottlenecks(metrics, thresholds)
        outcome = "validation_failure" if bottlenecks else "validation_success"

    guard_status = result.guard_report.status if result.guard_report is not None else None
    error = result.error
    if metrics is None and error is None:
        error = "validation metrics missing"

    return {
        "factor_id": int(entry["factor_id"]),
        "candidate_id": candidate.candidate_id,
        "domain_agent": str(entry["domain_agent"]),
        "pool": str(entry["pool"]),
        "factor_name": candidate.alpha.name,
        "train_metrics": _candidate_train_metrics(candidate),
        "validation_metrics": None if metrics is None else _metrics_to_dict(metrics),
        "validation_outcome": outcome,
        "metric_bottlenecks": bottlenecks,
        "guard_status": None if guard_status is None else str(guard_status),
        "error": error,
        "cache_hit": result.cache_hit,
    }


def _threshold_bottlenecks(
    metrics: FitnessMetrics,
    thresholds: FitnessMetrics,
) -> list[str]:
    return [
        field
        for field in METRIC_FIELDS
        if getattr(metrics, field) < getattr(thresholds, field)
    ]


def _candidate_train_metrics(candidate: AlphaCandidate) -> dict[str, float] | None:
    raw_metrics = candidate.metadata.get("fitness_metrics")
    if raw_metrics is None:
        return None
    return _metrics_to_dict(FitnessMetrics.model_validate(raw_metrics))


def _metrics_to_dict(metrics: FitnessMetrics) -> dict[str, float]:
    return {field: float(getattr(metrics, field)) for field in METRIC_FIELDS}


def _audit_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "validation_success": 0,
        "validation_failure": 0,
        "validation_error": 0,
        "total": len(records),
    }
    for record in records:
        counts[str(record["validation_outcome"])] += 1
    return counts
