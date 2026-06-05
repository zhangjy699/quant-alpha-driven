"""Export formal MVP factor pools from final-state artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogalpha.registry import DOMAIN_AGENT_SPECS
from cogalpha.schemas import AlphaCandidate, CandidateStage, CogAlphaState, FitnessMetrics

UNKNOWN_DOMAIN = "unknown_domain"
POOL_NAMES = ("elite", "qualified", "rejected")
METRIC_FIELDS = ("ic", "rank_ic", "icir", "rank_icir", "mi")
DEFAULT_FACTOR_POOL_ROOT = Path("outputs/factor_pool")
FACTOR_SUMMARY_FIELDS = {
    "run_id",
    "factor_name",
    "formula",
    "code",
    "rationale",
    "required_columns",
    "allowed_libraries",
    "metrics",
}


@dataclass(frozen=True)
class FactorPoolExportResult:
    """Summary of one factor-pool export."""

    run_dir: Path
    factor_pool_dir: Path
    counts: dict[str, int]
    index_path: Path


def export_factor_pool(
    run_dir: str | Path,
    *,
    output_root: str | Path = DEFAULT_FACTOR_POOL_ROOT,
    rejected_limit: int = 3,
) -> FactorPoolExportResult:
    """Export factor-pool JSON files for one formal MVP run directory."""

    run_path = Path(run_dir)
    state_path = run_path / "final_state.json"
    state = CogAlphaState.model_validate_json(state_path.read_text(encoding="utf-8"))
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    index = _load_factor_pool_index(index_path)
    next_factor_id = int(index["next_factor_id"])

    all_candidates = _all_candidates(state)
    candidate_by_id = _candidate_lookup(all_candidates)
    initial_domain_by_id = _initial_domain_lookup(state)
    selected = _select_pool_candidates(state, rejected_limit=rejected_limit)
    counts = {pool: 0 for pool in POOL_NAMES}
    exported_entries: list[dict[str, Any]] = []

    for pool_name, candidates in selected.items():
        for candidate in candidates:
            domains = resolve_domain_agents(candidate, candidate_by_id, initial_domain_by_id)
            summary = build_factor_summary(
                state,
                candidate,
            )
            for domain in domains:
                factor_id = next_factor_id
                next_factor_id += 1
                path = output_dir / pool_name / _safe_path_component(domain) / (
                    f"{factor_id}.json"
                )
                _write_json(path, summary)
                counts[pool_name] += 1
                exported_entries.append(
                    {
                        "factor_id": factor_id,
                        "file": str(path.relative_to(output_dir)),
                        "pool": pool_name,
                        "domain_agent": domain,
                        "run_id": state.metadata.get("run_id"),
                        "candidate_id": candidate.candidate_id,
                        "factor_name": candidate.alpha.name,
                    }
                )

    index["next_factor_id"] = next_factor_id
    index["factors"].extend(exported_entries)
    index["counts"] = _index_counts(index["factors"])
    _write_json(index_path, index)
    return FactorPoolExportResult(
        run_dir=run_path,
        factor_pool_dir=output_dir,
        counts=counts,
        index_path=index_path,
    )


def build_factor_summary(
    state: CogAlphaState,
    candidate: AlphaCandidate,
) -> dict[str, Any]:
    """Build the public JSON summary for one exported factor."""

    metrics = _candidate_metrics(candidate)
    return {
        "run_id": state.metadata.get("run_id"),
        "factor_name": candidate.alpha.name,
        "formula": candidate.alpha.formula,
        "code": candidate.alpha.code,
        "rationale": candidate.alpha.rationale,
        "required_columns": list(candidate.alpha.required_columns),
        "allowed_libraries": list(candidate.alpha.allowed_libraries),
        "metrics": None if metrics is None else metrics.model_dump(mode="json"),
    }


def resolve_domain_agents(
    candidate: AlphaCandidate,
    candidate_by_id: dict[str, AlphaCandidate],
    initial_domain_by_id: dict[str, str],
) -> list[str]:
    """Resolve domain-agent attribution through parent lineage."""

    domains = _resolve_domain_agents(candidate, candidate_by_id, initial_domain_by_id, set())
    if not domains:
        return [UNKNOWN_DOMAIN]
    return sorted(domains)


def _select_pool_candidates(
    state: CogAlphaState,
    *,
    rejected_limit: int,
) -> dict[str, list[AlphaCandidate]]:
    elite_ids = {candidate.candidate_id for candidate in state.elite_pool}
    rejected = [
        candidate
        for candidate in state.rejected_pool
        if candidate.stage == CandidateStage.REJECTED_BY_FITNESS
        and _candidate_metrics(candidate) is not None
    ]
    rejected = sorted(
        rejected,
        key=lambda candidate: composite_fitness_score(_candidate_metrics(candidate)),
    )
    return {
        "elite": list(state.elite_pool),
        "qualified": [
            candidate
            for candidate in state.qualified_pool
            if candidate.candidate_id not in elite_ids
        ],
        "rejected": rejected[:rejected_limit],
    }


def _all_candidates(state: CogAlphaState) -> list[AlphaCandidate]:
    candidates: list[AlphaCandidate] = []
    for node in state.node_history:
        candidates.extend(node.candidates)
    candidates.extend(state.qualified_pool)
    candidates.extend(state.elite_pool)
    candidates.extend(state.rejected_pool)
    candidates.extend(state.candidates)
    return candidates


def _candidate_lookup(candidates: list[AlphaCandidate]) -> dict[str, AlphaCandidate]:
    by_id: dict[str, AlphaCandidate] = {}
    for candidate in candidates:
        by_id.setdefault(candidate.candidate_id, candidate)
    return by_id


def _initial_domain_lookup(state: CogAlphaState) -> dict[str, str]:
    domain_skill_names = {spec.skill_name for spec in DOMAIN_AGENT_SPECS}
    domains: dict[str, str] = {}
    for node in state.node_history:
        for candidate in node.candidates:
            agent_skill = candidate.lineage.agent_skill
            if agent_skill in domain_skill_names:
                domains.setdefault(candidate.candidate_id, agent_skill)
    for candidate in state.qualified_pool + state.elite_pool + state.rejected_pool:
        agent_skill = candidate.lineage.agent_skill
        if agent_skill in domain_skill_names:
            domains.setdefault(candidate.candidate_id, agent_skill)
    return domains


def _resolve_domain_agents(
    candidate: AlphaCandidate,
    candidate_by_id: dict[str, AlphaCandidate],
    initial_domain_by_id: dict[str, str],
    seen: set[str],
) -> set[str]:
    if candidate.candidate_id in seen:
        return set()
    seen.add(candidate.candidate_id)

    direct_domain = initial_domain_by_id.get(candidate.candidate_id)
    if direct_domain is not None:
        return {direct_domain}

    domains: set[str] = set()
    for parent_id in candidate.lineage.parent_ids:
        if parent_id == candidate.candidate_id:
            parent = candidate_by_id.get(parent_id)
            if parent is not None and parent.lineage.parent_ids != candidate.lineage.parent_ids:
                for grandparent_id in parent.lineage.parent_ids:
                    parent_domain = initial_domain_by_id.get(grandparent_id)
                    if parent_domain is not None:
                        domains.add(parent_domain)
                        continue
                    grandparent = candidate_by_id.get(grandparent_id)
                    if grandparent is not None:
                        domains.update(
                            _resolve_domain_agents(
                                grandparent,
                                candidate_by_id,
                                initial_domain_by_id,
                                seen,
                            )
                        )
            continue
        parent_domain = initial_domain_by_id.get(parent_id)
        if parent_domain is not None:
            domains.add(parent_domain)
            continue
        parent = candidate_by_id.get(parent_id)
        if parent is not None:
            domains.update(
                _resolve_domain_agents(parent, candidate_by_id, initial_domain_by_id, seen)
            )

    if domains:
        return domains
    agent_skill = candidate.lineage.agent_skill
    if agent_skill and agent_skill not in {
        "alpha-mutation",
        "alpha-crossover",
        "alpha-code-repair",
    }:
        return {agent_skill}
    return set()


def _candidate_metrics(candidate: AlphaCandidate) -> FitnessMetrics | None:
    raw_metrics = candidate.metadata.get("fitness_metrics")
    if raw_metrics is None:
        return None
    return FitnessMetrics.model_validate(raw_metrics)


def composite_fitness_score(metrics: FitnessMetrics | None) -> float:
    """Return the project convention used for simple factor ordering."""

    if metrics is None:
        return float("-inf")
    return float(sum(getattr(metrics, field) for field in METRIC_FIELDS))


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or UNKNOWN_DOMAIN


def _load_factor_pool_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"next_factor_id": 0, "counts": {}, "factors": []}
    index = json.loads(path.read_text(encoding="utf-8"))
    if "next_factor_id" not in index or "factors" not in index:
        raise ValueError(f"Invalid factor pool index: {path}")
    index.setdefault("counts", {})
    return index


def _index_counts(factors: list[dict[str, Any]]) -> dict[str, Any]:
    by_pool = {pool: 0 for pool in POOL_NAMES}
    by_domain: dict[str, int] = {}
    for entry in factors:
        pool = entry["pool"]
        domain = entry["domain_agent"]
        by_pool[pool] = by_pool.get(pool, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1
    return {
        "total": len(factors),
        "by_pool": by_pool,
        "by_domain_agent": by_domain,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
