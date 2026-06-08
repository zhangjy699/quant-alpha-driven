"""Alpha Candidate lifecycle and pool transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cogalpha.fitness import composite_fitness_score
from cogalpha.schemas import (
    AlphaCandidate,
    CandidateStage,
    EvolutionOperation,
    FitnessDecision,
    FitnessMetrics,
)

PRIMARY_FITNESS_FIELDS = ("ic", "rank_ic", "icir", "rank_icir")


@dataclass(frozen=True)
class FitnessClassification:
    """Candidates after applying Fitness Gate decisions."""

    qualified: list[AlphaCandidate]
    elite: list[AlphaCandidate]
    rejected: list[AlphaCandidate]


def record_domain_generation(
    candidate: AlphaCandidate,
    *,
    skill_name: str,
    generation: int,
    guidance_mode: str | None = None,
) -> AlphaCandidate:
    """Attach Domain Agent Skill lineage to a generated Alpha Candidate."""

    updated = candidate.model_copy(deep=True)
    updated.lineage.agent_skill = updated.lineage.agent_skill or skill_name
    updated.lineage.generation = generation
    if guidance_mode is not None:
        updated.lineage.guidance_mode = updated.lineage.guidance_mode or guidance_mode
    return updated


def record_repair(candidate: AlphaCandidate) -> AlphaCandidate:
    """Mark a repaired Alpha Candidate without mutating the original artifact."""

    return _with_stage(candidate, CandidateStage.REPAIRED)


def record_quality_acceptance(candidate: AlphaCandidate) -> AlphaCandidate:
    """Mark an Alpha Candidate accepted by the Quality Pipeline."""

    return _with_stage(candidate, CandidateStage.ACCEPTED_BY_QUALITY)


def record_quality_rejection(candidate: AlphaCandidate) -> AlphaCandidate:
    """Mark an Alpha Candidate rejected by the Quality Pipeline."""

    return _with_stage(candidate, CandidateStage.REJECTED_BY_QUALITY)


def record_evolution_child(
    child: AlphaCandidate,
    *,
    operation: EvolutionOperation,
    parents: Sequence[AlphaCandidate],
    generation: int,
    lineage_parent_ids: Sequence[str] | None = None,
) -> AlphaCandidate:
    """Attach Thinking Evolution lineage to a child Alpha Candidate."""

    updated = child.model_copy(deep=True)
    updated.lineage.operation = updated.lineage.operation or operation
    if lineage_parent_ids is not None:
        updated.lineage.parent_ids = list(lineage_parent_ids)
    elif not updated.lineage.parent_ids:
        updated.lineage.parent_ids = [parent.candidate_id for parent in parents]
    updated.lineage.generation = generation
    return updated


def classify_candidates_by_fitness(
    candidates: Sequence[AlphaCandidate],
    fitness_decisions: Sequence[FitnessDecision],
) -> FitnessClassification:
    """Apply Fitness Gate decisions to Alpha Candidates and partition pools."""

    decision_by_id = {decision.candidate_id: decision for decision in fitness_decisions}
    qualified: list[AlphaCandidate] = []
    elite: list[AlphaCandidate] = []
    rejected: list[AlphaCandidate] = []

    for candidate in candidates:
        decision = decision_by_id.get(candidate.candidate_id)
        updated = record_fitness_decision(candidate, decision)
        if decision is None:
            rejected.append(updated)
        elif decision.stage == CandidateStage.ELITE:
            elite.append(updated)
            qualified.append(updated)
        elif decision.stage == CandidateStage.QUALIFIED:
            qualified.append(updated)
        else:
            rejected.append(updated)

    return FitnessClassification(qualified=qualified, elite=elite, rejected=rejected)


def record_fitness_decision(
    candidate: AlphaCandidate,
    decision: FitnessDecision | None,
) -> AlphaCandidate:
    """Attach one Fitness Gate decision to an Alpha Candidate."""

    if decision is None:
        return _with_stage(candidate, CandidateStage.REJECTED_BY_FITNESS)

    updated = _with_stage(candidate, decision.stage)
    updated.metadata["fitness_metrics"] = decision.metrics.model_dump(mode="python")
    return updated


def select_parent_pool(
    *,
    qualified: Sequence[AlphaCandidate],
    existing_elites: Sequence[AlphaCandidate],
    promising_rejected: Sequence[AlphaCandidate] = (),
    parent_pool_size: int,
    elite_carry_forward: int,
) -> list[AlphaCandidate]:
    """Select the parent-only pool used by evolution."""

    elite_carry = sorted(
        existing_elites,
        key=lambda candidate: composite_fitness_score(candidate_metrics(candidate)),
        reverse=True,
    )[:elite_carry_forward]
    promising = sorted(
        promising_rejected,
        key=lambda candidate: composite_fitness_score(candidate_metrics(candidate)),
        reverse=True,
    )
    return dedupe_candidates(list(elite_carry) + list(qualified) + promising)[:parent_pool_size]


def select_promising_rejected_parents(
    rejected: Sequence[AlphaCandidate],
    *,
    min_primary_metrics: int,
    min_composite_score: float,
) -> list[AlphaCandidate]:
    """Return rejected-by-fitness candidates that are useful only as evolution parents."""

    return [
        candidate
        for candidate in rejected
        if _is_promising_rejected_parent(
            candidate,
            min_primary_metrics=min_primary_metrics,
            min_composite_score=min_composite_score,
        )
    ]


def dedupe_candidates(candidates: Sequence[AlphaCandidate]) -> list[AlphaCandidate]:
    """Deduplicate Alpha Candidates by candidate id while preserving last write."""

    deduped: dict[str, AlphaCandidate] = {}
    for candidate in candidates:
        deduped[candidate.candidate_id] = candidate
    return list(deduped.values())


def candidate_metrics(candidate: AlphaCandidate) -> FitnessMetrics | None:
    """Return Fitness Metrics stored on an Alpha Candidate, when available."""

    raw_metrics = candidate.metadata.get("fitness_metrics")
    if raw_metrics is None:
        return None
    return FitnessMetrics.model_validate(raw_metrics)


def _is_promising_rejected_parent(
    candidate: AlphaCandidate,
    *,
    min_primary_metrics: int,
    min_composite_score: float,
) -> bool:
    if candidate.stage != CandidateStage.REJECTED_BY_FITNESS:
        return False
    metrics = candidate_metrics(candidate)
    if metrics is None:
        return False
    primary_passes = sum(1 for field in PRIMARY_FITNESS_FIELDS if getattr(metrics, field) > 0)
    return (
        primary_passes >= min_primary_metrics
        and composite_fitness_score(metrics) > min_composite_score
    )


def _with_stage(candidate: AlphaCandidate, stage: CandidateStage) -> AlphaCandidate:
    updated = candidate.model_copy(deep=True)
    updated.stage = stage
    return updated
