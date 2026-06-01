"""Adaptive generation feedback built from quality and fitness outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cogalpha.fitness import composite_fitness_score
from cogalpha.schemas import (
    AlphaCandidate,
    CandidateStage,
    FeedbackPolarity,
    FeedbackSample,
    FitnessDecision,
    GenerationFeedback,
)

EFFECTIVE_STAGES = {CandidateStage.QUALIFIED, CandidateStage.ELITE}


def build_generation_feedback(
    *,
    generation: int,
    candidates: Sequence[AlphaCandidate],
    fitness_decisions: Sequence[FitnessDecision],
    effective_sample_size: int = 2,
    ineffective_sample_size: int = 2,
) -> GenerationFeedback:
    """Summarize valid and invalid alphas for the next generation."""

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    effective_decisions = [
        decision for decision in fitness_decisions if decision.stage in EFFECTIVE_STAGES
    ]
    ineffective_decisions = [
        decision
        for decision in fitness_decisions
        if decision.stage == CandidateStage.REJECTED_BY_FITNESS
    ]

    effective_decisions.sort(
        key=lambda decision: composite_fitness_score(decision.metrics),
        reverse=True,
    )
    ineffective_decisions.sort(key=lambda decision: composite_fitness_score(decision.metrics))

    effective_samples = [
        _sample_from_decision(
            decision,
            candidates_by_id,
            FeedbackPolarity.EFFECTIVE,
        )
        for decision in effective_decisions[:effective_sample_size]
    ]
    ineffective_samples = [
        _sample_from_decision(
            decision,
            candidates_by_id,
            FeedbackPolarity.INEFFECTIVE,
        )
        for decision in ineffective_decisions[:ineffective_sample_size]
    ]

    return GenerationFeedback(
        generation=generation,
        effective_samples=effective_samples,
        ineffective_samples=ineffective_samples,
        effective_feedback_summary=_join_samples(effective_samples),
        ineffective_feedback_summary=_join_samples(ineffective_samples),
    )


def _sample_from_decision(
    decision: FitnessDecision,
    candidates_by_id: Mapping[str, AlphaCandidate],
    polarity: FeedbackPolarity,
) -> FeedbackSample:
    candidate = candidates_by_id.get(decision.candidate_id)
    rationale = candidate.alpha.rationale if candidate is not None else "Candidate not retained."
    return FeedbackSample(
        candidate_id=decision.candidate_id,
        polarity=polarity,
        stage=decision.stage,
        metrics=decision.metrics,
        summary=(
            f"{decision.candidate_id}: {decision.stage.value}; "
            f"IC={decision.metrics.ic:.4f}, RankIC={decision.metrics.rank_ic:.4f}, "
            f"ICIR={decision.metrics.icir:.4f}, RankICIR={decision.metrics.rank_icir:.4f}, "
            f"MI={decision.metrics.mi:.4f}. Hypothesis: {rationale}"
        ),
    )


def _join_samples(samples: Sequence[FeedbackSample]) -> str | None:
    if not samples:
        return None
    return "\n".join(sample.summary for sample in samples)
