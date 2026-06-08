"""Fitness gate node for evaluated alpha candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from cogalpha.candidate_lifecycle import (
    classify_candidates_by_fitness,
    dedupe_candidates,
    select_parent_pool,
    select_promising_rejected_parents,
)
from cogalpha.config import MVPLoopConfig
from cogalpha.feedback import build_generation_feedback
from cogalpha.fitness import apply_fitness_gate
from cogalpha.schemas import (
    AlphaCandidate,
    CandidateEvaluationResult,
    CogAlphaState,
    DAGNodeResult,
    FitnessMetrics,
)


class CandidateMetricsProvider(Protocol):
    """Evaluates alpha candidates against market data."""

    def evaluate(self, candidates: Sequence[AlphaCandidate]) -> Mapping[str, FitnessMetrics]:
        """Return metrics keyed by candidate id."""


@dataclass
class FitnessGateNode:
    """Apply the paper-defined five-metric selection policy."""

    config: MVPLoopConfig
    metrics_provider: CandidateMetricsProvider | None = None

    def __call__(self, state: dict) -> dict:
        parsed = CogAlphaState.model_validate(state)
        candidates = list(parsed.candidates)
        evaluation_results = self._collect_evaluation_results(candidates)
        metrics_by_id = {
            result.candidate_id: result.metrics
            for result in evaluation_results
            if result.metrics is not None
        }
        direction_by_id = {
            result.candidate_id: result.fitness_direction
            for result in evaluation_results
            if result.metrics is not None
        }
        decisions = apply_fitness_gate(metrics_by_id, self.config.experiment.fitness_gate)
        classification = classify_candidates_by_fitness(candidates, decisions)
        _attach_fitness_directions(classification.qualified, direction_by_id)
        _attach_fitness_directions(classification.elite, direction_by_id)
        _attach_fitness_directions(classification.rejected, direction_by_id)

        parsed.feedback = build_generation_feedback(
            generation=parsed.generation,
            candidates=classification.qualified + classification.rejected,
            fitness_decisions=decisions,
        )
        parsed.candidates = []
        parsed.rejected_pool.extend(classification.rejected)
        parsed.elite_pool = dedupe_candidates(parsed.elite_pool + classification.elite)
        parsed.qualified_pool = select_parent_pool(
            qualified=classification.qualified,
            existing_elites=parsed.elite_pool,
            promising_rejected=(),
            parent_pool_size=self.config.parent_pool_size,
            elite_carry_forward=self.config.elite_carry_forward,
        )
        promising_rejected = select_promising_rejected_parents(
            classification.rejected,
            min_primary_metrics=(
                self.config.experiment.fitness_gate.promising_parent_min_primary_metrics
            ),
            min_composite_score=(
                self.config.experiment.fitness_gate.promising_parent_min_composite
            ),
        )
        parsed.parent_pool = select_parent_pool(
            qualified=classification.qualified,
            existing_elites=parsed.elite_pool,
            promising_rejected=promising_rejected,
            parent_pool_size=self.config.parent_pool_size,
            elite_carry_forward=self.config.elite_carry_forward,
        )
        parsed.node_history.append(
            DAGNodeResult(
                node_name="fitness_gate",
                candidates=classification.qualified,
                fitness_decisions=decisions,
                evaluation_results=evaluation_results,
                metadata={
                    "evaluated": len(decisions),
                    "evaluation_results": len(evaluation_results),
                    "qualified": len(classification.qualified),
                    "elite": len(classification.elite),
                    "rejected": len(classification.rejected),
                    "promising_rejected_parents": len(promising_rejected),
                    "parent_pool": len(parsed.parent_pool),
                },
            )
        )
        return parsed.model_dump(mode="python")

    def _collect_metrics(self, candidates: Sequence[AlphaCandidate]) -> dict[str, FitnessMetrics]:
        if self.metrics_provider is not None and hasattr(
            self.metrics_provider,
            "evaluate_candidates",
        ):
            return {
                result.candidate_id: result.metrics
                for result in self.metrics_provider.evaluate_candidates(candidates)
                if result.metrics is not None
            }
        if self.metrics_provider is not None:
            return dict(self.metrics_provider.evaluate(candidates))

        metrics_by_id: dict[str, FitnessMetrics] = {}
        for candidate in candidates:
            raw_metrics = candidate.metadata.get("fitness_metrics")
            if raw_metrics is None:
                continue
            metrics_by_id[candidate.candidate_id] = FitnessMetrics.model_validate(raw_metrics)
        return metrics_by_id

    def _collect_evaluation_results(
        self,
        candidates: Sequence[AlphaCandidate],
    ) -> list[CandidateEvaluationResult]:
        if self.metrics_provider is not None and hasattr(
            self.metrics_provider,
            "evaluate_candidates",
        ):
            return list(self.metrics_provider.evaluate_candidates(candidates))
        metrics_by_id = self._collect_metrics(candidates)
        return [
            CandidateEvaluationResult(candidate_id=candidate_id, metrics=metrics)
            for candidate_id, metrics in metrics_by_id.items()
        ]


def _attach_fitness_directions(
    candidates: Sequence[AlphaCandidate],
    direction_by_id: Mapping[str, int],
) -> None:
    for candidate in candidates:
        direction = direction_by_id.get(candidate.candidate_id)
        if direction is not None:
            candidate.metadata["fitness_direction"] = direction
