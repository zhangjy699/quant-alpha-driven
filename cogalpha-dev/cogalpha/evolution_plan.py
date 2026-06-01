"""Thinking Evolution planning for the CogAlpha MVP Loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cogalpha.schemas import AlphaCandidate, EvolutionOperation, EvolutionSkillRequest

MUTATION_SKILL = "alpha-mutation"
CROSSOVER_SKILL = "alpha-crossover"


@dataclass(frozen=True)
class EvolutionSkillPlan:
    """One Evolution Operator Skill invocation and its lineage intent."""

    skill_name: str
    operation: EvolutionOperation
    parents: tuple[AlphaCandidate, ...]
    generation: int
    lineage_parent_ids: tuple[str, ...] | None = None

    def to_request(
        self,
        *,
        effective_feedback_summary: str | None,
        ineffective_feedback_summary: str | None,
    ) -> EvolutionSkillRequest:
        """Build the Runtime Schema payload for this evolution step."""

        return EvolutionSkillRequest(
            operation=self.operation,
            parents=list(self.parents),
            generation=self.generation,
            effective_feedback_summary=effective_feedback_summary,
            ineffective_feedback_summary=ineffective_feedback_summary,
        )


def select_evolution_parents(
    qualified_pool: Sequence[AlphaCandidate],
    parent_pool_size: int,
) -> list[AlphaCandidate]:
    """Choose the bounded Parent Pool slice used by Thinking Evolution."""

    return list(qualified_pool[:parent_pool_size])


def build_initial_evolution_plan(
    parents: Sequence[AlphaCandidate],
    generation: int,
) -> list[EvolutionSkillPlan]:
    """Plan mutation and crossover operations in the paper-defined MVP order."""

    plans = [
        EvolutionSkillPlan(
            skill_name=MUTATION_SKILL,
            operation=EvolutionOperation.MUTATION,
            parents=(parent,),
            generation=generation,
        )
        for parent in parents
    ]
    plans.extend(
        EvolutionSkillPlan(
            skill_name=CROSSOVER_SKILL,
            operation=EvolutionOperation.CROSSOVER,
            parents=(left, right),
            generation=generation,
        )
        for left, right in adjacent_parent_pairs(parents)
    )
    return plans


def build_crossover_then_mutation_plan(
    crossover_child: AlphaCandidate,
    original_parents: Sequence[AlphaCandidate],
    generation: int,
) -> EvolutionSkillPlan:
    """Plan mutation of a crossover child while preserving original parent ids."""

    return EvolutionSkillPlan(
        skill_name=MUTATION_SKILL,
        operation=EvolutionOperation.CROSSOVER_THEN_MUTATION,
        parents=(crossover_child,),
        generation=generation,
        lineage_parent_ids=tuple(parent.candidate_id for parent in original_parents),
    )


def adjacent_parent_pairs(
    candidates: Sequence[AlphaCandidate],
) -> list[tuple[AlphaCandidate, AlphaCandidate]]:
    """Pair adjacent Parent Pool entries for crossover."""

    return list(zip(candidates[0::2], candidates[1::2], strict=False))
