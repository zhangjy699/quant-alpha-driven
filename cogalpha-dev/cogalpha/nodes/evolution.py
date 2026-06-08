"""Thinking-evolution node for mutation and crossover skills."""

from __future__ import annotations

from dataclasses import dataclass

from cogalpha.candidate_lifecycle import record_evolution_child
from cogalpha.config import MVPLoopConfig
from cogalpha.evolution_plan import (
    EvolutionSkillPlan,
    build_crossover_then_mutation_plan,
    build_initial_evolution_plan,
    select_evolution_parents,
)
from cogalpha.schemas import (
    AlphaCandidate,
    CogAlphaState,
    DAGNodeResult,
    EvolutionOperation,
)
from cogalpha.skill_nodes import SkillNodeRuntime, StructuredArtifactInvoker


@dataclass
class EvolutionNode:
    """Generate children through mutation, crossover, and crossover-then-mutation."""

    invoker: StructuredArtifactInvoker
    config: MVPLoopConfig

    def __call__(self, state: dict) -> dict:
        parsed = CogAlphaState.model_validate(state)
        parent_candidates = parsed.parent_pool or parsed.qualified_pool
        parents = select_evolution_parents(parent_candidates, self.config.parent_pool_size)
        next_generation = parsed.generation + 1
        children: list[AlphaCandidate] = []
        errors: list[dict[str, str]] = []
        skill_runtime = SkillNodeRuntime(self.invoker)

        for plan in build_initial_evolution_plan(parents, next_generation):
            child = self._invoke_child(
                plan=plan,
                skill_runtime=skill_runtime,
                effective_feedback_summary=parsed.feedback.effective_feedback_summary,
                ineffective_feedback_summary=parsed.feedback.ineffective_feedback_summary,
                errors=errors,
            )
            if child is not None:
                children.append(child)
            if child is not None and plan.operation == EvolutionOperation.CROSSOVER:
                mutated_plan = build_crossover_then_mutation_plan(
                    child,
                    plan.parents,
                    next_generation,
                )
                mutated_child = self._invoke_child(
                    plan=mutated_plan,
                    skill_runtime=skill_runtime,
                    effective_feedback_summary=parsed.feedback.effective_feedback_summary,
                    ineffective_feedback_summary=parsed.feedback.ineffective_feedback_summary,
                    errors=errors,
                )
                if mutated_child is not None:
                    children.append(mutated_child)

        parsed.generation = next_generation
        parsed.candidates = children
        parsed.node_history.append(
            DAGNodeResult(
                node_name="thinking_evolution",
                candidates=children,
                metadata={
                    "parents": len(parents),
                    "children": len(children),
                    "errors": errors,
                },
            )
        )
        return parsed.model_dump(mode="python")

    def _invoke_child(
        self,
        *,
        plan: EvolutionSkillPlan,
        skill_runtime: SkillNodeRuntime,
        effective_feedback_summary: str | None,
        ineffective_feedback_summary: str | None,
        errors: list[dict[str, str]],
    ) -> AlphaCandidate | None:
        request = plan.to_request(
            effective_feedback_summary=effective_feedback_summary,
            ineffective_feedback_summary=ineffective_feedback_summary,
        )
        try:
            child = skill_runtime.alpha_candidate(plan.skill_name, request)
        except Exception as exc:  # noqa: BLE001 - graph records per-skill failures
            errors.append(
                {"skill": plan.skill_name, "operation": plan.operation.value, "error": str(exc)}
            )
            return None

        return record_evolution_child(
            child,
            operation=plan.operation,
            parents=plan.parents,
            generation=plan.generation,
            lineage_parent_ids=plan.lineage_parent_ids,
        )
