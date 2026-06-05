from cogalpha.evolution_plan import (
    build_crossover_then_mutation_plan,
    build_initial_evolution_plan,
)
from cogalpha.schemas import AlphaCandidate, AlphaFunction, EvolutionOperation


def make_candidate(candidate_id: str) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id}",
            code=f"def factor_{candidate_id}(df):\n    return df['close'] - df['open']\n",
            rationale=f"{candidate_id} rationale.",
        ),
    )


def test_evolution_plan_preserves_mvp_operation_order():
    parents = [make_candidate("a"), make_candidate("b")]

    plans = build_initial_evolution_plan(parents, generation=1)

    assert [(plan.skill_name, plan.operation) for plan in plans] == [
        ("alpha-mutation", EvolutionOperation.MUTATION),
        ("alpha-mutation", EvolutionOperation.MUTATION),
        ("alpha-crossover", EvolutionOperation.CROSSOVER),
    ]


def test_crossover_then_mutation_plan_preserves_original_parent_ids():
    left = make_candidate("left")
    right = make_candidate("right")
    crossover_child = make_candidate("child")

    plan = build_crossover_then_mutation_plan(crossover_child, [left, right], generation=2)

    assert plan.parents == (crossover_child,)
    assert plan.lineage_parent_ids == ("left", "right")
    assert plan.operation == EvolutionOperation.CROSSOVER_THEN_MUTATION
