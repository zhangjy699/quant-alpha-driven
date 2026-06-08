from cogalpha.candidate_lifecycle import (
    classify_candidates_by_fitness,
    record_domain_generation,
    record_evolution_child,
    record_quality_rejection,
    select_parent_pool,
    select_promising_rejected_parents,
)
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateStage,
    EvolutionOperation,
    FitnessDecision,
    FitnessMetrics,
)


def make_candidate(candidate_id: str) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id}",
            code=f"def factor_{candidate_id}(df):\n    return df['close'] - df['open']\n",
            rationale=f"{candidate_id} rationale.",
        ),
    )


def make_decision(candidate_id: str, stage: CandidateStage, score: float) -> FitnessDecision:
    metrics = FitnessMetrics(ic=score, rank_ic=score, icir=score, rank_icir=score, mi=score)
    return FitnessDecision(
        candidate_id=candidate_id,
        metrics=metrics,
        stage=stage,
        qualified_thresholds=metrics,
        elite_thresholds=metrics,
    )


def test_lifecycle_records_lineage_and_stage_without_mutating_original():
    candidate = make_candidate("alpha_1")

    generated = record_domain_generation(
        candidate,
        skill_name="alpha-market-cycle",
        generation=2,
    )
    rejected = record_quality_rejection(generated)

    assert candidate.stage == CandidateStage.GENERATED
    assert candidate.lineage.agent_skill is None
    assert generated.lineage.agent_skill == "alpha-market-cycle"
    assert generated.lineage.generation == 2
    assert rejected.stage == CandidateStage.REJECTED_BY_QUALITY


def test_lifecycle_classifies_fitness_and_selects_parent_pool():
    strong = make_candidate("strong")
    usable = make_candidate("usable")
    weak = make_candidate("weak")

    classification = classify_candidates_by_fitness(
        [strong, usable, weak],
        [
            make_decision("strong", CandidateStage.ELITE, 0.3),
            make_decision("usable", CandidateStage.QUALIFIED, 0.1),
            make_decision("weak", CandidateStage.REJECTED_BY_FITNESS, 0.0),
        ],
    )
    parent_pool = select_parent_pool(
        qualified=classification.qualified,
        existing_elites=classification.elite,
        parent_pool_size=2,
        elite_carry_forward=1,
    )

    assert [candidate.candidate_id for candidate in classification.qualified] == [
        "strong",
        "usable",
    ]
    assert [candidate.candidate_id for candidate in classification.rejected] == ["weak"]
    assert [candidate.candidate_id for candidate in parent_pool] == ["strong", "usable"]


def test_lifecycle_uses_promising_rejected_only_for_parent_pool():
    strong = make_candidate("strong")
    promising = make_candidate("promising")
    weak = make_candidate("weak")

    classification = classify_candidates_by_fitness(
        [strong, promising, weak],
        [
            make_decision("strong", CandidateStage.QUALIFIED, 0.2),
            make_decision("promising", CandidateStage.REJECTED_BY_FITNESS, 0.03),
            make_decision("weak", CandidateStage.REJECTED_BY_FITNESS, -0.1),
        ],
    )
    promising_rejected = select_promising_rejected_parents(
        classification.rejected,
        min_primary_metrics=2,
        min_composite_score=0.0,
    )
    parent_pool = select_parent_pool(
        qualified=classification.qualified,
        existing_elites=classification.elite,
        promising_rejected=promising_rejected,
        parent_pool_size=3,
        elite_carry_forward=1,
    )

    assert [candidate.candidate_id for candidate in classification.qualified] == ["strong"]
    assert [candidate.candidate_id for candidate in promising_rejected] == ["promising"]
    assert [candidate.candidate_id for candidate in parent_pool] == ["strong", "promising"]
    assert parent_pool[1].stage == CandidateStage.REJECTED_BY_FITNESS


def test_lifecycle_records_evolution_child_with_original_crossover_parent_ids():
    left = make_candidate("left")
    right = make_candidate("right")
    crossover_child = make_candidate("child")

    child = record_evolution_child(
        crossover_child,
        operation=EvolutionOperation.CROSSOVER_THEN_MUTATION,
        parents=[crossover_child],
        generation=3,
        lineage_parent_ids=[left.candidate_id, right.candidate_id],
    )

    assert child.lineage.operation == EvolutionOperation.CROSSOVER_THEN_MUTATION
    assert child.lineage.parent_ids == ["left", "right"]
    assert child.lineage.generation == 3
