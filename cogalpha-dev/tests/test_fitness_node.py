from cogalpha.config import MVPLoopConfig
from cogalpha.nodes import FitnessGateNode
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateEvaluationResult,
    CandidateStage,
    CogAlphaState,
    FitnessMetrics,
    GuardIssue,
    GuardReport,
    GuardStatus,
)


def make_candidate(candidate_id: str, metrics: dict) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id}",
            code=f"def factor_{candidate_id}(df):\n    return df['close'] - df['open']\n",
            rationale=f"{candidate_id} rationale.",
        ),
        metadata={"fitness_metrics": metrics},
    )


def test_fitness_gate_node_updates_pools_and_feedback():
    candidates = [
        make_candidate(
            "strong",
            {"ic": 0.03, "rank_ic": 0.03, "icir": 0.3, "rank_icir": 0.3, "mi": 0.04},
        ),
        make_candidate(
            "usable",
            {"ic": 0.01, "rank_ic": 0.01, "icir": 0.1, "rank_icir": 0.1, "mi": 0.03},
        ),
        make_candidate(
            "weak",
            {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "mi": 0.0},
        ),
    ]
    config = MVPLoopConfig()
    config.experiment.fitness_gate.qualified_percentile = 0.0
    config.experiment.fitness_gate.elite_percentile = 1.0
    node = FitnessGateNode(config=config)

    result = CogAlphaState.model_validate(
        node(CogAlphaState(candidates=candidates).model_dump(mode="python"))
    )

    assert result.candidates == []
    assert [candidate.candidate_id for candidate in result.qualified_pool] == ["strong", "usable"]
    assert [candidate.candidate_id for candidate in result.parent_pool] == ["strong", "usable"]
    assert [candidate.candidate_id for candidate in result.elite_pool] == ["strong"]
    assert result.rejected_pool[0].candidate_id == "weak"
    assert result.qualified_pool[0].stage == CandidateStage.ELITE
    assert result.feedback.effective_feedback_summary is not None
    assert result.feedback.ineffective_feedback_summary is not None


def test_fitness_gate_node_records_structured_evaluation_results():
    good = make_candidate(
        "good",
        {"ic": 0.03, "rank_ic": 0.03, "icir": 0.3, "rank_icir": 0.3, "mi": 0.04},
    )
    bad = make_candidate(
        "bad",
        {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "rank_icir": 0.0, "mi": 0.0},
    )
    node = FitnessGateNode(
        config=MVPLoopConfig(),
        metrics_provider=StructuredMetricsProvider(),
    )

    result = CogAlphaState.model_validate(
        node(CogAlphaState(candidates=[good, bad]).model_dump(mode="python"))
    )

    history = result.node_history[-1]
    assert [entry.candidate_id for entry in history.evaluation_results] == ["good", "bad"]
    assert history.evaluation_results[0].metrics is not None
    assert history.evaluation_results[1].guard_report is not None
    assert history.evaluation_results[1].error == "runtime guard failed"
    assert result.rejected_pool[-1].candidate_id == "bad"


def test_fitness_gate_node_keeps_promising_rejected_out_of_qualified_pool():
    candidate = make_candidate(
        "rank_weak",
        {"ic": 0.03, "rank_ic": -0.01, "icir": 0.3, "rank_icir": -0.1, "mi": 0.04},
    )
    config = MVPLoopConfig(parent_pool_size=2)
    config.experiment.fitness_gate.qualified_percentile = 0.0
    config.experiment.fitness_gate.elite_percentile = 1.0
    node = FitnessGateNode(config=config)

    result = CogAlphaState.model_validate(
        node(CogAlphaState(candidates=[candidate]).model_dump(mode="python"))
    )

    assert result.qualified_pool == []
    assert [candidate.candidate_id for candidate in result.parent_pool] == ["rank_weak"]
    assert result.parent_pool[0].stage == CandidateStage.REJECTED_BY_FITNESS


class StructuredMetricsProvider:
    def evaluate_candidates(self, candidates):
        return [
            CandidateEvaluationResult(
                candidate_id=candidates[0].candidate_id,
                metrics=FitnessMetrics(ic=0.03, rank_ic=0.03, icir=0.3, rank_icir=0.3, mi=0.04),
                data_version="unit-test-v1",
            ),
            CandidateEvaluationResult(
                candidate_id=candidates[1].candidate_id,
                guard_report=GuardReport(
                    guard_name="runtime_alpha_code",
                    status=GuardStatus.FAIL,
                    issues=[
                        GuardIssue(
                            code="all_nan_output",
                            message="Alpha execution produced only NaN values.",
                        )
                    ],
                ),
                error="runtime guard failed",
                data_version="unit-test-v1",
            ),
        ]
