import pandas as pd

from cogalpha.config import FitnessGateConfig
from cogalpha.fitness import apply_fitness_gate, compute_predictive_metrics
from cogalpha.schemas import CandidateStage, FitnessMetrics


def test_compute_predictive_metrics_from_panel_data():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=["a", "b", "c"],
    )
    returns = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]],
        index=factor.index,
        columns=factor.columns,
    )

    metrics = compute_predictive_metrics(factor, returns)

    assert metrics.ic == 1.0
    assert metrics.rank_ic == 1.0
    assert metrics.icir == 0.0
    assert metrics.rank_icir == 0.0
    assert metrics.mi >= 0.0


def test_apply_fitness_gate_uses_percentiles_and_minima():
    decisions = apply_fitness_gate(
        {
            "strong": FitnessMetrics(ic=0.03, rank_ic=0.03, icir=0.3, rank_icir=0.3, mi=0.04),
            "usable": FitnessMetrics(ic=0.01, rank_ic=0.01, icir=0.1, rank_icir=0.1, mi=0.03),
            "weak": FitnessMetrics(ic=0.0, rank_ic=0.0, icir=0.0, rank_icir=0.0, mi=0.0),
        },
        FitnessGateConfig(qualified_percentile=0.0, elite_percentile=1.0),
    )

    stages = {decision.candidate_id: decision.stage for decision in decisions}
    assert stages["strong"] == CandidateStage.ELITE
    assert stages["usable"] == CandidateStage.QUALIFIED
    assert stages["weak"] == CandidateStage.REJECTED_BY_FITNESS
