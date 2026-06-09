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


def test_qualified_gate_uses_minima_and_composite_percentile():
    decisions = apply_fitness_gate(
        {
            "balanced": FitnessMetrics(
                ic=0.02,
                rank_ic=0.02,
                icir=0.20,
                rank_icir=0.20,
                mi=0.02,
            ),
            "lopsided": FitnessMetrics(
                ic=0.05,
                rank_ic=0.011,
                icir=0.40,
                rank_icir=0.081,
                mi=0.006,
            ),
            "below_minima": FitnessMetrics(
                ic=0.20,
                rank_ic=0.001,
                icir=0.50,
                rank_icir=0.50,
                mi=0.05,
            ),
        },
        FitnessGateConfig(qualified_percentile=0.5, elite_percentile=1.0),
    )

    stages = {decision.candidate_id: decision.stage for decision in decisions}
    assert stages["balanced"] == CandidateStage.REJECTED_BY_FITNESS
    assert stages["lopsided"] == CandidateStage.QUALIFIED
    assert stages["below_minima"] == CandidateStage.REJECTED_BY_FITNESS
