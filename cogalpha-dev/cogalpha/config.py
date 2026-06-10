"""Configuration objects for the first CogAlpha reproduction target."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from cogalpha.schemas import FitnessMetrics, OHLCVColumn


class SplitConfig(BaseModel):
    """Chronological train/validation/test split."""

    model_config = ConfigDict(extra="forbid")

    train_start: date = date(2018, 1, 1)
    train_end: date = date(2021, 12, 31)
    valid_start: date = date(2022, 1, 1)
    valid_end: date = date(2022, 12, 31)
    test_start: date = date(2023, 1, 1)
    test_end: date = date(2024, 12, 1)


class FitnessGateConfig(BaseModel):
    """Fitness threshold policy for the formal MVP route."""

    model_config = ConfigDict(extra="forbid")

    qualified_percentile: float = Field(default=0.50, ge=0, le=1)
    elite_percentile: float = Field(default=0.80, ge=0, le=1)
    qualified_minima: FitnessMetrics = Field(
        default_factory=lambda: FitnessMetrics(
            ic=0.01,
            rank_ic=0.01,
            icir=0.1,
            rank_icir=0.1,
            mi=0.005,
        )
    )
    elite_minima: FitnessMetrics = Field(
        default_factory=lambda: FitnessMetrics(
            ic=0.02,
            rank_ic=0.03,
            icir=0.2,
            rank_icir=0.25,
            mi=0.02,
        )
    )
    promising_parent_min_primary_metrics: int = Field(default=2, ge=0, le=4)
    promising_parent_min_composite: float = 0.0


class BaselineExperimentConfig(BaseModel):
    """First-version reproduction target."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = "company_all_a"
    horizon_days: int = 1
    return_price_column: OHLCVColumn = "open"
    trade_delay_days: int = Field(default=1, ge=0)
    split: SplitConfig = Field(default_factory=SplitConfig)
    fitness_gate: FitnessGateConfig = Field(default_factory=FitnessGateConfig)


class MVPLoopConfig(BaseModel):
    """Small-scale runnable loop parameters."""

    model_config = ConfigDict(extra="forbid")

    experiment: BaselineExperimentConfig = Field(default_factory=BaselineExperimentConfig)
    alphas_per_domain_agent: int = Field(default=1, ge=1)
    max_generations: int = Field(default=2, ge=1)
    max_repair_attempts: int = Field(default=2, ge=0)
    parent_pool_size: int = Field(default=8, ge=1)
    elite_carry_forward: int = Field(default=2, ge=0)
