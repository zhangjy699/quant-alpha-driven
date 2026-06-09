"""Panel-backed candidate evaluation for the Fitness Gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cogalpha.data import MarketDataSplit
from cogalpha.fitness import compute_predictive_metrics
from cogalpha.guards.alpha_runtime import run_runtime_alpha_code_guard_with_values
from cogalpha.schemas import (
    AlphaCandidate,
    CandidateEvaluationResult,
    FitnessMetrics,
    GuardReport,
    GuardStatus,
)

FITNESS_DIRECTION_POLICY = "auto_flip_directional_metrics_v2"


@dataclass(frozen=True)
class EvaluationCacheRecord:
    """One public evaluation cache record."""

    cache_key: str
    candidate_id: str
    alpha_fingerprint: str
    data_version: str
    split_name: str | None
    metrics: FitnessMetrics | None
    guard_report: GuardReport | None
    raw_metrics: FitnessMetrics | None = None
    error: str | None = None
    fitness_direction: int = 1


@dataclass(frozen=True)
class EvaluationCache:
    """JSONL cache for deterministic candidate evaluation artifacts."""

    path: Path | str

    def get(
        self,
        candidate: AlphaCandidate,
        *,
        data_version: str,
        split_name: str | None = None,
        max_nan_fraction: float,
    ) -> EvaluationCacheRecord | None:
        """Return the cached record for one candidate/evaluation setting, if present."""

        cache_key = build_evaluation_cache_key(
            candidate,
            data_version=data_version,
            split_name=split_name,
            max_nan_fraction=max_nan_fraction,
        )
        for record in reversed(self.load_all()):
            if record.cache_key == cache_key:
                return record
        return None

    def put(self, record: EvaluationCacheRecord) -> None:
        """Persist a cache record unless this key is already present."""

        if any(existing.cache_key == record.cache_key for existing in self.load_all()):
            return
        cache_path = Path(self.path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record_to_json(record), sort_keys=True) + "\n")

    def load_all(self) -> list[EvaluationCacheRecord]:
        """Load all cache records in insertion order."""

        cache_path = Path(self.path)
        if not cache_path.exists():
            return []
        records: list[EvaluationCacheRecord] = []
        for raw_line in cache_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            records.append(_record_from_json(json.loads(raw_line)))
        return records


def build_evaluation_cache_key(
    candidate: AlphaCandidate,
    *,
    data_version: str,
    split_name: str | None = None,
    max_nan_fraction: float,
) -> str:
    """Return a stable key for deterministic candidate evaluation."""

    payload = {
        "candidate_id": candidate.candidate_id,
        "alpha_fingerprint": alpha_fingerprint(candidate),
        "data_version": data_version,
        "split_name": split_name,
        "max_nan_fraction": max_nan_fraction,
        "fitness_direction_policy": FITNESS_DIRECTION_POLICY,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def alpha_fingerprint(candidate: AlphaCandidate) -> str:
    """Return a stable fingerprint for an Alpha Candidate's executable contract."""

    payload = {
        "alpha": candidate.alpha.model_dump(mode="json"),
        "lineage": candidate.lineage.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class PanelBackedMetricsProvider:
    """Evaluate Alpha Candidates against OHLCV Input and forward returns."""

    ohlcv_panel: pd.DataFrame
    forward_returns: pd.DataFrame
    data_version: str = "unversioned"
    split_name: str | None = None
    max_nan_fraction: float = 0.30
    cache: EvaluationCache | None = None
    guard_reports_by_candidate_id: dict[str, GuardReport] = field(default_factory=dict)
    errors_by_candidate_id: dict[str, str] = field(default_factory=dict)
    cache_hits_by_candidate_id: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_split(
        cls,
        data_split: MarketDataSplit,
        *,
        data_version: str = "unversioned",
        max_nan_fraction: float = 0.30,
        cache: EvaluationCache | None = None,
    ) -> PanelBackedMetricsProvider:
        """Build a metrics provider from one prepared market-data split."""

        return cls(
            ohlcv_panel=data_split.ohlcv_panel,
            forward_returns=data_split.forward_returns,
            data_version=data_version,
            split_name=data_split.name,
            max_nan_fraction=max_nan_fraction,
            cache=cache,
        )

    def evaluate(self, candidates: Sequence[AlphaCandidate]) -> Mapping[str, FitnessMetrics]:
        """Return five-metric fitness scores for candidates that pass runtime guards."""

        return {
            result.candidate_id: result.metrics
            for result in self.evaluate_candidates(candidates)
            if result.metrics is not None
        }

    def evaluate_candidates(
        self,
        candidates: Sequence[AlphaCandidate],
    ) -> list[CandidateEvaluationResult]:
        """Return structured evaluation results for every candidate."""

        metrics_by_id: dict[str, FitnessMetrics] = {}
        results: list[CandidateEvaluationResult] = []
        self.guard_reports_by_candidate_id.clear()
        self.errors_by_candidate_id.clear()
        self.cache_hits_by_candidate_id.clear()

        for candidate in candidates:
            cached = self._cached_record(candidate)
            if cached is not None:
                self.cache_hits_by_candidate_id[candidate.candidate_id] = True
                if cached.guard_report is not None:
                    self.guard_reports_by_candidate_id[candidate.candidate_id] = cached.guard_report
                if cached.error is not None:
                    self.errors_by_candidate_id[candidate.candidate_id] = cached.error
                if cached.metrics is not None:
                    metrics_by_id[candidate.candidate_id] = cached.metrics
                results.append(
                    CandidateEvaluationResult(
                        candidate_id=candidate.candidate_id,
                        metrics=cached.metrics,
                        raw_metrics=cached.raw_metrics,
                        guard_report=cached.guard_report,
                        error=cached.error,
                        cache_hit=True,
                        data_version=cached.data_version,
                        fitness_direction=cached.fitness_direction,
                    )
                )
                continue

            self.cache_hits_by_candidate_id[candidate.candidate_id] = False
            guard_result = run_runtime_alpha_code_guard_with_values(
                candidate,
                self.ohlcv_panel,
                max_nan_fraction=self.max_nan_fraction,
            )
            guard_report = guard_result.report
            self.guard_reports_by_candidate_id[candidate.candidate_id] = guard_report
            if guard_report.status == GuardStatus.FAIL:
                self._cache_record(candidate, metrics=None, guard_report=guard_report)
                results.append(
                    CandidateEvaluationResult(
                        candidate_id=candidate.candidate_id,
                        guard_report=guard_report,
                        error="runtime guard failed",
                        cache_hit=False,
                        data_version=self.data_version,
                    )
                )
                continue

            if guard_result.factor_values is None:
                raise RuntimeError("Runtime guard passed without reusable factor values.")
            raw_metrics = compute_predictive_metrics(
                guard_result.factor_values,
                self.forward_returns,
            )
            metrics_by_id[candidate.candidate_id] = _choose_directional_metrics(raw_metrics)
            fitness_direction = _fitness_direction(raw_metrics)
            self._cache_record(
                candidate,
                metrics=metrics_by_id[candidate.candidate_id],
                raw_metrics=raw_metrics,
                guard_report=guard_report,
                fitness_direction=fitness_direction,
            )
            results.append(
                CandidateEvaluationResult(
                    candidate_id=candidate.candidate_id,
                    metrics=metrics_by_id[candidate.candidate_id],
                    raw_metrics=raw_metrics,
                    guard_report=guard_report,
                    cache_hit=False,
                    data_version=self.data_version,
                    fitness_direction=fitness_direction,
                )
            )

        return results

    def _cached_record(self, candidate: AlphaCandidate) -> EvaluationCacheRecord | None:
        if self.cache is None:
            return None
        return self.cache.get(
            candidate,
            data_version=self.data_version,
            split_name=self.split_name,
            max_nan_fraction=self.max_nan_fraction,
        )

    def _cache_record(
        self,
        candidate: AlphaCandidate,
        *,
        metrics: FitnessMetrics | None,
        guard_report: GuardReport | None,
        raw_metrics: FitnessMetrics | None = None,
        error: str | None = None,
        fitness_direction: int = 1,
    ) -> None:
        if self.cache is None:
            return
        record = EvaluationCacheRecord(
            cache_key=build_evaluation_cache_key(
                candidate,
                data_version=self.data_version,
                split_name=self.split_name,
                max_nan_fraction=self.max_nan_fraction,
            ),
            candidate_id=candidate.candidate_id,
            alpha_fingerprint=alpha_fingerprint(candidate),
            data_version=self.data_version,
            split_name=self.split_name,
            metrics=metrics,
            raw_metrics=raw_metrics,
            guard_report=guard_report,
            error=error,
            fitness_direction=fitness_direction,
        )
        self.cache.put(record)


def _record_to_json(record: EvaluationCacheRecord) -> dict[str, Any]:
    return {
        "cache_key": record.cache_key,
        "candidate_id": record.candidate_id,
        "alpha_fingerprint": record.alpha_fingerprint,
        "data_version": record.data_version,
        "split_name": record.split_name,
        "metrics": record.metrics.model_dump(mode="json") if record.metrics is not None else None,
        "raw_metrics": (
            record.raw_metrics.model_dump(mode="json")
            if record.raw_metrics is not None
            else None
        ),
        "guard_report": (
            record.guard_report.model_dump(mode="json")
            if record.guard_report is not None
            else None
        ),
        "error": record.error,
        "fitness_direction": record.fitness_direction,
    }


def _record_from_json(raw: dict[str, Any]) -> EvaluationCacheRecord:
    metrics = raw.get("metrics")
    raw_metrics = raw.get("raw_metrics")
    guard_report = raw.get("guard_report")
    return EvaluationCacheRecord(
        cache_key=raw["cache_key"],
        candidate_id=raw["candidate_id"],
        alpha_fingerprint=raw["alpha_fingerprint"],
        data_version=raw["data_version"],
        split_name=raw.get("split_name"),
        metrics=FitnessMetrics.model_validate(metrics) if metrics is not None else None,
        raw_metrics=(
            FitnessMetrics.model_validate(raw_metrics)
            if raw_metrics is not None
            else None
        ),
        guard_report=GuardReport.model_validate(guard_report) if guard_report is not None else None,
        error=raw.get("error"),
        fitness_direction=int(raw.get("fitness_direction", 1)),
    )


def _choose_directional_metrics(metrics: FitnessMetrics) -> FitnessMetrics:
    flipped = _flip_metric_direction(metrics)
    if _direction_score(flipped) > _direction_score(metrics):
        return flipped
    return metrics


def _fitness_direction(metrics: FitnessMetrics) -> int:
    flipped = _flip_metric_direction(metrics)
    return -1 if _direction_score(flipped) > _direction_score(metrics) else 1


def _flip_metric_direction(metrics: FitnessMetrics) -> FitnessMetrics:
    return FitnessMetrics(
        ic=-metrics.ic,
        rank_ic=-metrics.rank_ic,
        icir=-metrics.icir,
        rank_icir=-metrics.rank_icir,
        mi=metrics.mi,
    )


def _direction_score(metrics: FitnessMetrics) -> float:
    return metrics.ic + metrics.rank_ic + metrics.icir + metrics.rank_icir
