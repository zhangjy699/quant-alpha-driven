import json

from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateEvaluationResult,
    CandidateStage,
    CogAlphaState,
    FitnessMetrics,
    GuardReport,
    GuardStatus,
)
from cogalpha.validation_audit import validate_qualified_factors


class FakeMetricsProvider:
    def __init__(self, results):
        self.results = results
        self.data_version = "fake-data-version"
        self.evaluated_candidate_ids = []

    def evaluate_candidates(self, candidates):
        self.evaluated_candidate_ids = [candidate.candidate_id for candidate in candidates]
        return [self.results[candidate.candidate_id] for candidate in candidates]


def test_validation_audit_validates_only_qualified_and_elite(tmp_path):
    run_dir = tmp_path / "outputs" / "experiments" / "formal-run"
    factor_pool = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir(parents=True)
    elite = _candidate("elite-1", CandidateStage.ELITE, score=0.2)
    qualified = _candidate("qualified-1", CandidateStage.QUALIFIED, score=0.1)
    rejected = _candidate("rejected-1", CandidateStage.REJECTED_BY_FITNESS, score=-0.1)
    state = CogAlphaState(
        metadata={"run_id": "formal-run"},
        elite_pool=[elite],
        qualified_pool=[elite, qualified],
        rejected_pool=[rejected],
    )
    (run_dir / "final_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_index(
        factor_pool,
        [
            _entry(0, "elite", "alpha-market-cycle", "elite-1"),
            _entry(1, "qualified", "alpha-range-vol", "qualified-1"),
        ],
    )
    provider = FakeMetricsProvider(
        {
            "elite-1": CandidateEvaluationResult(
                candidate_id="elite-1",
                metrics=_metrics(0.2),
            ),
            "qualified-1": CandidateEvaluationResult(
                candidate_id="qualified-1",
                metrics=FitnessMetrics(
                    ic=0.1,
                    rank_ic=0.1,
                    icir=0.1,
                    rank_icir=0.0,
                    mi=0.1,
                ),
            ),
        }
    )

    result = validate_qualified_factors(
        run_dir=run_dir,
        factor_pool_root=factor_pool,
        metrics_provider=provider,
    )

    assert provider.evaluated_candidate_ids == ["elite-1", "qualified-1"]
    assert result.report_path == run_dir / "validation_audit_valid.json"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {
        "total": 2,
        "validation_error": 0,
        "validation_failure": 1,
        "validation_success": 1,
    }
    records_by_candidate = {
        record["candidate_id"]: record for record in report["records"]
    }
    assert records_by_candidate["elite-1"]["factor_id"] == 0
    assert records_by_candidate["elite-1"]["validation_outcome"] == "validation_success"
    assert records_by_candidate["qualified-1"]["validation_outcome"] == (
        "validation_failure"
    )
    assert records_by_candidate["qualified-1"]["metric_bottlenecks"] == ["rank_icir"]
    assert not (run_dir / "factor_pool").exists()


def test_validation_audit_records_runtime_errors(tmp_path):
    run_dir = tmp_path / "outputs" / "experiments" / "formal-run"
    factor_pool = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir(parents=True)
    candidate = _candidate("qualified-1", CandidateStage.QUALIFIED, score=0.1)
    state = CogAlphaState(metadata={"run_id": "formal-run"}, qualified_pool=[candidate])
    (run_dir / "final_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_index(
        factor_pool,
        [_entry(0, "qualified", "alpha-range-vol", "qualified-1")],
    )
    provider = FakeMetricsProvider(
        {
            "qualified-1": CandidateEvaluationResult(
                candidate_id="qualified-1",
                guard_report=GuardReport(
                    guard_name="runtime",
                    status=GuardStatus.FAIL,
                ),
                error="runtime guard failed",
            )
        }
    )

    result = validate_qualified_factors(
        run_dir=run_dir,
        factor_pool_root=factor_pool,
        metrics_provider=provider,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["counts"]["validation_error"] == 1
    assert report["records"][0]["validation_outcome"] == "validation_error"
    assert report["records"][0]["guard_status"] == "fail"
    assert report["records"][0]["error"] == "runtime guard failed"


def test_validation_audit_fails_when_factor_pool_mapping_is_missing(tmp_path):
    run_dir = tmp_path / "outputs" / "experiments" / "formal-run"
    factor_pool = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir(parents=True)
    candidate = _candidate("qualified-1", CandidateStage.QUALIFIED, score=0.1)
    state = CogAlphaState(metadata={"run_id": "formal-run"}, qualified_pool=[candidate])
    (run_dir / "final_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    _write_index(factor_pool, [])
    provider = FakeMetricsProvider({})

    try:
        validate_qualified_factors(
            run_dir=run_dir,
            factor_pool_root=factor_pool,
            metrics_provider=provider,
        )
    except ValueError as exc:
        assert "qualified-1" in str(exc)
    else:
        raise AssertionError("Expected missing factor pool mapping failure.")


def _candidate(candidate_id, stage, *, score):
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id.replace('-', '_')}",
            code="def factor_test(df):\n    return df['close'] - df['open']\n",
            formula="close - open",
            rationale="Synthetic test factor.",
        ),
        stage=stage,
        metadata={"fitness_metrics": _metrics(score).model_dump(mode="python")},
    )


def _metrics(value):
    return FitnessMetrics(
        ic=value,
        rank_ic=value,
        icir=value,
        rank_icir=value,
        mi=value,
    )


def _entry(factor_id, pool, domain_agent, candidate_id):
    return {
        "factor_id": factor_id,
        "file": f"{pool}/{domain_agent}/{factor_id}.json",
        "pool": pool,
        "domain_agent": domain_agent,
        "run_id": "formal-run",
        "candidate_id": candidate_id,
        "factor_name": f"factor_{candidate_id.replace('-', '_')}",
    }


def _write_index(factor_pool, factors):
    factor_pool.mkdir(parents=True)
    factor_pool.joinpath("index.json").write_text(
        json.dumps(
            {
                "next_factor_id": len(factors),
                "counts": {"total": len(factors)},
                "factors": factors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
