import json

from cogalpha.factor_memory import (
    FactorMemoryCompactionResult,
    build_prior_lessons,
    update_factor_memory,
    update_factor_memory_from_backtest_audit,
    update_factor_memory_from_validation_audit,
)


class FakeSummarizer:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.requests = []

    def summarize(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return FactorMemoryCompactionResult.model_validate(self.result)


def test_factor_memory_processes_only_new_factor_pool_entries(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    memory_root = tmp_path / "outputs" / "factor_memory"
    _write_factor(
        factor_pool,
        "elite/alpha-market-cycle/0.json",
        run_id="run-1",
        factor_name="factor_trend_strength",
        formula="slope(close, 60) / close",
        rationale="Long trend strength.",
        metrics={
            "ic": 0.03,
            "rank_ic": 0.02,
            "icir": 0.2,
            "rank_icir": 0.1,
            "mi": 0.03,
        },
    )
    _write_factor(
        factor_pool,
        "rejected/alpha-market-cycle/1.json",
        run_id="run-1",
        factor_name="factor_raw_spike",
        formula="high - low",
        rationale="Raw range spike.",
        metrics={
            "ic": 0.01,
            "rank_ic": -0.02,
            "icir": 0.05,
            "rank_icir": -0.1,
            "mi": 0.01,
        },
    )
    _write_index(
        factor_pool,
        [
            _entry(0, "elite/alpha-market-cycle/0.json", "elite", "alpha-market-cycle"),
            _entry(1, "rejected/alpha-market-cycle/1.json", "rejected", "alpha-market-cycle"),
        ],
    )

    result = update_factor_memory(factor_pool_root=factor_pool, memory_root=memory_root)

    assert result.processed_factor_ids == [0, 1]
    assert result.domain_updates == {"alpha-market-cycle": 2}
    state = json.loads((memory_root / "state.json").read_text(encoding="utf-8"))
    assert state["last_processed_factor_id"] == 2

    memory = json.loads(
        (memory_root / "domain_memory/alpha-market-cycle.json").read_text(encoding="utf-8")
    )
    assert memory["pool_counts"] == {"elite": 1, "qualified": 0, "rejected": 1}
    assert memory["processed_factor_ids"] == [0, 1]
    assert memory["success_patterns"][0]["evidence_factor_ids"] == [0]
    assert memory["failure_patterns"][0]["evidence_factor_ids"] == [1]
    assert memory["metric_bottlenecks"] == {"rank_ic": 1, "rank_icir": 1}
    assert "factor_raw_spike" in memory["failure_patterns"][0]["lesson"]

    cache = (memory_root / "retrieval_cache/alpha-market-cycle.md").read_text(
        encoding="utf-8"
    )
    assert "# Prior Lessons for alpha-market-cycle" in cache
    assert "Effective" in cache
    assert "Avoid" in cache

    second = update_factor_memory(factor_pool_root=factor_pool, memory_root=memory_root)

    assert second.processed_factor_ids == []
    unchanged = json.loads(
        (memory_root / "domain_memory/alpha-market-cycle.json").read_text(encoding="utf-8")
    )
    assert unchanged["processed_factor_ids"] == [0, 1]


def test_factor_memory_appends_later_factor_ids_and_builds_prior_lessons(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    memory_root = tmp_path / "outputs" / "factor_memory"
    _write_factor(
        factor_pool,
        "qualified/alpha-range-vol/0.json",
        run_id="run-1",
        factor_name="factor_range_ratio",
        formula="(high - low) / close",
        rationale="Range ratio.",
        metrics={
            "ic": 0.02,
            "rank_ic": 0.02,
            "icir": 0.15,
            "rank_icir": 0.12,
            "mi": 0.02,
        },
    )
    _write_index(
        factor_pool,
        [_entry(0, "qualified/alpha-range-vol/0.json", "qualified", "alpha-range-vol")],
    )
    update_factor_memory(factor_pool_root=factor_pool, memory_root=memory_root)

    _write_factor(
        factor_pool,
        "rejected/alpha-range-vol/1.json",
        run_id="run-2",
        factor_name="factor_range_noise",
        formula="high - low",
        rationale="Noisy raw range.",
        metrics={
            "ic": -0.01,
            "rank_ic": -0.03,
            "icir": -0.2,
            "rank_icir": -0.3,
            "mi": 0.01,
        },
    )
    _write_index(
        factor_pool,
        [
            _entry(0, "qualified/alpha-range-vol/0.json", "qualified", "alpha-range-vol"),
            _entry(1, "rejected/alpha-range-vol/1.json", "rejected", "alpha-range-vol"),
        ],
    )

    result = update_factor_memory(factor_pool_root=factor_pool, memory_root=memory_root)

    assert result.processed_factor_ids == [1]
    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert memory["processed_factor_ids"] == [0, 1]
    assert memory["pool_counts"] == {"elite": 0, "qualified": 1, "rejected": 1}
    prior_lessons = build_prior_lessons(memory)
    assert "factor_range_ratio reached qualified" in prior_lessons
    assert "factor_range_noise was rejected" in prior_lessons


def test_factor_memory_uses_valid_llm_summarizer_output(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    memory_root = tmp_path / "outputs" / "factor_memory"
    _write_factor(
        factor_pool,
        "rejected/alpha-range-vol/0.json",
        run_id="run-1",
        factor_name="factor_range_noise",
        formula="high - low",
        rationale="Noisy raw range.",
        metrics={
            "ic": -0.01,
            "rank_ic": -0.03,
            "icir": -0.2,
            "rank_icir": -0.3,
            "mi": 0.01,
        },
    )
    _write_index(
        factor_pool,
        [_entry(0, "rejected/alpha-range-vol/0.json", "rejected", "alpha-range-vol")],
    )
    summarizer = FakeSummarizer(
        {
            "success_patterns": [],
            "failure_patterns": [
                {
                    "lesson": "LLM compacted range failures into a rank-stability lesson.",
                    "evidence_factor_ids": [0],
                }
            ],
            "avoid_patterns": [
                {
                    "lesson": "Avoid raw range amplitude without normalization.",
                    "evidence_factor_ids": [0, 999],
                }
            ],
            "regime_hypotheses": [
                {
                    "hypothesis": (
                        "Validation evidence weakly favors normalized range mechanisms."
                    ),
                    "confidence": "low",
                    "evidence_factor_ids": [0, 999],
                    "risk": "May overfit one validation window.",
                }
            ],
        }
    )

    update_factor_memory(
        factor_pool_root=factor_pool,
        memory_root=memory_root,
        summarizer=summarizer,
    )

    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert len(summarizer.requests) == 1
    assert summarizer.requests[0].new_evidence[0].factor_id == 0
    assert memory["failure_patterns"] == [
        {
            "lesson": "LLM compacted range failures into a rank-stability lesson.",
            "evidence_factor_ids": [0],
        }
    ]
    assert memory["avoid_patterns"] == [
        {
            "lesson": "Avoid raw range amplitude without normalization.",
            "evidence_factor_ids": [0],
        }
    ]
    assert memory["regime_hypotheses"] == [
        {
            "hypothesis": "Validation evidence weakly favors normalized range mechanisms.",
            "confidence": "low",
            "evidence_factor_ids": [0],
            "risk": "May overfit one validation window.",
        }
    ]


def test_factor_memory_falls_back_when_llm_summarizer_fails(tmp_path):
    factor_pool = tmp_path / "outputs" / "factor_pool"
    memory_root = tmp_path / "outputs" / "factor_memory"
    _write_factor(
        factor_pool,
        "rejected/alpha-market-cycle/0.json",
        run_id="run-1",
        factor_name="factor_raw_trend",
        formula="close - open",
        rationale="Raw trend.",
        metrics={
            "ic": -0.01,
            "rank_ic": -0.02,
            "icir": -0.03,
            "rank_icir": -0.04,
            "mi": 0.01,
        },
    )
    _write_index(
        factor_pool,
        [_entry(0, "rejected/alpha-market-cycle/0.json", "rejected", "alpha-market-cycle")],
    )

    update_factor_memory(
        factor_pool_root=factor_pool,
        memory_root=memory_root,
        summarizer=FakeSummarizer(error=RuntimeError("llm unavailable")),
    )

    memory = json.loads(
        (memory_root / "domain_memory/alpha-market-cycle.json").read_text(encoding="utf-8")
    )
    assert "factor_raw_trend was rejected" in memory["failure_patterns"][0]["lesson"]


def test_factor_memory_ingests_validation_audit_success_and_failure(tmp_path):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "outputs" / "experiments" / "run-1" / (
        "validation_audit_valid.json"
    )
    _write_validation_audit(
        audit_path,
        [
            _validation_record(
                factor_id=1,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_ratio",
                outcome="validation_success",
            ),
            _validation_record(
                factor_id=2,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_noise",
                outcome="validation_failure",
                bottlenecks=["rank_ic", "rank_icir"],
            ),
        ],
    )

    result = update_factor_memory_from_validation_audit(
        audit_path=audit_path,
        memory_root=memory_root,
    )

    assert result.processed_audit_keys == ["run-1:valid:1", "run-1:valid:2"]
    assert result.domain_updates == {"alpha-range-vol": 2}
    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert memory["validation_counts"] == {
        "validation_error": 0,
        "validation_failure": 1,
        "validation_success": 1,
    }
    assert memory["validation_metric_bottlenecks"] == {
        "rank_ic": 1,
        "rank_icir": 1,
    }
    assert "passed valid validation minima" in memory["success_patterns"][0]["lesson"]
    assert "failed valid validation minima" in memory["failure_patterns"][0]["lesson"]
    assert "out-of-sample valid robustness" in memory["avoid_patterns"][0]["lesson"]

    cache = (memory_root / "retrieval_cache/alpha-range-vol.md").read_text(
        encoding="utf-8"
    )
    assert "factor_range_ratio passed valid validation minima" in cache
    assert "factor_range_noise qualified in the source run" in cache


def test_factor_memory_validation_audit_can_use_summarizer_for_regime_hypothesis(
    tmp_path,
):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "outputs" / "experiments" / "run-1" / (
        "validation_audit_valid.json"
    )
    _write_validation_audit(
        audit_path,
        [
            _validation_record(
                factor_id=1,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_ratio",
                outcome="validation_success",
            ),
            _validation_record(
                factor_id=2,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_noise",
                outcome="validation_failure",
                bottlenecks=["rank_icir"],
            ),
        ],
    )
    summarizer = FakeSummarizer(
        {
            "success_patterns": [
                {
                    "lesson": "Range normalization generalized better than raw range.",
                    "evidence_factor_ids": [1],
                }
            ],
            "failure_patterns": [],
            "avoid_patterns": [],
            "regime_hypotheses": [
                {
                    "hypothesis": (
                        "Recent validation evidence weakly favors normalized "
                        "range-volatility mechanisms."
                    ),
                    "confidence": "medium",
                    "evidence_factor_ids": [1, 2, 999],
                    "risk": "Could overfit the validation window.",
                }
            ],
        }
    )

    update_factor_memory_from_validation_audit(
        audit_path=audit_path,
        memory_root=memory_root,
        summarizer=summarizer,
    )

    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert summarizer.requests[0].new_evidence[0].validation_outcome == (
        "validation_success"
    )
    assert memory["regime_hypotheses"] == [
        {
            "hypothesis": (
                "Recent validation evidence weakly favors normalized "
                "range-volatility mechanisms."
            ),
            "confidence": "medium",
            "evidence_factor_ids": [1, 2],
            "risk": "Could overfit the validation window.",
        }
    ]
    cache = (memory_root / "retrieval_cache/alpha-range-vol.md").read_text(
        encoding="utf-8"
    )
    assert "## Regime Hypotheses" in cache
    assert "Use as inspiration only; keep mechanisms diverse." in cache


def test_factor_memory_keeps_static_validation_lessons_when_summarizer_fails(
    tmp_path,
):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "outputs" / "experiments" / "run-1" / (
        "validation_audit_valid.json"
    )
    _write_validation_audit(
        audit_path,
        [
            _validation_record(
                factor_id=1,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_ratio",
                outcome="validation_success",
            ),
            _validation_record(
                factor_id=2,
                domain_agent="alpha-range-vol",
                factor_name="factor_range_noise",
                outcome="validation_failure",
                bottlenecks=["rank_icir"],
            ),
        ],
    )

    update_factor_memory_from_validation_audit(
        audit_path=audit_path,
        memory_root=memory_root,
        summarizer=FakeSummarizer(error=RuntimeError("llm unavailable")),
    )

    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert "passed valid validation minima" in memory["success_patterns"][0]["lesson"]
    assert "failed valid validation minima" in memory["failure_patterns"][0]["lesson"]
    assert "out-of-sample valid robustness" in memory["avoid_patterns"][0]["lesson"]
    assert memory["regime_hypotheses"] == []


def test_factor_memory_validation_audit_ingestion_is_idempotent(tmp_path):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "outputs" / "experiments" / "run-1" / (
        "validation_audit_valid.json"
    )
    _write_validation_audit(
        audit_path,
        [
            _validation_record(
                factor_id=1,
                domain_agent="alpha-market-cycle",
                factor_name="factor_cycle",
                outcome="validation_success",
            )
        ],
    )

    update_factor_memory_from_validation_audit(
        audit_path=audit_path,
        memory_root=memory_root,
    )
    second = update_factor_memory_from_validation_audit(
        audit_path=audit_path,
        memory_root=memory_root,
    )

    assert second.processed_audit_keys == []
    memory = json.loads(
        (memory_root / "domain_memory/alpha-market-cycle.json").read_text(
            encoding="utf-8"
        )
    )
    assert memory["validation_counts"]["validation_success"] == 1
    assert len(memory["success_patterns"]) == 1


def test_factor_memory_rejects_test_audit_ingestion(tmp_path):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "outputs" / "experiments" / "run-1" / (
        "validation_audit_test.json"
    )
    _write_validation_audit(
        audit_path,
        [
            _validation_record(
                factor_id=1,
                domain_agent="alpha-market-cycle",
                factor_name="factor_cycle",
                outcome="validation_success",
            )
        ],
        split="test",
    )

    try:
        update_factor_memory_from_validation_audit(
            audit_path=audit_path,
            memory_root=memory_root,
        )
    except ValueError as exc:
        assert "Test split" in str(exc)
    else:
        raise AssertionError("Expected test audit ingestion failure.")


def test_factor_memory_ingests_backtest_success_and_dedupes(tmp_path):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "backtest_audit.json"
    _write_backtest_audit(
        audit_path,
        factor_id=3,
        factor_name="factor_range_strength",
        domain_agent="alpha-range-vol",
        outcome="backtest_success",
        bottlenecks=[],
        net_return=0.12,
        rank_ic=0.03,
    )

    first = update_factor_memory_from_backtest_audit(
        audit_path=audit_path,
        memory_root=memory_root,
    )
    second = update_factor_memory_from_backtest_audit(
        audit_path=audit_path,
        memory_root=memory_root,
    )

    assert first.processed_audit_keys == ["backtest:3:unit-test"]
    assert second.processed_audit_keys == []
    memory = json.loads(
        (memory_root / "domain_memory/alpha-range-vol.json").read_text(encoding="utf-8")
    )
    assert memory["backtest_counts"] == {
        "backtest_success": 1,
        "backtest_failure": 0,
    }
    assert "passed independent backtest" in memory["success_patterns"][0]["lesson"]
    cache = (memory_root / "retrieval_cache/alpha-range-vol.md").read_text(
        encoding="utf-8"
    )
    assert "factor_range_strength" in cache


def test_factor_memory_ingests_backtest_failure_and_fallbacks_invalid_summary(tmp_path):
    memory_root = tmp_path / "outputs" / "factor_memory"
    audit_path = tmp_path / "backtest_audit.json"
    _write_backtest_audit(
        audit_path,
        factor_id=4,
        factor_name="factor_raw_volume_spike",
        domain_agent="alpha-liquidity-shock",
        outcome="backtest_failure",
        bottlenecks=["rank_ic_mean", "transaction_cost"],
        net_return=-0.02,
        rank_ic=-0.01,
    )
    summarizer = FakeSummarizer(
        result={
            "success_patterns": [
                {"lesson": "invented", "evidence_factor_ids": [999]},
            ],
            "failure_patterns": [
                {"lesson": "invented", "evidence_factor_ids": [999]},
            ],
            "avoid_patterns": [],
            "regime_hypotheses": [],
        }
    )

    update_factor_memory_from_backtest_audit(
        audit_path=audit_path,
        memory_root=memory_root,
        summarizer=summarizer,
    )

    memory = json.loads(
        (memory_root / "domain_memory/alpha-liquidity-shock.json").read_text(
            encoding="utf-8"
        )
    )
    assert memory["backtest_counts"]["backtest_failure"] == 1
    assert memory["backtest_bottlenecks"] == {
        "rank_ic_mean": 1,
        "transaction_cost": 1,
    }
    assert "failed independent backtest" in memory["failure_patterns"][0]["lesson"]
    assert "invented" not in json.dumps(memory)


def _write_index(factor_pool, factors):
    payload = {
        "next_factor_id": max(entry["factor_id"] for entry in factors) + 1,
        "counts": {"total": len(factors)},
        "factors": factors,
    }
    path = factor_pool / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(factor_id, file, pool, domain_agent):
    return {
        "factor_id": factor_id,
        "file": file,
        "pool": pool,
        "domain_agent": domain_agent,
        "run_id": "run-1",
        "candidate_id": f"candidate-{factor_id}",
        "factor_name": f"factor_{factor_id}",
    }


def _write_factor(
    factor_pool,
    relative_path,
    *,
    run_id,
    factor_name,
    formula,
    rationale,
    metrics,
):
    path = factor_pool / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "factor_name": factor_name,
                "formula": formula,
                "code": f"def {factor_name}(df):\n    return df['close']\n",
                "rationale": rationale,
                "required_columns": ["open", "high", "low", "close", "volume"],
                "allowed_libraries": ["np", "pd"],
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_validation_audit(path, records, *, split="valid"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "split": split,
                "data_version": "test-data",
                "created_at": "2026-01-01T00:00:00+00:00",
                "validation_rule": "qualified_minima",
                "thresholds": {
                    "ic": 0.005,
                    "rank_ic": 0.005,
                    "icir": 0.05,
                    "rank_icir": 0.05,
                    "mi": 0.02,
                },
                "records": records,
                "counts": {"total": len(records)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_backtest_audit(
    path,
    *,
    factor_id,
    factor_name,
    domain_agent,
    outcome,
    bottlenecks,
    net_return,
    rank_ic,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "audit_id": f"backtest:{factor_id}:unit-test",
                "created_at": "2026-01-01T00:00:00+00:00",
                "factor_id": factor_id,
                "factor_name": factor_name,
                "candidate_id": f"candidate-{factor_id}",
                "domain_agent": domain_agent,
                "run_id": "run-1",
                "pool": "qualified",
                "data_version": "unit-test-data",
                "start_date": "2021-01-01",
                "end_date": "2022-12-31",
                "outcome": outcome,
                "bottlenecks": bottlenecks,
                "summary": {
                    "ic_mean": 0.02,
                    "rank_ic_mean": rank_ic,
                    "icir": 0.2,
                    "rank_icir": 0.2,
                    "mi": 0.03,
                    "long_short_gross_annual_return": 0.15,
                    "long_short_net_annual_return": net_return,
                    "top_excess_annual_return": 0.08,
                    "max_drawdown": -0.05,
                    "avg_turnover": 0.3,
                    "avg_coverage": 0.9,
                },
                "annual_metrics": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _validation_record(
    *,
    factor_id,
    domain_agent,
    factor_name,
    outcome,
    bottlenecks=None,
):
    return {
        "factor_id": factor_id,
        "candidate_id": f"candidate-{factor_id}",
        "domain_agent": domain_agent,
        "pool": "qualified",
        "factor_name": factor_name,
        "train_metrics": {
            "ic": 0.02,
            "rank_ic": 0.02,
            "icir": 0.2,
            "rank_icir": 0.2,
            "mi": 0.03,
        },
        "validation_metrics": {
            "ic": 0.02,
            "rank_ic": 0.02,
            "icir": 0.2,
            "rank_icir": 0.2,
            "mi": 0.03,
        },
        "validation_outcome": outcome,
        "metric_bottlenecks": bottlenecks or [],
        "guard_status": None,
        "error": None,
        "cache_hit": False,
    }
