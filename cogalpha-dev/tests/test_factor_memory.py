import json

from cogalpha.factor_memory import (
    FactorMemoryCompactionResult,
    build_prior_lessons,
    update_factor_memory,
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
