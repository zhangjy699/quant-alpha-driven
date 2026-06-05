import json

from cogalpha.factor_pool import FACTOR_SUMMARY_FIELDS, export_factor_pool
from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    DAGNodeResult,
    EvolutionLineage,
    EvolutionOperation,
    FitnessMetrics,
)


def test_factor_pool_exports_shared_numeric_files_and_minimal_summaries(tmp_path):
    run_dir = tmp_path / "formal-mvp-test"
    output_root = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir()
    elite = _candidate(
        "elite-1",
        "factor_duplicate_name",
        CandidateStage.ELITE,
        agent_skill="alpha-market-cycle",
        score=0.3,
    )
    qualified_elite_duplicate = elite.model_copy(deep=True)
    qualified = _candidate(
        "qualified-1",
        "factor_duplicate_name",
        CandidateStage.QUALIFIED,
        agent_skill="alpha-daily-trend",
        score=0.2,
    )
    rejected_quality = _candidate(
        "quality-reject",
        "factor_quality_reject",
        CandidateStage.REJECTED_BY_QUALITY,
        agent_skill="alpha-liquidity",
    )
    rejected = [
        _candidate(
            f"reject-{index}",
            f"factor_reject_{index}",
            CandidateStage.REJECTED_BY_FITNESS,
            agent_skill="alpha-reversal",
            score=score,
        )
        for index, score in enumerate([0.05, -0.5, 0.0, -0.2], start=1)
    ]
    state = CogAlphaState(
        metadata={"run_id": "formal-mvp-test", "split": "valid"},
        elite_pool=[elite],
        qualified_pool=[qualified_elite_duplicate, qualified],
        rejected_pool=[rejected_quality, *rejected],
        node_history=[
            DAGNodeResult(
                node_name="domain_agents",
                candidates=[elite, qualified, rejected_quality, *rejected],
            )
        ],
    )
    (run_dir / "final_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = export_factor_pool(run_dir, output_root=output_root)

    assert result.factor_pool_dir == output_root
    assert result.counts == {"elite": 1, "qualified": 1, "rejected": 3}
    assert not (run_dir / "factor_pool").exists()
    assert (output_root / "elite/alpha-market-cycle/0.json").exists()
    assert (output_root / "qualified/alpha-daily-trend/1.json").exists()
    assert not (output_root / "qualified/alpha-market-cycle/elite-1.json").exists()

    rejected_files = sorted(
        path.name
        for path in (output_root / "rejected/alpha-reversal").glob("*.json")
    )
    assert rejected_files == ["2.json", "3.json", "4.json"]
    assert not (output_root / "rejected/alpha-liquidity").exists()

    summary = json.loads((output_root / "elite/alpha-market-cycle/0.json").read_text())
    assert set(summary) == FACTOR_SUMMARY_FIELDS
    assert summary["run_id"] == "formal-mvp-test"
    assert summary["factor_name"] == "factor_duplicate_name"
    assert summary["metrics"] == {
        "ic": 0.3,
        "rank_ic": 0.3,
        "icir": 0.3,
        "rank_icir": 0.3,
        "mi": 0.3,
    }

    index = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert index["next_factor_id"] == 5
    assert index["counts"]["by_pool"] == {"elite": 1, "qualified": 1, "rejected": 3}
    assert index["factors"][0] == {
        "factor_id": 0,
        "file": "elite/alpha-market-cycle/0.json",
        "pool": "elite",
        "domain_agent": "alpha-market-cycle",
        "run_id": "formal-mvp-test",
        "candidate_id": "elite-1",
        "factor_name": "factor_duplicate_name",
    }
    assert [entry["candidate_id"] for entry in index["factors"][-3:]] == [
        "reject-2",
        "reject-4",
        "reject-3",
    ]


def test_factor_pool_appends_after_existing_index(tmp_path):
    run_dir = tmp_path / "formal-mvp-test"
    output_root = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir()
    output_root.mkdir(parents=True)
    (output_root / "index.json").write_text(
        json.dumps(
            {
                "next_factor_id": 7,
                "counts": {"total": 1},
                "factors": [
                    {
                        "factor_id": 6,
                        "file": "elite/alpha-market-cycle/6.json",
                        "pool": "elite",
                        "domain_agent": "alpha-market-cycle",
                        "run_id": "old-run",
                        "candidate_id": "old",
                        "factor_name": "factor_old",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    elite = _candidate(
        "elite-1",
        "factor_new",
        CandidateStage.ELITE,
        agent_skill="alpha-market-cycle",
        score=0.3,
    )
    state = CogAlphaState(
        metadata={"run_id": "formal-mvp-test", "split": "valid"},
        elite_pool=[elite],
        qualified_pool=[elite],
        node_history=[DAGNodeResult(node_name="domain_agents", candidates=[elite])],
    )
    (run_dir / "final_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")

    export_factor_pool(run_dir, output_root=output_root)

    assert (output_root / "elite/alpha-market-cycle/7.json").exists()
    index = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert index["next_factor_id"] == 8
    assert [entry["factor_id"] for entry in index["factors"]] == [6, 7]


def test_factor_pool_attributes_evolved_and_crossover_candidates_to_parent_domains(tmp_path):
    run_dir = tmp_path / "formal-mvp-test"
    output_root = tmp_path / "outputs" / "factor_pool"
    run_dir.mkdir()
    left = _candidate(
        "left-parent",
        "factor_left",
        CandidateStage.ELITE,
        agent_skill="alpha-market-cycle",
        score=0.5,
    )
    right = _candidate(
        "right-parent",
        "factor_right",
        CandidateStage.ELITE,
        agent_skill="alpha-range-vol",
        score=0.4,
    )
    repaired_mutation = _candidate(
        "repaired-child",
        "factor_repaired_child",
        CandidateStage.QUALIFIED,
        agent_skill="alpha-code-repair",
        parent_ids=["left-parent"],
        generation=1,
        score=0.2,
    )
    crossover = _candidate(
        "cross-child",
        "factor_cross_child",
        CandidateStage.QUALIFIED,
        agent_skill="alpha-crossover",
        operation=EvolutionOperation.CROSSOVER,
        parent_ids=["left-parent", "right-parent"],
        generation=1,
        score=0.3,
    )
    unknown = _candidate(
        "unknown-child",
        "factor_unknown_child",
        CandidateStage.QUALIFIED,
        agent_skill="alpha-code-repair",
        parent_ids=["missing-parent"],
        generation=1,
        score=0.1,
    )
    state = CogAlphaState(
        metadata={"run_id": "formal-mvp-test", "split": "valid"},
        elite_pool=[left, right],
        qualified_pool=[left, right, repaired_mutation, crossover, unknown],
        node_history=[
            DAGNodeResult(node_name="domain_agents", candidates=[left, right]),
            DAGNodeResult(
                node_name="thinking_evolution",
                candidates=[repaired_mutation, crossover, unknown],
            ),
        ],
    )
    (run_dir / "final_state.json").write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )

    export_factor_pool(run_dir, output_root=output_root)

    index = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    entries_by_candidate = {}
    for entry in index["factors"]:
        entries_by_candidate.setdefault(entry["candidate_id"], []).append(entry)

    assert entries_by_candidate["repaired-child"][0]["domain_agent"] == "alpha-market-cycle"
    assert {
        entry["domain_agent"] for entry in entries_by_candidate["cross-child"]
    } == {"alpha-market-cycle", "alpha-range-vol"}
    assert entries_by_candidate["unknown-child"][0]["domain_agent"] == "unknown_domain"
    for entry in entries_by_candidate["cross-child"]:
        assert (output_root / entry["file"]).exists()


def _candidate(
    candidate_id: str,
    factor_name: str,
    stage: CandidateStage,
    *,
    agent_skill: str,
    score: float | None = None,
    parent_ids: list[str] | None = None,
    generation: int = 0,
    operation: EvolutionOperation | None = None,
) -> AlphaCandidate:
    metadata = {}
    if score is not None:
        metadata["fitness_metrics"] = FitnessMetrics(
            ic=score,
            rank_ic=score,
            icir=score,
            rank_icir=score,
            mi=score,
        ).model_dump(mode="python")
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=factor_name,
            code=f"def {factor_name}(df):\n    return df['close'] - df['open']\n",
            formula="close - open",
            rationale=f"{factor_name} rationale.",
        ),
        stage=stage,
        lineage=EvolutionLineage(
            operation=operation,
            parent_ids=parent_ids or [],
            generation=generation,
            agent_skill=agent_skill,
        ),
        metadata=metadata,
    )
