from cogalpha.governance import (
    ChangeType,
    EvidenceDecision,
    EvidenceRecord,
    PromotionDecision,
    append_jsonl_record,
    load_jsonl_records,
)


def test_evidence_record_round_trips_as_jsonl(tmp_path):
    path = tmp_path / "evidence.jsonl"
    record = EvidenceRecord(
        evidence_id="ev-trade-delay",
        observation="Preflight IC was inflated under same-open labels.",
        evidence="Trade-delay preflight reduced same-day body IC from 0.286 to 0.005.",
        attribution_layer="Data / sample",
        hypothesis="Same-day OHLCV with open[t] label leaks intraday information.",
        proposed_change="Use trade_delay_days=1 for Baseline Experiment labels.",
        change_type=ChangeType.DATA_POLICY,
        required_ablation="Compare fixed preflight candidates before and after trade delay.",
        artifacts=["outputs/preflight/valid_fixed_preflight.json"],
        decision=EvidenceDecision.PROMOTE,
        reviewer="codex",
        rollback="Set trade_delay_days back to 0 and regenerate data.",
    )

    append_jsonl_record(path, record)
    loaded = load_jsonl_records(path, EvidenceRecord)

    assert loaded == [record]


def test_promotion_decision_requires_evidence_and_rollback():
    decision = PromotionDecision(
        promotion_id="promote-trade-delay",
        candidate_change="Baseline Experiment labels use trade_delay_days=1.",
        evidence_ids=["ev-trade-delay"],
        reviewer="codex-review",
        approver="user",
        decision=EvidenceDecision.HOLD,
        rollback_target="configs/baseline.yaml: trade_delay_days=0 plus data regeneration",
    )

    assert decision.evidence_ids == ["ev-trade-delay"]
    assert decision.rollback_target
