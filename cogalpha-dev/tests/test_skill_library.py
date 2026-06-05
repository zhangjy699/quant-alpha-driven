import pytest
from pydantic import ValidationError

from cogalpha.schemas import CandidateStage
from cogalpha.tracing import CogAlphaTraceEvent, TraceEventKind


def test_skill_utility_record_updates_from_qualified_trace():
    from cogalpha.skill_library import SkillUtilityRecord, update_skill_utility_from_trace

    record = SkillUtilityRecord(skill_name="alpha-herding", utility=0.0)
    updated = update_skill_utility_from_trace(record, _trace_for_stage(CandidateStage.QUALIFIED))

    assert updated.skill_name == "alpha-herding"
    assert updated.utility > record.utility
    assert updated.usage_count == 1
    assert updated.evidence_ids == ["trace-qualified"]
    assert updated.last_evidence_id == "trace-qualified"


def test_skill_utility_record_rewards_elite_more_than_qualified():
    from cogalpha.skill_library import SkillUtilityRecord, update_skill_utility_from_trace

    base = SkillUtilityRecord(skill_name="alpha-herding", utility=0.0)

    qualified = update_skill_utility_from_trace(
        base, _trace_for_stage(CandidateStage.QUALIFIED)
    )
    elite = update_skill_utility_from_trace(base, _trace_for_stage(CandidateStage.ELITE))

    assert elite.utility > qualified.utility


def test_skill_utility_record_penalizes_rejected_trace_but_stays_bounded():
    from cogalpha.skill_library import SkillUtilityRecord, update_skill_utility_from_trace

    record = SkillUtilityRecord(skill_name="alpha-herding", utility=-0.95)
    updated = update_skill_utility_from_trace(
        record, _trace_for_stage(CandidateStage.REJECTED_BY_FITNESS)
    )

    assert updated.utility <= record.utility
    assert updated.utility >= -1.0
    assert updated.usage_count == 1
    assert updated.evidence_ids == ["trace-rejected_by_fitness"]


def test_failed_qualified_fitness_event_does_not_increase_skill_utility():
    from cogalpha.skill_library import SkillUtilityRecord, update_skill_utility_from_trace

    record = SkillUtilityRecord(skill_name="alpha-herding", utility=0.0)
    updated = update_skill_utility_from_trace(
        record,
        [
            _event(
                0,
                TraceEventKind.SKILL_INVOCATION_FINISHED,
                skill_name="alpha-herding",
                evidence_id="trace-failed-qualified",
            ),
            _event(
                1,
                TraceEventKind.FITNESS_EVALUATION_RECORDED,
                candidate_id="alpha-1",
                skill_name="alpha-herding",
                stage=CandidateStage.QUALIFIED,
                status="error",
                evidence_id="trace-failed-qualified",
            ),
        ],
    )

    assert updated.utility == 0.0
    assert updated.usage_count == 1
    assert updated.evidence_ids == ["trace-failed-qualified"]


def test_skill_selection_record_captures_eligible_and_selected_skills():
    from cogalpha.skill_library import SkillSelectionRecord

    record = SkillSelectionRecord(
        decision_id="decision-1",
        selected_skill="alpha-herding",
        eligible_skills=["alpha-herding", "alpha-market-cycle"],
        reason="highest trace-grounded utility for crowding task",
        evidence_ids=["trace-qualified"],
    )

    assert record.selected_skill in record.eligible_skills
    assert record.evidence_ids == ["trace-qualified"]


def test_skill_utility_and_selection_artifacts_round_trip(tmp_path):
    from cogalpha.skill_library import (
        SkillSelectionRecord,
        SkillUtilityRecord,
        read_skill_selection_records,
        read_skill_utility_records,
        write_skill_selection_records,
        write_skill_utility_records,
    )

    utility_records = [
        SkillUtilityRecord(
            skill_name="alpha-herding",
            utility=0.2,
            usage_count=1,
            evidence_ids=["trace-qualified"],
            last_evidence_id="trace-qualified",
        )
    ]
    selection_records = [
        SkillSelectionRecord(
            decision_id="decision-1",
            selected_skill="alpha-herding",
            eligible_skills=["alpha-herding", "alpha-market-cycle"],
            reason="highest trace-grounded utility for crowding task",
            evidence_ids=["trace-qualified"],
        )
    ]

    utility_path = tmp_path / "skill_utility.json"
    selection_path = tmp_path / "skill_selection.jsonl"
    write_skill_utility_records(utility_path, utility_records)
    write_skill_selection_records(selection_path, selection_records)

    assert read_skill_utility_records(utility_path) == utility_records
    assert read_skill_selection_records(selection_path) == selection_records


def test_skill_update_proposal_requires_governance_fields_for_promote():
    from cogalpha.skill_library import SkillUpdateProposal

    with pytest.raises(ValidationError):
        SkillUpdateProposal(
            proposal_id="proposal-1",
            skill_name="alpha-herding",
            proposed_change="Prefer slow-horizon rank stabilization.",
            status="promote",
        )

    proposal = SkillUpdateProposal(
        proposal_id="proposal-1",
        skill_name="alpha-herding",
        proposed_change="Prefer slow-horizon rank stabilization.",
        status="promote",
        evidence_id="evidence-r9",
        reviewer="runtime-reviewer",
        rollback="skills/alpha-herding/SKILL.md@previous",
    )

    assert proposal.status == "promote"


def _trace_for_stage(stage: CandidateStage) -> list[CogAlphaTraceEvent]:
    evidence_id = f"trace-{stage.value}"
    return [
        _event(
            0,
            TraceEventKind.SKILL_INVOCATION_FINISHED,
            skill_name="alpha-herding",
            evidence_id=evidence_id,
        ),
        _event(
            1,
            TraceEventKind.CANDIDATE_STAGE_CHANGED,
            candidate_id="alpha-1",
            skill_name="alpha-herding",
            stage=stage,
            evidence_id=evidence_id,
        ),
        _event(
            2,
            TraceEventKind.FITNESS_EVALUATION_RECORDED,
            candidate_id="alpha-1",
            skill_name="alpha-herding",
            stage=stage,
            metrics={
                "ic": 0.04,
                "rank_ic": 0.05,
                "icir": 0.2,
                "rank_icir": 0.3,
                "mi": 0.02,
            },
            evidence_id=evidence_id,
        ),
    ]


def _event(sequence: int, kind: TraceEventKind, **payload: object) -> CogAlphaTraceEvent:
    return CogAlphaTraceEvent(
        run_id="run-skill-library",
        sequence=sequence,
        kind=kind,
        payload=payload,
    )
