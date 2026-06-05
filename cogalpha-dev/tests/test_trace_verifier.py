from cogalpha.schemas import (
    AlphaCandidate,
    AlphaFunction,
    CandidateStage,
    CogAlphaState,
    DAGNodeResult,
)
from cogalpha.tracing import CogAlphaTraceEvent, TraceEventKind
from cogalpha.verification.trace_verifier import verify_cogalpha_trace


def test_verify_cogalpha_trace_accepts_minimal_valid_final_state_and_trace():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    report = verify_cogalpha_trace(state, _valid_trace())

    assert report.passed
    assert report.findings == []


def test_fake_complete_node_history_without_tool_result_events_is_rejected():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        _event(0, TraceEventKind.RUN_STARTED),
        _event(1, TraceEventKind.STOP_DECISION, reason="generation_limit"),
        _event(2, TraceEventKind.RUN_FINISHED),
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "missing_tool_result" in _codes(report)


def test_evaluated_final_pool_candidate_without_fitness_event_is_rejected():
    state = _final_state(stage=CandidateStage.ELITE)
    events = [
        event
        for event in _valid_trace()
        if event.kind != TraceEventKind.FITNESS_EVALUATION_RECORDED
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "missing_fitness_evaluation" in _codes(report)


def test_qualified_candidate_with_failed_fitness_event_is_rejected():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        _replace_event_payload(
            event,
            status="error",
            stage=CandidateStage.QUALIFIED,
            error="metrics failed",
        )
        if event.kind == TraceEventKind.FITNESS_EVALUATION_RECORDED
        else event
        for event in _valid_trace()
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "failed_fitness_evaluation" in _codes(report)


def test_qualified_candidate_without_fitness_metrics_is_rejected():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        _replace_event_payload(event, status="ok", stage=CandidateStage.QUALIFIED)
        if event.kind == TraceEventKind.FITNESS_EVALUATION_RECORDED
        else event
        for event in _valid_trace()
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "missing_fitness_metrics" in _codes(report)


def test_qualified_candidate_requires_single_successful_complete_fitness_event():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        event
        for event in _valid_trace()
        if event.kind != TraceEventKind.FITNESS_EVALUATION_RECORDED
    ]
    events.extend(
        [
            _event(
                7,
                TraceEventKind.FITNESS_EVALUATION_RECORDED,
                candidate_id="alpha-1",
                status="error",
                stage=CandidateStage.QUALIFIED,
                metrics=_fitness_metrics(),
            ),
            _event(
                8,
                TraceEventKind.FITNESS_EVALUATION_RECORDED,
                candidate_id="alpha-1",
                status="ok",
                stage=CandidateStage.REJECTED_BY_FITNESS,
            ),
        ]
    )

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "failed_fitness_evaluation" in _codes(report)
    assert "missing_fitness_metrics" in _codes(report)
    assert "fitness_stage_mismatch" in _codes(report)


def test_final_pool_candidate_with_mismatched_trace_stage_is_rejected():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        _replace_event_payload(
            event,
            status="ok",
            stage=CandidateStage.REJECTED_BY_FITNESS,
        )
        if event.kind == TraceEventKind.FITNESS_EVALUATION_RECORDED
        else event
        for event in _valid_trace()
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "fitness_stage_mismatch" in _codes(report)


def test_rejected_pool_candidate_with_qualified_trace_stage_is_rejected():
    state = _final_state(stage=CandidateStage.REJECTED_BY_FITNESS)
    events = [
        _replace_event_payload(event, status="ok", stage=CandidateStage.QUALIFIED)
        if event.kind == TraceEventKind.FITNESS_EVALUATION_RECORDED
        else event
        for event in _valid_trace()
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "fitness_stage_mismatch" in _codes(report)


def test_accepted_candidate_without_guard_or_quality_evidence_is_rejected():
    state = CogAlphaState(
        candidates=[_candidate("alpha-1", CandidateStage.ACCEPTED_BY_QUALITY)],
        node_history=[DAGNodeResult(node_name="domain_agents")],
    )
    events = [
        _event(0, TraceEventKind.RUN_STARTED),
        _event(
            1,
            TraceEventKind.TOOL_CALL_FINISHED,
            tool_name="domain_agents",
            status="ok",
        ),
        _event(
            2,
            TraceEventKind.CANDIDATE_STAGE_CHANGED,
            candidate_id="alpha-1",
            stage=CandidateStage.GENERATED,
        ),
        _event(3, TraceEventKind.STOP_DECISION, reason="controller_stop"),
    ]

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "missing_quality_evidence" in _codes(report)


def test_stop_decision_without_reason_is_rejected():
    state = _final_state(stage=CandidateStage.QUALIFIED)
    events = [
        event
        for event in _valid_trace()
        if event.kind != TraceEventKind.STOP_DECISION
    ]
    events.append(_event(9, TraceEventKind.STOP_DECISION))

    report = verify_cogalpha_trace(state, events)

    assert not report.passed
    assert "missing_stop_reason" in _codes(report)


def _valid_trace() -> list[CogAlphaTraceEvent]:
    return [
        _event(0, TraceEventKind.RUN_STARTED),
        _event(
            1,
            TraceEventKind.TOOL_CALL_FINISHED,
            tool_name="domain_agents",
            status="ok",
        ),
        _event(
            2,
            TraceEventKind.TOOL_CALL_FINISHED,
            tool_name="quality_pipeline",
            status="ok",
        ),
        _event(
            3,
            TraceEventKind.TOOL_CALL_FINISHED,
            tool_name="fitness_gate",
            status="ok",
        ),
        _event(
            4,
            TraceEventKind.CANDIDATE_STAGE_CHANGED,
            candidate_id="alpha-1",
            stage=CandidateStage.GENERATED,
        ),
        _event(
            5,
            TraceEventKind.GUARD_REPORT_RECORDED,
            candidate_id="alpha-1",
            status="pass",
        ),
        _event(
            6,
            TraceEventKind.SKILL_INVOCATION_FINISHED,
            candidate_id="alpha-1",
            skill_kind="quality_checker",
            status="ok",
        ),
        _event(
            7,
            TraceEventKind.FITNESS_EVALUATION_RECORDED,
            candidate_id="alpha-1",
            status="ok",
            metrics=_fitness_metrics(),
            stage=CandidateStage.QUALIFIED,
        ),
        _event(8, TraceEventKind.STOP_DECISION, reason="generation_limit"),
        _event(9, TraceEventKind.RUN_FINISHED),
    ]


def _final_state(*, stage: CandidateStage) -> CogAlphaState:
    candidate = _candidate("alpha-1", stage)
    if stage == CandidateStage.REJECTED_BY_FITNESS:
        return CogAlphaState(
            candidates=[],
            rejected_pool=[candidate],
            node_history=[
                DAGNodeResult(node_name="domain_agents"),
                DAGNodeResult(node_name="quality_pipeline"),
                DAGNodeResult(node_name="fitness_gate"),
            ],
        )
    return CogAlphaState(
        candidates=[],
        qualified_pool=[candidate],
        elite_pool=[candidate] if stage == CandidateStage.ELITE else [],
        node_history=[
            DAGNodeResult(node_name="domain_agents"),
            DAGNodeResult(node_name="quality_pipeline"),
            DAGNodeResult(node_name="fitness_gate"),
        ],
    )


def _candidate(candidate_id: str, stage: CandidateStage) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id.replace('-', '_')}",
            code="def factor_alpha_1(df):\n    return df['close'] - df['open']\n",
            rationale="Test alpha.",
        ),
        stage=stage,
    )


def _event(sequence: int, kind: TraceEventKind, **payload: object) -> CogAlphaTraceEvent:
    return CogAlphaTraceEvent(
        run_id="run-1",
        sequence=sequence,
        kind=kind,
        payload=payload,
    )


def _fitness_metrics() -> dict[str, float]:
    return {
        "ic": 0.08,
        "rank_ic": 0.07,
        "icir": 1.2,
        "rank_icir": 1.1,
        "mi": 0.03,
    }


def _replace_event_payload(
    event: CogAlphaTraceEvent,
    **payload: object,
) -> CogAlphaTraceEvent:
    return CogAlphaTraceEvent(
        run_id=event.run_id,
        sequence=event.sequence,
        kind=event.kind,
        payload={
            "candidate_id": "alpha-1",
            **payload,
        },
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}
