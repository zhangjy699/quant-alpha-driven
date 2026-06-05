from cogalpha.tracing import (
    CogAlphaTraceEvent,
    TraceEventKind,
    TraceLedger,
    read_trace_events,
    write_trace_events,
)


def test_trace_ledger_writes_jsonl_and_round_trips_events_in_sequence_order(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger = TraceLedger(run_id="run-1", path=trace_path)

    first = ledger.record(
        TraceEventKind.RUN_STARTED,
        payload={"config": "fixture"},
    )
    second = ledger.record(
        TraceEventKind.AGENT_DECISION,
        payload={"tool": "domain_agents.generate"},
    )

    loaded = read_trace_events(trace_path)

    assert [event.sequence for event in loaded] == [0, 1]
    assert [event.sequence for event in loaded] == [first.sequence, second.sequence]
    assert [event.run_id for event in loaded] == ["run-1", "run-1"]
    assert [event.kind for event in loaded] == [
        TraceEventKind.RUN_STARTED,
        TraceEventKind.AGENT_DECISION,
    ]
    assert loaded[0].payload == {"config": "fixture"}
    assert loaded[1].payload == {"tool": "domain_agents.generate"}


def test_write_trace_events_preserves_explicit_event_order(tmp_path):
    trace_path = tmp_path / "manual-trace.jsonl"
    events = [
        CogAlphaTraceEvent(
            run_id="run-2",
            sequence=4,
            kind=TraceEventKind.TOOL_CALL_STARTED,
            payload={"tool_call_id": "call-1"},
        ),
        CogAlphaTraceEvent(
            run_id="run-2",
            sequence=5,
            kind=TraceEventKind.TOOL_CALL_FINISHED,
            payload={"tool_call_id": "call-1", "status": "ok"},
        ),
    ]

    write_trace_events(trace_path, events)

    loaded = read_trace_events(trace_path)
    assert [event.sequence for event in loaded] == [4, 5]
    assert loaded[0].kind == TraceEventKind.TOOL_CALL_STARTED
    assert loaded[1].payload == {"tool_call_id": "call-1", "status": "ok"}
