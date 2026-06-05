from cogalpha.reporting import (
    EvaluationRunReport,
    ReportLayer,
    StopGoDecision,
    build_agentic_run_report,
    write_evaluation_run_report,
)
from cogalpha.verification.trace_verifier import TraceVerificationReport


def test_evaluation_run_report_records_layered_stop_go(tmp_path):
    report = EvaluationRunReport(
        report_id="preflight",
        purpose="fixed preflight",
        data_version="data-v1",
        manifest_path="outputs/preflight/run_manifest.json",
        layers=[
            ReportLayer(
                name="validation",
                status=StopGoDecision.GO,
                summary="Schema and data artifacts validated.",
            ),
            ReportLayer(
                name="promotion",
                status=StopGoDecision.HOLD,
                summary="Promotion awaits approval.",
            ),
        ],
        decision=StopGoDecision.HOLD,
        blockers=["No promotion approval."],
    )

    output = tmp_path / "report.json"
    write_evaluation_run_report(output, report)

    assert output.exists()
    assert report.layers[1].status == StopGoDecision.HOLD


def test_agentic_run_report_stops_workflow_when_trace_verification_fails(tmp_path):
    summary = {
        "run_id": "agentic",
        "split": "valid",
        "node_history": ["domain_agents", "quality_pipeline", "fitness_gate"],
        "skill_errors": 0,
        "remaining_candidates": 0,
        "qualified": 1,
        "elite": 1,
        "rejected": 0,
    }
    trace_report = TraceVerificationReport.model_validate(
        {
            "findings": [
                {
                    "code": "missing_tool_result",
                    "severity": "error",
                    "message": "No matching tool result.",
                }
            ]
        }
    )

    report = build_agentic_run_report(
        summary=summary,
        data_version="data-v1",
        manifest_path=tmp_path / "run_manifest.json",
        trace_verification=trace_report,
    )

    workflow_layer = next(layer for layer in report.layers if layer.name == "workflow_execution")
    assert workflow_layer.status == StopGoDecision.STOP
    assert report.decision == StopGoDecision.STOP
    assert any("Trace verification failed" in blocker for blocker in report.blockers)


def test_agentic_run_report_holds_effect_when_zero_qualified_but_trace_passes(tmp_path):
    summary = {
        "run_id": "agentic",
        "split": "valid",
        "node_history": ["domain_agents", "quality_pipeline", "fitness_gate"],
        "skill_errors": 0,
        "remaining_candidates": 0,
        "qualified": 0,
        "elite": 0,
        "rejected": 1,
    }
    trace_report = TraceVerificationReport(findings=[])

    report = build_agentic_run_report(
        summary=summary,
        data_version="data-v1",
        manifest_path=tmp_path / "run_manifest.json",
        trace_verification=trace_report,
    )

    workflow_layer = next(layer for layer in report.layers if layer.name == "workflow_execution")
    effect_layer = next(layer for layer in report.layers if layer.name == "effect_evaluation")
    promotion_layer = next(layer for layer in report.layers if layer.name == "promotion_governance")
    assert workflow_layer.status == StopGoDecision.GO
    assert effect_layer.status == StopGoDecision.HOLD
    assert promotion_layer.status == StopGoDecision.HOLD
    assert report.decision == StopGoDecision.HOLD
