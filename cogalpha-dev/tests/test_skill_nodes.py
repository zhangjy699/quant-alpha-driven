import json

from cogalpha.schemas import (
    AlphaCandidate,
    AlphaCandidateBatch,
    AlphaFunction,
    DomainAgentRequest,
    QualitySkillRequest,
)
from cogalpha.skill_nodes import SkillNodeRuntime


class CapturingInvoker:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append((skill_name, runtime_payload, output_schema))
        return AlphaCandidateBatch(candidates=[])


def test_skill_node_runtime_serializes_runtime_schema_requests():
    invoker = CapturingInvoker()
    runtime = SkillNodeRuntime(invoker)
    request = DomainAgentRequest(
        skill_name="alpha-market-cycle",
        paper_agent_name="AgentMarketCycle",
        level=1,
        layer="Market Structure & Cycle",
        focus="Cycle state transitions.",
    )

    batch = runtime.candidate_batch("alpha-market-cycle", request)

    assert batch.candidates == []
    skill_name, runtime_payload, output_schema = invoker.calls[0]
    assert skill_name == "alpha-market-cycle"
    assert json.loads(runtime_payload)["paper_agent_name"] == "AgentMarketCycle"
    assert output_schema is AlphaCandidateBatch


def test_skill_node_runtime_omits_private_candidate_payload_fields():
    invoker = CapturingInvoker()
    runtime = SkillNodeRuntime(invoker)
    candidate = AlphaCandidate(
        candidate_id="candidate-1",
        alpha=AlphaFunction(
            name="factor_close_open_gap",
            code="def factor_close_open_gap(df):\n    return df['close'] - df['open']\n",
            rationale="Test alpha.",
        ),
        metadata={"private_debug": "do not send"},
    )
    request = QualitySkillRequest(candidate=candidate)

    runtime.quality_decision("alpha-code-quality", request)

    payload = json.loads(invoker.calls[0][1])
    assert "created_at" not in payload["candidate"]
    assert "metadata" not in payload["candidate"]
