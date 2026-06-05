from cogalpha.config import MVPLoopConfig
from cogalpha.nodes import DomainAgentNode
from cogalpha.registry import DOMAIN_AGENT_SPECS
from cogalpha.schemas import AlphaCandidate, AlphaCandidateBatch, AlphaFunction, CogAlphaState


class FakeBatchInvoker:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append((skill_name, runtime_payload, output_schema))
        return AlphaCandidateBatch(
            candidates=[
                AlphaCandidate(
                    candidate_id=f"{skill_name}-candidate",
                    alpha=AlphaFunction(
                        name=f"factor_{skill_name.replace('-', '_')}",
                        code=(
                            f"def factor_{skill_name.replace('-', '_')}(df):\n"
                            "    df_copy = df.copy()\n"
                            f"    df_copy['factor_{skill_name.replace('-', '_')}'] = "
                            "df_copy['close'] - df_copy['open']\n"
                            f"    return df_copy['factor_{skill_name.replace('-', '_')}']\n"
                        ),
                        rationale="Measures intraday body direction.",
                    ),
                )
            ]
        )


def test_domain_agent_node_invokes_all_domain_skills():
    invoker = FakeBatchInvoker()
    node = DomainAgentNode(invoker=invoker, config=MVPLoopConfig())

    result = node(CogAlphaState().model_dump(mode="python"))
    state = CogAlphaState.model_validate(result)

    assert len(invoker.calls) == len(DOMAIN_AGENT_SPECS)
    assert len(state.candidates) == len(DOMAIN_AGENT_SPECS)
    assert state.node_history[-1].node_name == "domain_agents"
    assert state.node_history[-1].metadata["candidates_generated"] == 21

