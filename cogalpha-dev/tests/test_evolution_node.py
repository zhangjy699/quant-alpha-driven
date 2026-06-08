from cogalpha.config import MVPLoopConfig
from cogalpha.nodes import EvolutionNode
from cogalpha.schemas import AlphaCandidate, AlphaFunction, CogAlphaState, EvolutionOperation


def make_candidate(candidate_id: str) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=candidate_id,
        alpha=AlphaFunction(
            name=f"factor_{candidate_id}",
            code=f"def factor_{candidate_id}(df):\n    return df['close'] - df['open']\n",
            rationale=f"{candidate_id} rationale.",
        ),
    )


class FakeEvolutionInvoker:
    def __init__(self) -> None:
        self.calls = []
        self.counter = 0

    def invoke(self, skill_name, runtime_payload, output_schema):
        self.calls.append((skill_name, runtime_payload, output_schema))
        self.counter += 1
        return make_candidate(f"child_{self.counter}")


def test_evolution_node_generates_mutation_and_crossover_children():
    invoker = FakeEvolutionInvoker()
    node = EvolutionNode(invoker=invoker, config=MVPLoopConfig())
    state = CogAlphaState(parent_pool=[make_candidate("parent_a"), make_candidate("parent_b")])

    result = CogAlphaState.model_validate(node(state.model_dump(mode="python")))

    assert result.generation == 1
    assert [call[0] for call in invoker.calls] == [
        "alpha-mutation",
        "alpha-mutation",
        "alpha-crossover",
        "alpha-mutation",
    ]
    assert len(result.candidates) == 4
    assert result.candidates[0].lineage.operation == EvolutionOperation.MUTATION
    assert result.candidates[2].lineage.operation == EvolutionOperation.CROSSOVER
    assert result.candidates[3].lineage.operation == EvolutionOperation.CROSSOVER_THEN_MUTATION
