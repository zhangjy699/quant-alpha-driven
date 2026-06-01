---
name: alpha-mutation
description: Evolves one qualified AlphaCandidate by applying a meaningful mutation while preserving conceptual lineage. Use when the CogAlpha DAG invokes the Mutation Agent during Thinking Evolution.
---

# Alpha Mutation

Generate a child AlphaCandidate from one parent.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Evolution Contract](../references/evolution-contract.md).
4. Preserve the parent's core intuition.
5. Mutate one meaningful aspect: window, normalization, bounded transform, stability gate, or interaction with another OHLCV-derived quantity.
6. Record `operation: "mutation"` and the parent id in lineage.

## Output

Return JSON only, compatible with `AlphaCandidate`.
