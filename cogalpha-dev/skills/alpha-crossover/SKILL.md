---
name: alpha-crossover
description: Evolves two qualified AlphaCandidates by combining complementary insights into a new child factor. Use when the CogAlpha DAG invokes the Crossover Agent during Thinking Evolution.
---

# Alpha Crossover

Generate a child AlphaCandidate from two parents.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Evolution Contract](../references/evolution-contract.md).
4. Identify the complementary insight from each parent.
5. Combine them through an interpretable interaction, gate, normalization, or regime dependency.
6. Record `operation: "crossover"` and both parent ids in lineage.

## Output

Return JSON only, compatible with `AlphaCandidate`.
