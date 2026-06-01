---
name: alpha-judge
description: Semantically judges whether an AlphaCandidate is logically, technically, and economically sound enough for fitness evaluation. Use when the CogAlpha DAG invokes the Judge Agent after deterministic guards pass.
---

# Alpha Judge

Evaluate practical soundness. This skill does not decide execution safety or leakage alone; Deterministic Guards are the hard gate.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Quality Checking](../references/quality-checking.md).
4. Assess economic interpretability, logical consistency, technical correctness, efficiency, and whether the factor is worth testing.
5. Return a `QualityDecision`.

## Verdict Guidance

- Use `accept` for factors that are coherent and test-worthy.
- Use `repair` for factors with a viable core idea but weak implementation or rationale.
- Use `reject` for fabricated, non-economic, redundant, or incoherent factors.

## Output

Return JSON only, compatible with `QualityDecision`.
