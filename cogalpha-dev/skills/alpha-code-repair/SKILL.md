---
name: alpha-code-repair
description: Repairs generated alpha function code after deterministic guard or code-quality failure. Use when the CogAlpha DAG invokes the Code Repair Agent with an AlphaCandidate and concrete error feedback.
---

# Alpha Code Repair

Repair one AlphaCandidate while preserving its core hypothesis when possible.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Quality Checking](../references/quality-checking.md).
4. Apply the provided guard/code-quality feedback exactly.
5. Return a `QualityDecision` with `verdict: "repair"` and `repaired_candidate` when a valid repair is possible.

## Rules

- Do not introduce non-OHLCV columns.
- Do not broaden the factor into a different theme unless the original is unrecoverable.
- Keep lineage and candidate metadata traceable.

## Output

Return JSON only, compatible with `QualityDecision`.
