---
name: alpha-logic-improvement
description: Improves alpha factor logic after Judge feedback while preserving the candidate's core modeling intent. Use when the CogAlpha DAG invokes the Logic Improvement Agent for a rejected-but-repairable AlphaCandidate.
---

# Alpha Logic Improvement

Refine a candidate with weak logic into a cleaner, more interpretable, and more robust AlphaCandidate.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Quality Checking](../references/quality-checking.md).
4. Address the Judge feedback directly.
5. Return a `QualityDecision` with a repaired candidate when improvement succeeds.

## Rules

- Preserve the candidate's core hypothesis unless the feedback says it is invalid.
- Prefer simpler logic over stacked transformations.
- Improve economic rationale together with code.

## Output

Return JSON only, compatible with `QualityDecision`.
