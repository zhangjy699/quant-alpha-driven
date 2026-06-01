---
name: alpha-code-quality
description: Reviews generated alpha function code for syntax, pandas usage, output shape, naming, and complexity before repair or semantic judgment. Use when the CogAlpha DAG invokes the Code Quality Agent for an AlphaCandidate.
---

# Alpha Code Quality

Perform first-pass semantic code review for one AlphaCandidate. Deterministic Guards enforce hard syntax, leakage, and execution checks; this skill explains fixable code-quality problems and format issues.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Read [Quality Checking](../references/quality-checking.md).
4. Inspect function shape, naming, pandas assignment style, undefined variables, redundant transforms, and excessive complexity.
5. Return a `QualityDecision`.

## Verdict Guidance

- Use `accept` when the code is clean enough for deterministic guards and semantic judgment.
- Use `repair` when issues are concrete and fixable.
- Use `reject` only when the implementation is structurally incoherent.

## Output

Return JSON only, compatible with `QualityDecision`.
