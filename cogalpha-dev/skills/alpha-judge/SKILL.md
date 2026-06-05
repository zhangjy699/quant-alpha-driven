---
name: alpha-judge
description: Semantically judges whether an AlphaCandidate is logically, technically, and economically sound enough for fitness evaluation. Use when the CogAlpha DAG invokes the Judge Agent after deterministic guards pass.
---

# Alpha Judge

## Role

Judge whether an AlphaCandidate is logically, technically, and economically sound enough to advance to fitness evaluation. This quality skill converts candidate review into an explicit accept, repair, or reject decision while preserving structured feedback for later agentic traces.

## Inputs

- Runtime payload assembled by SkillInvoker, including candidate, generation, guidance mode, prior feedback, and schema request as applicable.
- [Alpha Factor Contract](../references/alpha-factor-contract.md) for allowed OHLCV inputs, code shape, and hard safety rules.
- [Structured Artifacts](../references/structured-artifacts.md) for strict JSON output schemas.
- [Agentic Workflow](../references/agentic-workflow.md) for observe-plan-generate-self-check-output discipline.
- [Metric Objectives](../references/metric-objectives.md) for IC, RankIC, ICIR, RankICIR, MI, qualified minima, and elite minima.
- [Trace-Grounded Learning](../references/trace-grounded-learning.md) for evidence_id, reviewer, rollback, and utility boundaries.
- [Quality Checking](../references/quality-checking.md) for verdict semantics and review priorities.

## Workflow

1. Observe the candidate id, alpha code, rationale, guard reports, previous quality decisions, repair attempt, and feedback.
2. Plan the narrowest useful verdict: accept if the candidate is coherent and test-worthy, repair if the idea is viable but implementation or rationale needs correction, reject if the mechanism is invalid, unsafe, leaky, redundant, or incoherent.
3. Generate a QualityDecision with concrete feedback tied to the candidate and guard evidence.
4. For repair, preserve candidate identity or lineage clearly and change only what is needed to satisfy the contract.
5. For reject, explain the blocking reason so future generation can avoid the failure mode.

## Anti-Leakage Rules

- Do not use future returns, labels, hidden evaluator data, or sealed test outcomes to justify accept, repair, or reject.
- Do not relax deterministic guard constraints; syntax, restricted execution, leakage, and numerical safety remain hard gates.
- Do not invent metrics that were not provided in the payload.

## Metric Objective

Quality decisions should improve the chance that accepted or repaired candidates clear IC, RankIC, ICIR, RankICIR, and MI gates without overfitting to a single diagnostic. Reject candidates whose apparent metric promise depends on leakage, unstable arithmetic, or incoherent economic logic.

## Self-Check

- Confirm the verdict is exactly accept, repair, or reject.
- Confirm feedback is actionable and does not ask the user to inspect unavailable lines manually.
- Confirm any repaired candidate remains schema-valid and traceable to the original candidate.
- Confirm output is strict JSON compatible with QualityDecision.

## Trace Expectations

Runtime traces should record skill_name: alpha-judge, candidate_id, action, verdict, stage, repair_attempt, and evidence_id. Review evidence may update skill utility only after downstream candidate lifecycle and fitness outcomes are observed.

## Output

Return JSON only, compatible with QualityDecision.
