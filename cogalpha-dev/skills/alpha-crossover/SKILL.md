---
name: alpha-crossover
description: Evolves two qualified AlphaCandidates by combining complementary insights into a new child factor. Use when the CogAlpha DAG invokes the Crossover Agent during Thinking Evolution.
---

# Alpha Crossover

## Role

Alpha Crossover is a Thinking Evolution operator that must fuse complementary qualified parents while preserving conceptual lineage and creating a child with a clear parent-strength target. It should follow QuantaAlpha-style trajectory evolution: parent evidence, targeted change, evaluation feedback, and reusable library signal.

## Inputs

- Runtime payload assembled by SkillInvoker, including candidate, generation, guidance mode, prior feedback, and schema request as applicable.
- [Alpha Factor Contract](../references/alpha-factor-contract.md) for allowed OHLCV inputs, code shape, and hard safety rules.
- [Structured Artifacts](../references/structured-artifacts.md) for strict JSON output schemas.
- [Agentic Workflow](../references/agentic-workflow.md) for observe-plan-generate-self-check-output discipline.
- [Metric Objectives](../references/metric-objectives.md) for IC, RankIC, ICIR, RankICIR, MI, qualified minima, and elite minima.
- [Trace-Grounded Learning](../references/trace-grounded-learning.md) for evidence_id, reviewer, rollback, and utility boundaries.
- [Evolution Contract](../references/evolution-contract.md) for mutation, crossover, and child lineage rules.

- [Operator Library](../references/operator-library.md) for common operator primitives and safe usage patterns.

## Workflow

1. Observe two parent candidates, parent metrics, feedback summaries, generation, and requested operation.
2. Identify parent_strength_target: the specific IC, RankIC, ICIR, RankICIR, MI, robustness, or interpretability strength to preserve from the parent set.
3. Identify the bottleneck to improve without destroying the parent strength.
4. Generate one child AlphaCandidate with operation: crossover, parent ids, generation, agent_skill: alpha-crossover, and traceable lineage.
5. Avoid superficial renaming, naive averaging, or unrelated feature stacking.

## Anti-Leakage Rules

- Use only OHLCV-derived parent logic and provided feedback; do not use future labels, hidden validation outcomes, or sealed test information.
- Do not copy parent code with decorative variable renames.
- Do not introduce imports, file IO, network IO, subprocesses, or mutable global state.

## Metric Objective

Evolution should preserve at least one parent strength while targeting a clear metric bottleneck across IC, RankIC, ICIR, RankICIR, and MI. A child that improves MI but collapses RankIC, or improves RankIC but loses all IC, should be treated as diagnostic rather than promotion-grade.

## Self-Check

- Confirm lineage.parent_ids includes the required parent ids and lineage.operation is crossover.
- Confirm parent_strength_target is present in metadata or rationale.
- Confirm the child obeys the Alpha Factor Contract and remains compact and interpretable.
- Confirm output is strict JSON compatible with AlphaCandidate.

## Trace Expectations

Runtime traces should record skill_name: alpha-crossover, candidate_id, parent_ids, lineage, parent_strength_target, generation, stage, and evidence_id. Utility updates require downstream quality and fitness evidence before skill changes can be proposed.

## Output

Return JSON only, compatible with AlphaCandidate.
