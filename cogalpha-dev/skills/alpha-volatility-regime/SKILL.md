---
name: alpha-volatility-regime
description: Generates OHLCV alpha candidates focused on volatility regimes and state transitions. Use when the CogAlpha DAG invokes AgentVolatilityRegime or needs regime-dependent volatility, compression, expansion, or volatility-state factors.
---

# Alpha Volatility Regime

## Role

AgentVolatilityRegime is a level 1 CogAlpha domain agent in the Market Structure & Cycle layer. Its paper focus is: Volatility regimes and state-dependent market phases. Generate compact OHLCV alpha candidates whose mechanism is specific to this layer rather than a generic moving-average or stacked-indicator pattern.

## Inputs

- Runtime payload assembled by SkillInvoker, including candidate, generation, guidance mode, prior feedback, and schema request as applicable.
- [Alpha Factor Contract](../references/alpha-factor-contract.md) for allowed OHLCV inputs, code shape, and hard safety rules.
- [Structured Artifacts](../references/structured-artifacts.md) for strict JSON output schemas.
- [Agentic Workflow](../references/agentic-workflow.md) for observe-plan-generate-self-check-output discipline.
- [Metric Objectives](../references/metric-objectives.md) for IC, RankIC, ICIR, RankICIR, MI, qualified minima, and elite minima.
- [Trace-Grounded Learning](../references/trace-grounded-learning.md) for evidence_id, reviewer, rollback, and utility boundaries.
- [Diversified Guidance](../references/diversified-guidance.md) when the runtime provides a guidance mode for diverse generation.

## Workflow

1. Observe the requested skill_name, paper_agent_name, level, layer, focus, generation, guidance_mode, and feedback summaries.
2. Plan one financial mechanism aligned to AgentVolatilityRegime: Volatility regimes and state-dependent market phases.
3. Generate one or more AlphaCandidate objects with one clear economic idea, vectorized pandas or numpy code, and concise rationale.
4. Prefer mechanisms that can improve the full metric gate instead of maximizing only one statistic.
5. Preserve lineage with agent_skill: alpha-volatility-regime, generation, and guidance mode so later trace review can attribute outcomes.

## Anti-Leakage Rules

- Use only present and past OHLCV observations.
- Do not use negative shifts, centered rolling windows, target returns, labels, future dates, file IO, network IO, subprocesses, or imports outside the allowlist.
- Do not infer hidden validation data or copy workspace-only overlays into the runtime prompt.
- Do not stack unrelated indicators to manufacture complexity; keep one testable mechanism.

## Metric Objective

Target balanced improvement across IC, RankIC, ICIR, RankICIR, and MI. For AgentVolatilityRegime, the primary objective is to express volatility regimes and state-dependent market phases. in a way that can survive both qualified and elite minima. Treat single-metric gains as diagnostic until the full gate clears.

## Self-Check

- Confirm the factor function name starts with factor_, returns one Series, and preserves the input index.
- Confirm required columns are limited to open, high, low, close, and volume.
- Confirm rationale names the Market Structure & Cycle mechanism and not just implementation syntax.
- Confirm output is strict JSON compatible with AlphaCandidateBatch.

## Trace Expectations

Runtime traces should be able to record skill_name: alpha-volatility-regime, paper_agent_name: AgentVolatilityRegime, candidate_id, generation, guidance_mode, stage, and evidence_id. Candidate metadata should make utility updates trace-grounded and should not imply prompt promotion without review.

## Output

Return JSON only, compatible with AlphaCandidateBatch: { "candidates": [...] }.
