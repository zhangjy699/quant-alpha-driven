---
name: alpha-drawdown
description: Generates OHLCV alpha candidates focused on drawdown and recovery geometry. Use when the CogAlpha DAG invokes AgentDrawdown or needs drawdown depth, duration, recovery, resilience, or underwater-state factors.
---

# Alpha Drawdown

Explore cumulative loss depth, recovery geometry, drawdown persistence, and resilience after price declines.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer causal rolling peaks and recovery measures; never inspect future peaks or troughs.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
