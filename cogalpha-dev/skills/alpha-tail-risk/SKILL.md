---
name: alpha-tail-risk
description: Generates OHLCV alpha candidates focused on tail risk and stress accumulation. Use when the CogAlpha DAG invokes AgentTailRisk or needs downside shock, fragility, extreme move, or tail-exposure factors.
---

# Alpha Tail Risk

Explore tail-risk exposure, downside stress, asymmetric shock behavior, and accumulation of fragile conditions before large adverse moves.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer bounded downside-pressure, extreme-range, and stress-persistence measures with clear economic rationale.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
