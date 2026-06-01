---
name: alpha-market-cycle
description: Generates OHLCV alpha candidates focused on market cycles and phase-state dynamics. Use when the CogAlpha DAG invokes AgentMarketCycle or needs long-horizon cyclicality, phase angle, trend rhythm, or cycle-turn factors.
---

# Alpha Market Cycle

Explore large-scale temporal structures such as long-term trends, market phases, cyclical state transitions, hidden rhythm, and alternating volatility compression or expansion.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer phase differences, curvature, normalized cycle energy, or oscillation amplitude over ordinary moving-average crossovers.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
