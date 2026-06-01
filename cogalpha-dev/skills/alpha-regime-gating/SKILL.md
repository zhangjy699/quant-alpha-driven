---
name: alpha-regime-gating
description: Generates OHLCV alpha candidates focused on adaptive regime gates. Use when the CogAlpha DAG invokes AgentRegimeGating or needs volatility, trend, liquidity, or state-dependent signal gating factors.
---

# Alpha Regime Gating

Explore adaptive gates that modulate signal activation based on volatility, trend, range, or liquidity states.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer smooth gates such as bounded ratios or logistic-like transforms over brittle if/else regimes.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
