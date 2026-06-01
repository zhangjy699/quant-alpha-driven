---
name: alpha-volatility-regime
description: Generates OHLCV alpha candidates focused on volatility regimes and state transitions. Use when the CogAlpha DAG invokes AgentVolatilityRegime or needs regime-dependent volatility, compression, expansion, or volatility-state factors.
---

# Alpha Volatility Regime

Explore volatility states, regime transitions, volatility clustering, and state-dependent signal activation inferred from daily OHLCV behavior.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer adaptive volatility-state measures, normalized ranges, and regime shifts over generic rolling volatility.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
