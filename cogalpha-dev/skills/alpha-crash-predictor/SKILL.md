---
name: alpha-crash-predictor
description: Generates OHLCV alpha candidates focused on crash precursors and regime breakdown risk. Use when the CogAlpha DAG invokes AgentCrashPredictor or needs fragility, breakdown, drawdown acceleration, or crash-warning factors.
---

# Alpha Crash Predictor

Explore early-warning patterns for potential crash regimes, including stress buildup, failed rebounds, range expansion, and liquidity-thin price drops.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer interpretable breakdown precursors over direct future-return proxies.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
