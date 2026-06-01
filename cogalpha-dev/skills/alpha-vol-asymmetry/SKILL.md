---
name: alpha-vol-asymmetry
description: Generates OHLCV alpha candidates focused on asymmetric volatility between up and down moves. Use when the CogAlpha DAG invokes AgentVolAsymmetry or needs downside/upside volatility skew or asymmetric risk factors.
---

# Alpha Vol Asymmetry

Explore skewed risk behavior by comparing volatility, range, body, and volume response on upward versus downward price moves.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer interpretable asymmetry measures that are stable when returns or ranges are near zero.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
