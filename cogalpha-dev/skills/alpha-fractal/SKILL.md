---
name: alpha-fractal
description: Generates OHLCV alpha candidates focused on multi-scale roughness and long-memory structure. Use when the CogAlpha DAG invokes AgentFractal or needs fractal, roughness, cross-horizon variability, or scale-irregularity factors.
---

# Alpha Fractal

Explore multi-scale roughness, cross-horizon variability, irregularity, and long-memory characteristics in OHLCV time series.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer simple cross-window contrasts and roughness proxies over expensive or opaque fractal estimators.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
