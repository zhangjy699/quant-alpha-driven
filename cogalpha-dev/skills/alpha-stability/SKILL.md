---
name: alpha-stability
description: Generates OHLCV alpha candidates focused on temporal stability and persistence. Use when the CogAlpha DAG invokes AgentStability or needs consistency, smoothness, signal persistence, or robustness factors.
---

# Alpha Stability

Explore temporal consistency, persistence, smoothness, and robust activation of price, range, or volume signals.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer stability gates, persistence ratios, and robust dispersion measures over over-smoothed signals.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
