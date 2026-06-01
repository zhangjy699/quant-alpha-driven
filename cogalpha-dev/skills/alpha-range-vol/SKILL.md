---
name: alpha-range-vol
description: Generates OHLCV alpha candidates focused on range-based volatility dynamics. Use when the CogAlpha DAG invokes AgentRangeVol or needs range compression, expansion, realized range, or volatility-energy factors.
---

# Alpha Range Vol

Explore daily high-low range dynamics, compression-expansion cycles, volatility energy, and normalized range behavior.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer range-normalized volatility signals with stable denominators and no target leakage.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
