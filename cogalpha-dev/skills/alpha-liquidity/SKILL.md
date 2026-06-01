---
name: alpha-liquidity
description: Generates OHLCV alpha candidates focused on liquidity and price impact. Use when the CogAlpha DAG invokes AgentLiquidity or needs illiquidity, volume-adjusted movement, price-impact, or turnover-pressure factors.
---

# Alpha Liquidity

Explore liquidity, price impact per unit volume, volume-adjusted movement, and thin-market behavior using daily OHLCV data.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer robust volume normalization, bounded transforms, and clear illiquidity intuition.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
