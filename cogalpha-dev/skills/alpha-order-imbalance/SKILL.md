---
name: alpha-order-imbalance
description: Generates OHLCV alpha candidates focused on inferred order imbalance. Use when the CogAlpha DAG invokes AgentOrderImbalance or needs directional pressure, buy-sell imbalance proxy, or intraday pressure factors.
---

# Alpha Order Imbalance

Infer directional pressure from close position, bar body, range, and volume because true order-book imbalance is unavailable in OHLCV Input.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer pressure proxies that combine price location and volume without fabricating unavailable order-book fields.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
