---
name: alpha-lag-response
description: Generates OHLCV alpha candidates focused on delayed price adjustment. Use when the CogAlpha DAG invokes AgentLagResponse or needs lagged response, delayed volatility-volume feedback, or slow-adjustment factors.
---

# Alpha Lag Response

Explore delayed adjustment between returns, volatility, range, and volume using past and present observations only.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer causal lag structures with positive shifts or rolling summaries; never use future shifts.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
