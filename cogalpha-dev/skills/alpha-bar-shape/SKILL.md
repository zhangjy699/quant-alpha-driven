---
name: alpha-bar-shape
description: Generates OHLCV alpha candidates focused on candlestick geometry and bar-shape patterns. Use when the CogAlpha DAG invokes AgentBarShape or needs body, shadow, symmetry, wick, or candle-location factors.
---

# Alpha Bar Shape

Translate candlestick geometry into continuous numerical signals: body, shadow, symmetry, close location, and shape persistence.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer continuous, smooth, interpretable geometry metrics over binary candlestick labels.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
