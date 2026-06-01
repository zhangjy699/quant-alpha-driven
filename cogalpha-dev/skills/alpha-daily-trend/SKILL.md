---
name: alpha-daily-trend
description: Generates OHLCV alpha candidates focused on daily trend persistence and momentum strength. Use when the CogAlpha DAG invokes AgentDailyTrend or needs directional persistence, trend quality, or multi-day momentum factors.
---

# Alpha Daily Trend

Explore persistent directional movement, trend quality, multi-day momentum strength, and trend smoothness from daily OHLCV data.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer trend quality, slope stability, and risk-adjusted momentum over plain returns.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
