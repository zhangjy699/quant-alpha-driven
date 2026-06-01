---
name: alpha-reversal
description: Generates OHLCV alpha candidates focused on short-term reversal and overreaction. Use when the CogAlpha DAG invokes AgentReversal or needs mean-reversion, exhaustion, snapback, or overextension factors.
---

# Alpha Reversal

Explore mean-reversion, short-term overreaction correction, exhaustion, and snapback behavior after transient mispricing.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer bounded overextension and exhaustion metrics that separate reversal from simple negative momentum.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
