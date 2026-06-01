---
name: alpha-herding
description: Generates OHLCV alpha candidates focused on herding and crowding behavior inferred from OHLCV dynamics. Use when the CogAlpha DAG invokes AgentHerding or needs consensus, crowding, directional alignment, or participation-intensity factors.
---

# Alpha Herding

Infer collective crowding, consensus intensity, and directional alignment from volume, bar direction, range, and price persistence.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Use OHLCV proxies carefully; do not claim unavailable order-flow or investor-position data.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
