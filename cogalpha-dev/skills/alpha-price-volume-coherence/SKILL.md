---
name: alpha-price-volume-coherence
description: Generates OHLCV alpha candidates focused on coherence between price movement and volume behavior. Use when the CogAlpha DAG invokes AgentPriceVolumeCoherence or needs confirmation, divergence, or price-volume alignment factors.
---

# Alpha Price Volume Coherence

Explore whether price movement is confirmed, contradicted, amplified, or weakened by volume dynamics.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer divergence, confirmation, and normalized co-movement metrics with stable denominators.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
