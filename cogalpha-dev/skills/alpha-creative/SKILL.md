---
name: alpha-creative
description: Generates OHLCV alpha candidates focused on novel nonlinear feature representations. Use when the CogAlpha DAG invokes AgentCreative or needs unconventional, bounded, transformed, or reparameterized OHLCV factors.
---

# Alpha Creative

Explore novel but interpretable transformations, reparameterizations, soft gates, bounded nonlinearities, and alternative representations of OHLCV behavior.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Be inventive without becoming decorative: every transform must improve robustness, interpretability, or signal shape.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
