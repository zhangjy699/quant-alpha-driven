---
name: alpha-composite
description: Generates OHLCV alpha candidates focused on composite factor construction and information fusion. Use when the CogAlpha DAG invokes AgentComposite or needs synergy, orthogonalization, fusion, or regime-weighted composite factors.
---

# Alpha Composite

Fuse multiple independent OHLCV-derived signals into coherent composites that emphasize synergy, de-noising, and orthogonal information.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Avoid simple sums; prefer interpretable interactions, gates, normalized fusion, or redundancy reduction.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
