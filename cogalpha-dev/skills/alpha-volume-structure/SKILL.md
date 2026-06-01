---
name: alpha-volume-structure
description: Generates OHLCV alpha candidates focused on temporal volume structure. Use when the CogAlpha DAG invokes AgentVolumeStructure or needs abnormal volume, volume rhythm, persistence, exhaustion, or volume-shock factors.
---

# Alpha Volume Structure

Explore volume persistence, abnormal volume, exhaustion, rhythm, and volume-state changes in relation to price behavior.

## Workflow

1. Read [Alpha Factor Contract](../references/alpha-factor-contract.md).
2. Read [Structured Artifacts](../references/structured-artifacts.md).
3. Apply the requested guidance mode from [Diversified Guidance](../references/diversified-guidance.md), if provided.
4. Generate compact AlphaCandidates that use OHLCV Input only.
5. Prefer stable volume ratios, smoothed shocks, and exhaustion signals over raw volume levels.

## Output

Return JSON only, compatible with `AlphaCandidateBatch`: `{"candidates": [...]}`.
