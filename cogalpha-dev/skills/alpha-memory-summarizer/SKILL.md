---
name: alpha-memory-summarizer
description: Compacts bounded factor-pool evidence into traceable domain-agent memory lessons. Use when CogAlpha updates factor_memory after deterministic evidence selection.
---

# Alpha Memory Summarizer

## Role

Compact a small, preselected evidence batch into concise domain-agent lessons. The goal is to improve future alpha generation by preserving mechanism-level success patterns, failure patterns, and avoid patterns without copying the full factor pool into future prompts.

## Inputs

- Runtime payload assembled by the memory updater, including domain_agent, new_evidence, existing patterns, metric_bottlenecks, and max_patterns_per_kind.
- [Metric Objectives](../references/metric-objectives.md) for interpreting IC, RankIC, ICIR, RankICIR, and MI.
- [Trace-Grounded Learning](../references/trace-grounded-learning.md) for evidence discipline and non-promotion boundaries.

## Workflow

1. Observe only the provided bounded evidence and existing patterns.
2. Merge duplicate or near-duplicate lessons while preserving distinct mechanisms.
3. Prefer lessons that explain why a mechanism succeeded or failed across the five-metric gate.
4. Preserve evidence_factor_ids from the input evidence for every lesson.
5. Keep lessons short, operational, and mechanism-level.

## Anti-Leakage Rules

- Do not invent factor IDs, formulas, metrics, outcomes, or domain agents.
- Do not use test results, hidden evaluator data, future returns, labels, or information outside the runtime payload.
- Do not recommend changing validation thresholds, fitness gates, or data splits to make factors pass.
- Do not issue parameter-tuning instructions such as "set window to 17"; write mechanism-level lessons instead.
- Do not copy full factor code into lessons.

## Output Constraints

- Return strict JSON only.
- Every lesson must cite one or more evidence_factor_ids that appear in the runtime payload.
- Do not exceed max_patterns_per_kind for any pattern list.
- If evidence is weak or contradictory, keep the existing pattern or write a conservative lesson.

## Output

Return JSON only, compatible with FactorMemoryCompactionResult:

```json
{
  "success_patterns": [
    {
      "lesson": "Smoothed range-volatility ratios were more stable than raw range amplitude.",
      "evidence_factor_ids": [1, 7]
    }
  ],
  "failure_patterns": [
    {
      "lesson": "Raw range spikes failed RankICIR despite nonzero IC.",
      "evidence_factor_ids": [3, 8]
    }
  ],
  "avoid_patterns": [
    {
      "lesson": "Avoid unnormalized high-low amplitude when rank stability is weak.",
      "evidence_factor_ids": [3, 8]
    }
  ]
}
```
