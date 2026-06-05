# Factor Memory

Factor memory is a compressed experience layer for future alpha generation.
It is not a raw factor store and it is not a skill editing mechanism.

## Source of Truth

- `factor_pool` is the factual store for factor code, formulas, metrics, and run
  attribution.
- `factor_memory` stores only concise lessons, weak hypotheses, counters, and
  `factor_id` references.
- Do not copy factor code into memory lessons.
- Do not invent factor IDs, metrics, formulas, outcomes, splits, or domain
  agents.

## Evidence Bounds

- Use only the bounded runtime payload provided to the summarizer.
- Do not read or assume the full historical factor pool.
- Preserve evidence factor IDs for every lesson and hypothesis.
- If evidence is sparse or contradictory, keep the lesson conservative.

## Validation Feedback

- Train and validation outcomes may update generation memory.
- Test outcomes must not update generation memory.
- Validation failures are generalization feedback, not threshold-tuning
  instructions.
- Do not recommend lowering gates, changing splits, or tuning parameters to make
  a factor pass validation.

## Regime Hypotheses

- Regime hypotheses are weak research hypotheses, not market facts.
- Every hypothesis must include evidence factor IDs, confidence, and risk.
- Confidence must remain `low` or `medium`.
- Phrase hypotheses as inspiration for diverse search, not as commands to
  conform to a market style.

## Retrieval Contract

- Retrieval injects only a small top-k subset into future prompts.
- Prefer mechanism-level lessons over factor-specific anecdotes.
- Keep output short enough to remain useful in repeated 24h mining loops.
