# Agentic Workflow

CogAlpha Standard Skills are executable workflow documents. They must help an
agent produce schema-valid artifacts while leaving enough trace evidence for
later verification.

## Observe

- Read the runtime payload before proposing any factor, repair, judgment, or
  evolution step.
- Identify the active skill name, candidate ids, generation, guidance mode,
  parent ids, and prior feedback summaries.
- Use only payload fields and allowed OHLCV columns. Do not infer hidden labels,
  future returns, sealed test data, or unavailable workspace evidence.

## Plan

- State the market mechanism or quality objective internally before generating
  the artifact.
- Choose one compact action: generate, accept, repair, reject, mutate, or
  crossover.
- For domain skills, align the plan to the named paper agent and seven-level
  CogAlpha hierarchy layer.
- For quality skills, decide whether the candidate is accepted, repairable, or
  rejected.
- For evolution skills, identify the parent strength to preserve and the metric
  bottleneck to target.

## Generate

- Return only the schema requested by the runtime.
- Keep factor code vectorized and compatible with the Alpha Factor Contract.
- Keep rationale tied to an economic mechanism rather than decorative feature
  stacking.

## Self-Check

- Check no future information leakage, negative shifts, centered windows, target
  returns, imports, file IO, network IO, subprocesses, or mutable global state.
- Check every output id, lineage field, and verdict matches the runtime schema.
- Check the output can be traced back to the observed payload.

## Trace Expectations

Skills should expose or preserve these trace fields when the runtime records
their invocation:

- `skill_name`: registered CogAlpha skill name.
- `candidate_id`: generated, reviewed, repaired, or evolved candidate id.
- `parent_ids`: parent candidates for mutation or crossover.
- `generation`: current generation number.
- `guidance_mode`: diversification, mutation, crossover, or repair mode.
- `action`: generate, accept, repair, reject, mutate, or crossover.
- `evidence_id`: trace or governance record supporting later utility updates.
- `stage`: candidate lifecycle stage after the skill action.
