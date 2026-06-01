# Structured Artifacts

CogAlpha skills exchange schema-validated artifacts rather than free-form completions.

## AlphaCandidate

Evolution skills return a JSON object compatible with `AlphaCandidate`:

```json
{
  "candidate_id": "string",
  "alpha": {
    "name": "factor_descriptive_name",
    "code": "def factor_descriptive_name(df): ...",
    "formula": "optional formula",
    "rationale": "financial intuition and implementation summary",
    "required_columns": ["open", "high", "low", "close", "volume"],
    "allowed_libraries": ["np", "pd", "stats", "talib", "math"]
  },
  "stage": "generated",
  "lineage": {
    "operation": null,
    "parent_ids": [],
    "generation": 0,
    "agent_skill": "skill-name",
    "guidance_mode": "optional"
  },
  "metadata": {}
}
```

## AlphaCandidateBatch

Domain Agent Skills return a JSON object compatible with `AlphaCandidateBatch`:

```json
{
  "candidates": [
    {
      "candidate_id": "string",
      "alpha": {
        "name": "factor_descriptive_name",
        "code": "def factor_descriptive_name(df): ...",
        "formula": "optional formula",
        "rationale": "financial intuition and implementation summary",
        "required_columns": ["open", "high", "low", "close", "volume"],
        "allowed_libraries": ["np", "pd", "stats", "talib", "math"]
      },
      "stage": "generated",
      "lineage": {
        "operation": null,
        "parent_ids": [],
        "generation": 0,
        "agent_skill": "skill-name",
        "guidance_mode": "optional"
      },
      "metadata": {}
    }
  ]
}
```

## QualityDecision

Quality checker skills return a JSON object compatible with `QualityDecision`:

```json
{
  "skill": {
    "name": "alpha-judge",
    "path": "skills/alpha-judge/SKILL.md",
    "kind": "quality_checker"
  },
  "verdict": "accept",
  "practical_soundness": "concise assessment",
  "feedback": "specific repair or rejection guidance",
  "repaired_candidate": null
}
```

## Output Rules

- Return JSON only.
- Do not wrap output in markdown fences.
- Do not include extra commentary outside the JSON object.
- Keep code as a JSON string value inside `alpha.code`.
