# CogAlpha Runtime Harness

This is the clean runtime harness extracted from the CogAlpha evolution
workspace. It contains only the code, skills, configs, and entrypoints needed to
run the current MVP workflow.

## Contents

```text
harness_system/
├── cogalpha/
├── configs/
│   ├── baseline.yaml
│   └── mvp.yaml
├── scripts/
│   ├── prepare_hf_qlib_csi300.py
│   └── run_formal_mvp.py
├── skills/
├── pyproject.toml
└── uv.lock
```

## Run

Prepare data:

```bash
uv run python scripts/prepare_hf_qlib_csi300.py
```

Run the formal MVP workflow with DeepSeek:

```bash
uv run python scripts/run_formal_mvp.py \
  --split valid \
  --agent-limit 3 \
  --max-generations 1 \
  --alphas-per-agent 1 \
  --parent-pool-size 4 \
  --max-repair-attempts 0 \
  --inline-references \
  --provider deepseek
```

Provide credentials via environment variables or a local `KEY.md`; never commit
`KEY.md`.
