# Runtime Harness Manifest

Included:

- `cogalpha/`: runtime package for Skill-Driven DAG execution.
- `skills/`: Standard Skills needed by the MVP graph.
- `configs/baseline.yaml` and `configs/mvp.yaml`: paper-aligned runtime defaults.
- `scripts/prepare_hf_qlib_csi300.py`: data preparation entrypoint.
- `scripts/run_formal_mvp.py`: formal MVP workflow entrypoint.
- `pyproject.toml` and `uv.lock`: reproducible dependency surface.

Excluded:

- workspace logs and experiment history,
- generated candidates,
- data files,
- outputs,
- tests and temporary benchmark scripts,
- secrets such as `KEY.md`,
- protected evaluator assets.
