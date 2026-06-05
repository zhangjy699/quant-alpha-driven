# Runtime Harness Manifest

Included:

- `cogalpha/`: runtime package for the Agent Loop Module, Tool Dispatch Module,
  Skill-Driven MVP compatibility runner, guards, scoring, and reporting.
- `cogalpha/harness/`: agent loop, tool dispatch, model-directed agentic
  controller validation, CogAlpha tool adapters, and the invoker-backed MVP
  compatibility runner.
- `cogalpha/tracing.py`: append-only JSONL trace event schemas and IO.
- `cogalpha/verification/trace_verifier.py`: semantic replay verifier for
  agentic trace artifacts and final state links.
- `skills/`: Standard Skills needed by the MVP graph.
- `configs/baseline.yaml` and `configs/mvp.yaml`: paper-aligned runtime defaults.
- `scripts/prepare_hf_qlib_csi300.py`: data preparation entrypoint.
- `scripts/run_agentic_mvp.py`: intended paper-aligned agentic workflow
  entrypoint; writes `trace.jsonl`, `trace_manifest.json`, and
  `trace_verification.json` in addition to formal run artifacts.
- `scripts/run_formal_mvp.py`: deterministic compatibility/formal MVP workflow
  entrypoint.
- `pyproject.toml` and `uv.lock`: reproducible dependency surface.

Excluded:

- workspace logs and experiment history,
- generated candidates,
- data files,
- outputs,
- tests and temporary benchmark scripts,
- secrets such as `KEY.md`,
- protected evaluator assets.
