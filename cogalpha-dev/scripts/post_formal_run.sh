#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/post_formal_run.sh --run-dir outputs/experiments/<formal-run-id> [options]

Options:
  --run-dir PATH             Formal MVP run directory containing final_state.json.
  --data-dir PATH            Prepared data directory. Default: data/processed/csi300
  --split NAME               Validation split. Default: valid
  --factor-pool PATH         Shared factor pool root. Default: outputs/factor_pool
  --memory-root PATH         Factor memory root. Default: outputs/factor_memory
  --rejected-limit N         Worst rejected_by_fitness exports. Default: 3
  --use-llm-summarizer       Enable bounded LLM memory summarization.
  --inline-references        Inline skill references for summarizer calls.
  --python CMD               Python command. Default: python
  --                         Pass remaining arguments to validate_qualified_factors.py.
USAGE
}

RUN_DIR=""
DATA_DIR="data/processed/csi300"
SPLIT="valid"
FACTOR_POOL="outputs/factor_pool"
MEMORY_ROOT="outputs/factor_memory"
REJECTED_LIMIT="3"
PYTHON_CMD="${PYTHON:-python}"
USE_LLM_SUMMARIZER=0
INLINE_REFERENCES=0
HAVE_EXTRA_VALIDATE_ARGS=0
EXTRA_VALIDATE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --factor-pool)
      FACTOR_POOL="$2"
      shift 2
      ;;
    --memory-root)
      MEMORY_ROOT="$2"
      shift 2
      ;;
    --rejected-limit)
      REJECTED_LIMIT="$2"
      shift 2
      ;;
    --use-llm-summarizer)
      USE_LLM_SUMMARIZER=1
      shift
      ;;
    --inline-references)
      INLINE_REFERENCES=1
      shift
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        EXTRA_VALIDATE_ARGS=("$@")
        HAVE_EXTRA_VALIDATE_ARGS=1
      fi
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$RUN_DIR/final_state.json" ]]; then
  echo "Missing final_state.json under run dir: $RUN_DIR" >&2
  exit 1
fi

VALIDATE_ARGS=(
  --run-dir "$RUN_DIR"
  --data-dir "$DATA_DIR"
  --split "$SPLIT"
  --factor-pool "$FACTOR_POOL"
  --memory-root "$MEMORY_ROOT"
)
if [[ "$USE_LLM_SUMMARIZER" -eq 1 ]]; then
  VALIDATE_ARGS+=(--use-llm-summarizer)
fi
if [[ "$INLINE_REFERENCES" -eq 1 ]]; then
  VALIDATE_ARGS+=(--inline-references)
fi
if [[ "$HAVE_EXTRA_VALIDATE_ARGS" -eq 1 ]]; then
  VALIDATE_ARGS+=("${EXTRA_VALIDATE_ARGS[@]}")
fi

"$PYTHON_CMD" scripts/export_factor_pool.py \
  --run-dir "$RUN_DIR" \
  --output-root "$FACTOR_POOL" \
  --rejected-limit "$REJECTED_LIMIT"

"$PYTHON_CMD" scripts/validate_qualified_factors.py "${VALIDATE_ARGS[@]}"

"$PYTHON_CMD" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
state = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
quality_rejected = sum(
    1
    for candidate in state.get("rejected_pool", [])
    if candidate.get("stage") == "rejected_by_quality"
)
if (
    not state.get("elite_pool")
    and not state.get("qualified_pool")
    and quality_rejected
):
    print(
        "Note: quality-rejected candidates are not exported to factor_pool "
        "or validated because they have no five-metric fitness scores."
    )
PY
