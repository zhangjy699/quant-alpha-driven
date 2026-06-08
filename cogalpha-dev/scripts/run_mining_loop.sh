#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_mining_loop.sh [options] [-- run_formal_mvp.py args]

Options:
  --target-qualified-count N       Stop when factor_pool has this many elite+qualified factors. Default: 50
  --max-runs N                     Maximum formal runs to launch. Default: 100
  --sleep-seconds N                Pause between runs. Default: 0
  --summarizer-every N             Enable LLM summarizer every N post runs. Default: 5
  --run-prefix TEXT                Run ID prefix. Default: formal-mvp-loop
  --data-dir PATH                  Prepared data directory. Default: data/processed/company_all_a
  --run-output-root PATH           Formal run output root. Default: outputs/experiments
  --factor-pool PATH               Shared factor pool root. Default: outputs/factor_pool
  --memory-root PATH               Factor memory root. Default: outputs/factor_memory
  --loop-output-root PATH          Loop state output root. Default: outputs/loops
  --split NAME                     Formal mining split. Default: train
  --validation-split NAME          Validation audit split. Default: valid
  --agent-limit N                  Formal run agent limit. Default: 21
  --max-generations N              Formal run max generations. Default: 2
  --alphas-per-agent N             Formal run alphas per agent. Default: 1
  --parent-pool-size N             Formal run parent pool size. Default: 8
  --max-repair-attempts N          Formal run max repair attempts. Default: 2
  --python CMD                     Python command. Default: python
  --post-script PATH               Post-run script path. Default: scripts/post_formal_run.sh
  --backtest-script PATH           Backtest worker script path. Default: scripts/backtest_factor_pool_run.py
  --backtest-ingest-script PATH    Backtest audit ingest script. Default: scripts/ingest_backtest_audits.py
  --no-backtest                    Do not launch async backtests after post-run export.
  --backtest-pools LIST            Comma-separated pools. Default: elite,qualified
  --backtest-cost-bps N            Transaction cost in bps. Default: 10
  --backtest-output-root PATH      Backtest output root. Default: outputs/backtests
  --backtest-use-llm-summarizer    Use LLM summarizer when ingesting completed audits.
  --no-factor-memory               Do not inject factor_memory into formal run prompts.
  --inline-references              Inline skill references. Default: enabled
  --no-inline-references           Disable inlined skill references.
  --help, -h                       Show this help.
  --                               Pass remaining args to run_formal_mvp.py.
USAGE
}

TARGET_QUALIFIED_COUNT=3
MAX_RUNS=3
SLEEP_SECONDS=30
SUMMARIZER_EVERY=2
RUN_PREFIX="formal-mvp-loop"
DATA_DIR="data/processed/company_all_a"
RUN_OUTPUT_ROOT="outputs/experiments"
FACTOR_POOL="outputs/factor_pool"
MEMORY_ROOT="outputs/factor_memory"
LOOP_OUTPUT_ROOT="outputs/loops"
SPLIT="train"
VALIDATION_SPLIT="valid"
AGENT_LIMIT=21
MAX_GENERATIONS=2
ALPHAS_PER_AGENT=1
PARENT_POOL_SIZE=4
MAX_REPAIR_ATTEMPTS=1
PYTHON_CMD="${PYTHON:-python}"
POST_SCRIPT="scripts/post_formal_run.sh"
BACKTEST_SCRIPT="scripts/backtest_factor_pool_run.py"
BACKTEST_INGEST_SCRIPT="scripts/ingest_backtest_audits.py"
ENABLE_BACKTEST=1
BACKTEST_POOLS="elite,qualified"
BACKTEST_COST_BPS=10
BACKTEST_OUTPUT_ROOT="outputs/backtests"
BACKTEST_USE_LLM_SUMMARIZER=0
USE_FACTOR_MEMORY=1
INLINE_REFERENCES=1
HAVE_EXTRA_RUN_ARGS=0
EXTRA_RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-qualified-count)
      TARGET_QUALIFIED_COUNT="$2"
      shift 2
      ;;
    --max-runs)
      MAX_RUNS="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --summarizer-every)
      SUMMARIZER_EVERY="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --run-output-root)
      RUN_OUTPUT_ROOT="$2"
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
    --loop-output-root)
      LOOP_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --validation-split)
      VALIDATION_SPLIT="$2"
      shift 2
      ;;
    --agent-limit)
      AGENT_LIMIT="$2"
      shift 2
      ;;
    --max-generations)
      MAX_GENERATIONS="$2"
      shift 2
      ;;
    --alphas-per-agent)
      ALPHAS_PER_AGENT="$2"
      shift 2
      ;;
    --parent-pool-size)
      PARENT_POOL_SIZE="$2"
      shift 2
      ;;
    --max-repair-attempts)
      MAX_REPAIR_ATTEMPTS="$2"
      shift 2
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --post-script)
      POST_SCRIPT="$2"
      shift 2
      ;;
    --backtest-script)
      BACKTEST_SCRIPT="$2"
      shift 2
      ;;
    --backtest-ingest-script)
      BACKTEST_INGEST_SCRIPT="$2"
      shift 2
      ;;
    --no-backtest)
      ENABLE_BACKTEST=0
      shift
      ;;
    --backtest-pools)
      BACKTEST_POOLS="$2"
      shift 2
      ;;
    --backtest-cost-bps)
      BACKTEST_COST_BPS="$2"
      shift 2
      ;;
    --backtest-output-root)
      BACKTEST_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --backtest-use-llm-summarizer)
      BACKTEST_USE_LLM_SUMMARIZER=1
      shift
      ;;
    --no-factor-memory)
      USE_FACTOR_MEMORY=0
      shift
      ;;
    --inline-references)
      INLINE_REFERENCES=1
      shift
      ;;
    --no-inline-references)
      INLINE_REFERENCES=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        EXTRA_RUN_ARGS=("$@")
        HAVE_EXTRA_RUN_ARGS=1
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

LOOP_ID="mining-loop-$(date -u +%Y%m%d-%H%M%S)"
LOOP_DIR="$LOOP_OUTPUT_ROOT/$LOOP_ID"
RUNS_PATH="$LOOP_DIR/runs.jsonl"
STATE_PATH="$LOOP_DIR/loop_state.json"
LOOP_STARTED_EPOCH="$(date +%s)"
mkdir -p "$LOOP_DIR"

effective_factor_count() {
  "$PYTHON_CMD" - "$FACTOR_POOL" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1]) / "index.json"
if not index_path.exists():
    print(0)
    raise SystemExit
index = json.loads(index_path.read_text(encoding="utf-8"))
print(
    sum(
        1
        for entry in index.get("factors", [])
        if entry.get("pool") in {"elite", "qualified"}
    )
)
PY
}

write_loop_state() {
  local status="$1"
  local runs_completed="$2"
  local effective_count="$3"
  "$PYTHON_CMD" - "$STATE_PATH" <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

path = Path("$STATE_PATH")
payload = {
    "loop_id": "$LOOP_ID",
    "status": "$status",
    "updated_at": datetime.now(UTC).isoformat(),
    "runs_completed": int("$runs_completed"),
    "target_qualified_count": int("$TARGET_QUALIFIED_COUNT"),
    "effective_factor_count": int("$effective_count"),
    "max_runs": int("$MAX_RUNS"),
    "factor_pool": "$FACTOR_POOL",
    "memory_root": "$MEMORY_ROOT",
    "runs_path": "$RUNS_PATH",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

append_run_record() {
  local payload="$1"
  printf '%s\n' "$payload" >> "$RUNS_PATH"
}

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

ingest_completed_backtests() {
  if [[ "$ENABLE_BACKTEST" -ne 1 ]]; then
    return 0
  fi

  local ingest_args=(
    --backtest-root "$BACKTEST_OUTPUT_ROOT"
    --memory-root "$MEMORY_ROOT"
  )
  if [[ "$BACKTEST_USE_LLM_SUMMARIZER" -eq 1 ]]; then
    ingest_args+=(--use-llm-summarizer)
  fi
  if [[ "$INLINE_REFERENCES" -eq 1 ]]; then
    ingest_args+=(--inline-references)
  fi

  "$PYTHON_CMD" "$BACKTEST_INGEST_SCRIPT" "${ingest_args[@]}"
}

launch_backtest_worker() {
  local run_id="$1"
  local run_dir="$2"
  if [[ "$ENABLE_BACKTEST" -ne 1 ]]; then
    return 0
  fi

  local log_path="$run_dir/backtest_worker.log"
  local pools_args=()
  local old_ifs="$IFS"
  IFS=","
  read -r -a pools_args <<< "$BACKTEST_POOLS"
  IFS="$old_ifs"

  local backtest_args=(
    --run-id "$run_id"
    --data-dir "$DATA_DIR"
    --factor-pool "$FACTOR_POOL"
    --output-dir "$BACKTEST_OUTPUT_ROOT"
    --cost-bps "$BACKTEST_COST_BPS"
    --pools "${pools_args[@]}"
  )

  (
    "$PYTHON_CMD" "$BACKTEST_SCRIPT" "${backtest_args[@]}"
  ) > "$log_path" 2>&1 &
  echo $! > "$run_dir/backtest_worker.pid"
  echo "Started async backtest worker pid=$(cat "$run_dir/backtest_worker.pid") "\
"log=$log_path pools=$BACKTEST_POOLS"
}

runs_completed=0
current_count="$(effective_factor_count)"
write_loop_state "running" "$runs_completed" "$current_count"

while [[ "$runs_completed" -lt "$MAX_RUNS" ]]; do
  ingest_completed_backtests

  current_count="$(effective_factor_count)"
  if [[ "$current_count" -ge "$TARGET_QUALIFIED_COUNT" ]]; then
    write_loop_state "target_reached" "$runs_completed" "$current_count"
    total_elapsed_seconds=$(($(date +%s) - LOOP_STARTED_EPOCH))
    echo "Target reached: effective factor count $current_count >= "\
"$TARGET_QUALIFIED_COUNT total_time=$(format_duration "$total_elapsed_seconds")"
    exit 0
  fi

  run_number=$((runs_completed + 1))
  run_id="${RUN_PREFIX}-$(date -u +%Y%m%d-%H%M%S)-r${run_number}"
  run_dir="$RUN_OUTPUT_ROOT/$run_id"
  before_count="$current_count"
  run_started_epoch="$(date +%s)"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  run_args=(
    --run-id "$run_id"
    --data-dir "$DATA_DIR"
    --split "$SPLIT"
    --output-root "$RUN_OUTPUT_ROOT"
    --agent-limit "$AGENT_LIMIT"
    --max-generations "$MAX_GENERATIONS"
    --alphas-per-agent "$ALPHAS_PER_AGENT"
    --parent-pool-size "$PARENT_POOL_SIZE"
    --max-repair-attempts "$MAX_REPAIR_ATTEMPTS"
    --factor-memory-root "$MEMORY_ROOT"
  )
  if [[ "$USE_FACTOR_MEMORY" -eq 1 ]]; then
    run_args+=(--use-factor-memory)
  fi
  if [[ "$INLINE_REFERENCES" -eq 1 ]]; then
    run_args+=(--inline-references)
  fi
  if [[ "$HAVE_EXTRA_RUN_ARGS" -eq 1 ]]; then
    run_args+=("${EXTRA_RUN_ARGS[@]}")
  fi

  post_args=(
    --run-dir "$run_dir"
    --data-dir "$DATA_DIR"
    --split "$VALIDATION_SPLIT"
    --factor-pool "$FACTOR_POOL"
    --memory-root "$MEMORY_ROOT"
    --python "$PYTHON_CMD"
  )
  use_summarizer=0
  if [[ "$SUMMARIZER_EVERY" -gt 0 && $((run_number % SUMMARIZER_EVERY)) -eq 0 ]]; then
    post_args+=(--use-llm-summarizer)
    use_summarizer=1
  fi
  if [[ "$INLINE_REFERENCES" -eq 1 ]]; then
    post_args+=(--inline-references)
  fi

  echo "Starting run $run_number/$MAX_RUNS: $run_id"
  status="ok"
  backtest_worker_pid=""
  backtest_worker_log=""
  if ! "$PYTHON_CMD" scripts/run_formal_mvp.py "${run_args[@]}"; then
    status="run_failed"
  elif ! "$POST_SCRIPT" "${post_args[@]}"; then
    status="post_failed"
  fi
  if [[ "$status" == "ok" ]]; then
    launch_backtest_worker "$run_id" "$run_dir"
    if [[ -f "$run_dir/backtest_worker.pid" ]]; then
      backtest_worker_pid="$(cat "$run_dir/backtest_worker.pid")"
      backtest_worker_log="$run_dir/backtest_worker.log"
    fi
  fi

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_finished_epoch="$(date +%s)"
  run_elapsed_seconds=$((run_finished_epoch - run_started_epoch))
  total_elapsed_seconds=$((run_finished_epoch - LOOP_STARTED_EPOCH))
  after_count="$(effective_factor_count)"
  runs_completed="$run_number"
  write_loop_state "$status" "$runs_completed" "$after_count"
  append_run_record "$(
    "$PYTHON_CMD" - <<PY
import json
payload = {
    "run_number": $run_number,
    "run_id": "$run_id",
    "run_dir": "$run_dir",
    "status": "$status",
    "started_at": "$started_at",
    "finished_at": "$finished_at",
    "run_elapsed_seconds": int("$run_elapsed_seconds"),
    "total_elapsed_seconds": int("$total_elapsed_seconds"),
    "effective_factor_count_before": int("$before_count"),
    "effective_factor_count_after": int("$after_count"),
    "used_llm_summarizer": bool($use_summarizer),
    "backtest_enabled": bool($ENABLE_BACKTEST),
    "backtest_worker_pid": "$backtest_worker_pid" or None,
    "backtest_worker_log": "$backtest_worker_log" or None,
    "backtest_pools": "$BACKTEST_POOLS",
}
print(json.dumps(payload, sort_keys=True))
PY
  )"

  echo "Finished run $run_number/$MAX_RUNS: $run_id status=$status "\
"run_time=$(format_duration "$run_elapsed_seconds") "\
"total_time=$(format_duration "$total_elapsed_seconds") "\
"effective_factors=$after_count/$TARGET_QUALIFIED_COUNT"

  if [[ "$status" != "ok" ]]; then
    echo "Stopping loop because $status for $run_id" >&2
    exit 1
  fi

  if [[ "$SLEEP_SECONDS" -gt 0 && "$runs_completed" -lt "$MAX_RUNS" ]]; then
    sleep "$SLEEP_SECONDS"
  fi
done

ingest_completed_backtests
current_count="$(effective_factor_count)"
write_loop_state "max_runs_reached" "$runs_completed" "$current_count"
total_elapsed_seconds=$(($(date +%s) - LOOP_STARTED_EPOCH))
echo "Max runs reached: $runs_completed; effective factor count: $current_count "\
"total_time=$(format_duration "$total_elapsed_seconds")"
