#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

data_dir="data/processed/company_all_a"
output_dir=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --data-dir)
      data_dir="${args[$((i + 1))]}"
      ;;
    --output-dir)
      output_dir="${args[$((i + 1))]}"
      ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  output_dir="outputs/backtests/full-factor-pool"
  if [[ "$data_dir" == *"exclude_st"* ]]; then
    output_dir="outputs/backtests/full-factor-pool_exclude_st"
  fi
fi

python scripts/backtest_factor_pool_full.py "$@"
python factor_backtest/compact_output.py --input-dir "$output_dir"
