#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/backtest_factor_pool_full.py "$@"
python factor_backtest/compact_output.py
