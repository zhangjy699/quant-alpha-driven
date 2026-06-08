"""Independent factor backtest platform for CogAlpha factor_pool artifacts."""

from factor_backtest.engine import FactorBacktestResult, run_factor_backtest
from factor_backtest.loader import FactorBacktestInput, load_factor_from_pool

__all__ = [
    "FactorBacktestInput",
    "FactorBacktestResult",
    "load_factor_from_pool",
    "run_factor_backtest",
]
