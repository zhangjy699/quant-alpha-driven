"""Shared Alpha Function and OHLCV Input contract."""

from __future__ import annotations

DEFAULT_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
DEFAULT_ALPHA_LIBRARY_ALIASES: tuple[str, ...] = ("np", "pd", "stats", "talib", "math")

ALLOWED_IMPORT_MODULES: frozenset[str] = frozenset(
    {"math", "numpy", "pandas", "scipy", "scipy.stats", "talib"}
)
ALLOWED_ALPHA_ALIASES: frozenset[str] = frozenset(DEFAULT_ALPHA_LIBRARY_ALIASES)

FORBIDDEN_ALPHA_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "open", "__import__"}
)
FORBIDDEN_TIME_ORDER_PATTERNS: frozenset[str] = frozenset(
    {"iloc[::-1]", "sort_index(ascending=False)"}
)


def is_ohlcv_or_factor_column(column: str) -> bool:
    """Return whether a generated Alpha Function may read this DataFrame column."""

    return column in DEFAULT_OHLCV_COLUMNS or column.startswith("factor_")
