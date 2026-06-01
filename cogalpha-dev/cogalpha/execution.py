"""Restricted execution for generated alpha functions over OHLCV panels."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from cogalpha.alpha_contract import ALLOWED_IMPORT_MODULES, DEFAULT_OHLCV_COLUMNS
from cogalpha.schemas import AlphaCandidate, AlphaFunction

try:  # pragma: no cover - availability depends on the local TA-Lib install
    import talib
except ModuleNotFoundError:  # pragma: no cover
    talib = None

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class AlphaExecutionError(RuntimeError):
    """Raised when an Alpha Function cannot be executed under the runtime contract."""


def execute_alpha_candidate(
    candidate: AlphaCandidate,
    ohlcv_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Execute one AlphaCandidate against a two-level `(date, asset)` OHLCV panel."""

    return execute_alpha_function(candidate.alpha, ohlcv_panel)


def execute_alpha_function(
    alpha: AlphaFunction,
    ohlcv_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Return factor values as a date-by-asset DataFrame."""

    function = compile_alpha_function(alpha)
    outputs: dict[object, pd.Series] = {}
    for asset, frame in _iter_asset_frames(ohlcv_panel):
        outputs[asset] = _execute_one_asset(function, alpha.name, frame)
    if not outputs:
        raise AlphaExecutionError("OHLCV panel contains no assets.")
    return pd.DataFrame(outputs).sort_index().sort_index(axis=1)


def compile_alpha_function(alpha: AlphaFunction) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile one generated Alpha Function in a restricted namespace."""

    namespace = _runtime_namespace()
    try:
        exec(compile(alpha.code, f"<{alpha.name}>", "exec"), namespace, namespace)
    except Exception as exc:  # noqa: BLE001 - preserve runtime failure as guard context
        raise AlphaExecutionError(f"Failed to compile alpha code: {exc}") from exc

    function = namespace.get(alpha.name)
    if not callable(function):
        raise AlphaExecutionError(f"Alpha code did not define callable {alpha.name!r}.")
    return function


def _iter_asset_frames(ohlcv_panel: pd.DataFrame):
    if not isinstance(ohlcv_panel, pd.DataFrame):
        raise AlphaExecutionError("OHLCV panel must be a pandas DataFrame.")
    if not isinstance(ohlcv_panel.index, pd.MultiIndex) or ohlcv_panel.index.nlevels != 2:
        raise AlphaExecutionError("OHLCV panel must use a two-level MultiIndex: date, asset.")

    missing = [column for column in DEFAULT_OHLCV_COLUMNS if column not in ohlcv_panel.columns]
    if missing:
        raise AlphaExecutionError(f"OHLCV panel is missing required columns: {missing}.")

    sorted_panel = ohlcv_panel.sort_index()
    asset_level = sorted_panel.index.names[-1] or 1
    for asset, frame in sorted_panel.groupby(level=asset_level, sort=True):
        asset_frame = frame.droplevel(asset_level).loc[:, DEFAULT_OHLCV_COLUMNS].copy()
        yield asset, asset_frame


def _execute_one_asset(
    function: Callable[[pd.DataFrame], pd.Series],
    function_name: str,
    frame: pd.DataFrame,
) -> pd.Series:
    try:
        output = function(frame.copy())
    except Exception as exc:  # noqa: BLE001 - preserve runtime failure as guard context
        raise AlphaExecutionError(f"Alpha function raised during execution: {exc}") from exc

    if not isinstance(output, pd.Series):
        raise AlphaExecutionError("Alpha function must return a pandas Series.")
    if not output.index.equals(frame.index):
        raise AlphaExecutionError("Alpha output index must match the input asset frame index.")
    if output.name != function_name:
        raise AlphaExecutionError("Alpha output Series name must match the function name.")
    return output


def _runtime_namespace() -> dict[str, object]:
    builtins = dict(SAFE_BUILTINS)
    builtins["__import__"] = _restricted_import
    return {
        "__builtins__": builtins,
        "math": math,
        "np": np,
        "pd": pd,
        "stats": stats,
        "talib": talib,
    }


def _restricted_import(
    name: str,
    globals=None,  # noqa: ANN001 - matches Python's __import__ protocol
    locals=None,  # noqa: ANN001 - matches Python's __import__ protocol
    fromlist=(),  # noqa: ANN001 - matches Python's __import__ protocol
    level: int = 0,
):
    if level != 0:
        raise ImportError("Relative imports are not allowed in Alpha Functions.")
    if name == "scipy" and set(fromlist).issubset({"stats"}):
        return importlib.import_module(name)
    if name not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"Import {name!r} is outside the alpha runtime allowlist.")
    if name == "talib" and talib is None:
        raise ImportError("TA-Lib is not available in this runtime.")
    return importlib.import_module(name)
