"""Static deterministic guards for generated alpha function code."""

from __future__ import annotations

import ast
import re

from cogalpha.alpha_contract import (
    ALLOWED_IMPORT_MODULES,
    FORBIDDEN_ALPHA_CALLS,
    FORBIDDEN_TIME_ORDER_PATTERNS,
    is_ohlcv_or_factor_column,
)
from cogalpha.schemas import GuardIssue, GuardReport, GuardStatus


def run_static_alpha_code_guard(code: str, function_name: str | None = None) -> GuardReport:
    """Run syntax, import, leakage, and shape checks that do not execute code."""

    issues: list[GuardIssue] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return GuardReport(
            guard_name="static_alpha_code",
            status=GuardStatus.FAIL,
            issues=[
                GuardIssue(
                    code="syntax_error",
                    message=str(exc),
                    location=f"line {exc.lineno}" if exc.lineno else None,
                )
            ],
        )

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        issues.append(
            GuardIssue(
                code="function_count",
                message="Alpha code must define exactly one top-level function.",
            )
        )
    elif function_name and functions[0].name != function_name:
        issues.append(
            GuardIssue(
                code="function_name_mismatch",
                message=f"Expected function {function_name!r}, found {functions[0].name!r}.",
                location=f"line {functions[0].lineno}",
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, issues)
        elif isinstance(node, ast.Call):
            _check_call(node, issues)
        elif isinstance(node, ast.Subscript):
            _check_subscript(node, issues)

    text = re.sub(r"\s+", "", code)
    for pattern in FORBIDDEN_TIME_ORDER_PATTERNS:
        if pattern in text:
            issues.append(
                GuardIssue(
                    code="possible_reverse_time_order",
                    message=f"Forbidden time-order pattern detected: {pattern}",
                )
            )

    status = (
        GuardStatus.FAIL
        if any(issue.severity == "error" for issue in issues)
        else GuardStatus.PASS
    )
    return GuardReport(guard_name="static_alpha_code", status=status, issues=issues)


def _check_import(node: ast.Import | ast.ImportFrom, issues: list[GuardIssue]) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_MODULES:
                issues.append(
                    GuardIssue(
                        code="forbidden_import",
                        message=f"Import {alias.name!r} is outside the alpha runtime allowlist.",
                        location=f"line {node.lineno}",
                    )
                )
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        root = module.split(".")[0]
        if root not in ALLOWED_IMPORT_MODULES:
            issues.append(
                GuardIssue(
                    code="forbidden_import",
                    message=f"Import from {module!r} is outside the alpha runtime allowlist.",
                    location=f"line {node.lineno}",
                )
            )


def _check_call(node: ast.Call, issues: list[GuardIssue]) -> None:
    name = _call_name(node.func)
    if name in FORBIDDEN_ALPHA_CALLS:
        issues.append(
            GuardIssue(
                code="forbidden_call",
                message=f"Forbidden call {name!r} detected.",
                location=f"line {node.lineno}",
            )
        )

    if name.endswith(".shift") and node.args:
        first = node.args[0]
        if _is_negative_number(first):
            issues.append(
                GuardIssue(
                    code="future_shift",
                    message="Negative shift uses future information and is forbidden.",
                    location=f"line {node.lineno}",
                )
            )

    if name.endswith(".rolling"):
        for keyword in node.keywords:
            if keyword.arg == "center" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    issues.append(
                        GuardIssue(
                            code="centered_rolling_window",
                            message=(
                                "Centered rolling windows use future observations "
                                "and are forbidden."
                            ),
                            location=f"line {node.lineno}",
                        )
                    )

    if _is_rolling_apply_call(node):
        issues.append(
            GuardIssue(
                code="expensive_rolling_apply",
                message=(
                    "Custom rolling.apply functions are forbidden because they can stall "
                    "runtime validation on large market panels; use vectorized rolling "
                    "aggregations instead."
                ),
                location=f"line {node.lineno}",
            )
        )


def _check_subscript(node: ast.Subscript, issues: list[GuardIssue]) -> None:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        column = node.slice.value
        if _looks_like_ohlcv_access(node) and not _is_known_dataframe_column(column):
            issues.append(
                GuardIssue(
                    code="unknown_input_column",
                    message=f"Column {column!r} is outside the OHLCV Input contract.",
                    location=f"line {node.lineno}",
                )
            )


def _looks_like_ohlcv_access(node: ast.Subscript) -> bool:
    value = node.value
    return isinstance(value, ast.Name) and value.id in {"df", "df_copy"}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    if isinstance(func, ast.Subscript):
        return _call_name(func.value)
    return ""


def _is_rolling_apply_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "apply":
        return False
    rolling_call = node.func.value
    if not isinstance(rolling_call, ast.Call):
        return False
    return _call_name(rolling_call.func).endswith(".rolling")


def _is_known_dataframe_column(column: str) -> bool:
    return is_ohlcv_or_factor_column(column)


def _is_negative_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    return False
