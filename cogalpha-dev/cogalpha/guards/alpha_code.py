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

    alpha_function = functions[0] if len(functions) == 1 else None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, issues)
        elif isinstance(node, ast.Call):
            _check_call(node, issues)

    if alpha_function is not None:
        _check_column_accesses(alpha_function, issues)

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


def _check_column_accesses(function: ast.FunctionDef, issues: list[GuardIssue]) -> None:
    assigned_df_copy_columns: set[str] = set()
    for statement in function.body:
        _check_statement_column_reads(statement, assigned_df_copy_columns, issues)
        _record_statement_df_copy_assignments(statement, assigned_df_copy_columns, issues)


def _check_statement_column_reads(
    statement: ast.stmt,
    assigned_df_copy_columns: set[str],
    issues: list[GuardIssue],
) -> None:
    if isinstance(statement, ast.Assign):
        for value in [statement.value]:
            _check_expression_column_reads(value, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.AnnAssign):
        if statement.value is not None:
            _check_expression_column_reads(statement.value, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.AugAssign):
        _check_expression_column_reads(statement.target, assigned_df_copy_columns, issues)
        _check_expression_column_reads(statement.value, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.Return):
        if statement.value is not None:
            _check_expression_column_reads(statement.value, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.Expr):
        _check_expression_column_reads(statement.value, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.For):
        _check_expression_column_reads(statement.iter, assigned_df_copy_columns, issues)
        for child_statement in statement.body + statement.orelse:
            _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.If):
        _check_expression_column_reads(statement.test, assigned_df_copy_columns, issues)
        for child_statement in statement.body + statement.orelse:
            _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.While):
        _check_expression_column_reads(statement.test, assigned_df_copy_columns, issues)
        for child_statement in statement.body + statement.orelse:
            _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.Try):
        for child_statement in statement.body + statement.orelse + statement.finalbody:
            _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)
        for handler in statement.handlers:
            for child_statement in handler.body:
                _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)
        return
    if isinstance(statement, ast.With):
        for item in statement.items:
            _check_expression_column_reads(
                item.context_expr,
                assigned_df_copy_columns,
                issues,
            )
        for child_statement in statement.body:
            _check_statement_column_reads(child_statement, assigned_df_copy_columns, issues)


def _check_expression_column_reads(
    expression: ast.AST,
    assigned_df_copy_columns: set[str],
    issues: list[GuardIssue],
) -> None:
    if isinstance(expression, ast.Subscript) and _is_dataframe_column_access(expression):
        _check_column_read(expression, assigned_df_copy_columns, issues)
    for child in ast.iter_child_nodes(expression):
        _check_expression_column_reads(child, assigned_df_copy_columns, issues)


def _check_column_read(
    node: ast.Subscript,
    assigned_df_copy_columns: set[str],
    issues: list[GuardIssue],
) -> None:
    dataframe_name = _dataframe_name(node)
    column = _string_subscript_key(node)
    if dataframe_name is None or column is None:
        return
    if dataframe_name == "df" and not _is_known_dataframe_column(column):
        _append_unknown_input_column_issue(column, node, issues)
        return
    if dataframe_name == "df_copy" and not (
        _is_known_dataframe_column(column) or column in assigned_df_copy_columns
    ):
        _append_unknown_input_column_issue(column, node, issues)


def _record_statement_df_copy_assignments(
    statement: ast.stmt,
    assigned_df_copy_columns: set[str],
    issues: list[GuardIssue],
) -> None:
    for target in _assignment_targets(statement):
        for subscript in _subscript_targets(target):
            dataframe_name = _dataframe_name(subscript)
            column = _string_subscript_key(subscript)
            if dataframe_name is None or column is None:
                continue
            if dataframe_name == "df":
                if not _is_known_dataframe_column(column):
                    _append_unknown_input_column_issue(column, subscript, issues)
            elif dataframe_name == "df_copy":
                assigned_df_copy_columns.add(column)


def _assignment_targets(statement: ast.stmt) -> list[ast.AST]:
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        if isinstance(statement, ast.Assign):
            return list(statement.targets)
        return [statement.target]
    if isinstance(statement, ast.For):
        targets = [statement.target]
        for child in statement.body + statement.orelse:
            targets.extend(_assignment_targets(child))
        return targets
    if isinstance(statement, (ast.With, ast.If, ast.While, ast.Try)):
        targets: list[ast.AST] = []
        child_statements: list[ast.stmt] = []
        if isinstance(statement, ast.With):
            child_statements = statement.body
        elif isinstance(statement, (ast.If, ast.While)):
            child_statements = statement.body + statement.orelse
        elif isinstance(statement, ast.Try):
            child_statements = statement.body + statement.orelse + statement.finalbody
            for handler in statement.handlers:
                child_statements.extend(handler.body)
        for child in child_statements:
            targets.extend(_assignment_targets(child))
        return targets
    return []


def _subscript_targets(node: ast.AST) -> list[ast.Subscript]:
    targets: list[ast.Subscript] = []
    if isinstance(node, ast.Subscript):
        targets.append(node)
    for child in ast.iter_child_nodes(node):
        targets.extend(_subscript_targets(child))
    return targets


def _append_unknown_input_column_issue(
    column: str,
    node: ast.AST,
    issues: list[GuardIssue],
) -> None:
    issues.append(
        GuardIssue(
            code="unknown_input_column",
            message=f"Column {column!r} is outside the OHLCV Input contract.",
            location=f"line {node.lineno}",
        )
    )


def _is_dataframe_column_access(node: ast.Subscript) -> bool:
    return _dataframe_name(node) is not None and _string_subscript_key(node) is not None


def _dataframe_name(node: ast.Subscript) -> str | None:
    value = node.value
    if isinstance(value, ast.Name) and value.id in {"df", "df_copy"}:
        return value.id
    return None


def _string_subscript_key(node: ast.Subscript) -> str | None:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


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
