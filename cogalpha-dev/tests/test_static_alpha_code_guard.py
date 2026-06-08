from cogalpha.guards import run_static_alpha_code_guard
from cogalpha.schemas import GuardStatus


def test_static_guard_accepts_simple_causal_factor():
    code = '''
def factor_close_open_gap(df):
    """Close-open gap normalized by open."""
    df_copy = df.copy()
    eps = 1e-9
    gap = df_copy["close"] - df_copy["open"]
    df_copy["factor_close_open_gap"] = gap / (df_copy["open"] + eps)
    return df_copy["factor_close_open_gap"]
'''
    report = run_static_alpha_code_guard(code, "factor_close_open_gap")
    assert report.status == GuardStatus.PASS


def test_static_guard_rejects_future_shift():
    code = '''
def factor_future_return(df):
    """Invalid future return."""
    df_copy = df.copy()
    df_copy["factor_future_return"] = df_copy["close"].shift(-1) / df_copy["close"] - 1
    return df_copy["factor_future_return"]
'''
    report = run_static_alpha_code_guard(code, "factor_future_return")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "future_shift" for issue in report.issues)


def test_static_guard_rejects_unknown_input_column():
    code = '''
def factor_uses_target(df):
    """Invalid target access."""
    df_copy = df.copy()
    df_copy["factor_uses_target"] = df_copy["target_return"]
    return df_copy["factor_uses_target"]
'''
    report = run_static_alpha_code_guard(code, "factor_uses_target")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "unknown_input_column" for issue in report.issues)


def test_static_guard_accepts_previously_assigned_df_copy_temp_column():
    code = '''
def factor_range_pressure(df):
    """Range pressure using an internal temporary column."""
    df_copy = df.copy()
    df_copy["range"] = df_copy["high"] - df_copy["low"]
    df_copy["factor_range_pressure"] = df_copy["range"] / (df_copy["close"] + 1e-9)
    return df_copy["factor_range_pressure"]
'''
    report = run_static_alpha_code_guard(code, "factor_range_pressure")
    assert report.status == GuardStatus.PASS


def test_static_guard_rejects_unassigned_df_copy_temp_column_read():
    code = '''
def factor_unassigned_volatility(df):
    """Invalid read of a missing temporary column."""
    df_copy = df.copy()
    df_copy["factor_unassigned_volatility"] = df_copy["volatility"] * df_copy["close"]
    return df_copy["factor_unassigned_volatility"]
'''
    report = run_static_alpha_code_guard(code, "factor_unassigned_volatility")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "unknown_input_column" for issue in report.issues)


def test_static_guard_rejects_df_copy_temp_self_reference_before_assignment():
    code = '''
def factor_self_referenced_volatility(df):
    """Invalid self-reference before temporary column assignment."""
    df_copy = df.copy()
    df_copy["volatility"] = df_copy["volatility"] + df_copy["close"]
    df_copy["factor_self_referenced_volatility"] = df_copy["volatility"]
    return df_copy["factor_self_referenced_volatility"]
'''
    report = run_static_alpha_code_guard(code, "factor_self_referenced_volatility")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "unknown_input_column" for issue in report.issues)


def test_static_guard_rejects_raw_df_temp_column_write():
    code = '''
def factor_raw_df_temp_write(df):
    """Invalid mutation of raw input df with a temporary column."""
    df["temp"] = df["close"] - df["open"]
    return df["temp"]
'''
    report = run_static_alpha_code_guard(code, "factor_raw_df_temp_write")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "unknown_input_column" for issue in report.issues)


def test_static_guard_rejects_raw_df_temp_column_read():
    code = '''
def factor_raw_df_temp_read(df):
    """Invalid read of a non-contract raw input column."""
    return df["temp"]
'''
    report = run_static_alpha_code_guard(code, "factor_raw_df_temp_read")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "unknown_input_column" for issue in report.issues)


def test_static_guard_rejects_custom_rolling_apply():
    code = '''
def factor_trend_strength(df):
    """Invalid expensive rolling apply."""
    df_copy = df.copy()

    def slope_tstat(window):
        return window.mean()

    df_copy["factor_trend_strength"] = (
        df_copy["close"].rolling(window=20).apply(slope_tstat, raw=False)
    )
    return df_copy["factor_trend_strength"]
'''
    report = run_static_alpha_code_guard(code, "factor_trend_strength")
    assert report.status == GuardStatus.FAIL
    assert any(issue.code == "expensive_rolling_apply" for issue in report.issues)
