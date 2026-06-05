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
