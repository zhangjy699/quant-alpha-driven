# Alpha Factor Contract

All generated factors must obey this contract.

## Input

- The input is a pandas DataFrame for one stock's daily time series.
- Rows are sorted from earliest date to latest date.
- The index is preserved and represents `(date, ticker)` in the wider dataset.
- The only input columns are `open`, `high`, `low`, `close`, and `volume`.

## Output

- Return one pandas Series with the same index as the input DataFrame.
- The Series name must exactly match the Python function name.
- The function name must start with `factor_` and be descriptive.

## Alpha Runtime Allowlist

Generated code may use only these pre-imported names:

- `np` for numpy
- `pd` for pandas
- `stats` for scipy.stats
- `talib` for TA-Lib
- `math` for Python math

## Hard Rules

- Use only present and past observations.
- Never use negative shifts, centered rolling windows, reversed time order, future labels, or target returns.
- Do not import libraries outside the allowlist.
- Do not use `eval`, `exec`, file IO, network IO, subprocesses, or global mutable state.
- Avoid nested loops and infinite loops.
- Do not use custom Python callbacks inside `rolling(...).apply(...)`; use vectorized rolling aggregations such as `mean`, `std`, `sum`, `min`, `max`, `rank`, or algebraic combinations.
- Prefer vectorized pandas/numpy operations.
- Keep one clear economic idea per factor.
- Keep the factor concise; if more than 3 logical steps are needed, the rationale must explain why.

## Column Access Rules

- Treat `open`, `high`, `low`, `close`, and `volume` as the only external input columns.
- Prefer local variables for intermediate calculations, for example `range_ = df_copy["high"] - df_copy["low"]`.
- You may create temporary columns on `df_copy`, but only read them after assigning them in the same function.
- Never read uncreated columns such as `df_copy["target_return"]`, `df_copy["future_return"]`, `df_copy["label"]`, or `df_copy["volatility"]` unless that exact temporary column was assigned earlier in the function.
- Do not create temporary columns on the raw input `df`; use `df_copy` or local variables.

## Function Shape

```python
def factor_example_name(df):
    """One clear rationale and short formula."""
    df_copy = df.copy()
    range_ = df_copy["high"] - df_copy["low"]
    # factor computation using local variables or previously assigned df_copy columns
    df_copy["factor_example_name"] = ...
    return df_copy["factor_example_name"]
```
