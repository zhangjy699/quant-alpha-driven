# Operator Library

Common WorldQuant-style building blocks and safe **pandas/numpy/talib** mappings for CogAlpha alpha-code agents. Obey [Alpha Factor Contract](./alpha-factor-contract.md).

## Scope — Guidance, Not an Allowlist

- This catalog is **illustrative and inspirational**: familiar names, economic intuition, and implementation hints.
- It is **not** a closed list. You may use any vectorized OHLCV expression permitted by the Alpha Factor Contract — including `pct_change`, `ewm`, algebraic combinations, and other **allowlisted** `np` / `pd` / `stats` / `talib` / `math` APIs — even when no matching row appears below.
- Prefer catalog operators when they fit; invent equivalent pandas/talib code when the economic idea needs something not listed.

## Raw Factor Policy

- Alpha functions output a **raw** single-stock signal from OHLCV only.
- Cross-sectional ranking, industry/cap neutralization, and quantile layering run **after** per-stock execution in the CogAlpha engine — **do not** implement them in alpha code.
- Inputs: only `open`, `high`, `low`, `close`, `volume`.
- Window `d` = past `d` **bars** on the current stock (not calendar days).
- No future data: never `shift(-d)` or centered windows.
- Prefer vectorized `rolling` aggregations; avoid custom `rolling.apply` callbacks.
- One clear economic mechanism per factor.

---

## Shared Implementation Patterns

Use these templates; do not reinvent per-operator code blocks.

**1. Column access** — `col("close")` → `df_copy["close"]`

**2. Rolling** — `x.rolling(d, min_periods=d).mean()` (also `std`, `sum`, `max`, `min`, `rank`, `corr`, `skew`)

**3. Lag / delta** — `x.shift(d)` with `d > 0`; `x - x.shift(d)`

**4. Time-series normalize**

```python
mean_ = x.rolling(d, min_periods=d).mean()
std_ = x.rolling(d, min_periods=d).std(ddof=1)
z = (x - mean_) / std_.replace(0.0, np.nan)
pct = x.rolling(d, min_periods=d).rank(pct=True)
denom = x.abs().rolling(d, min_periods=d).sum().replace(0.0, np.nan)
scaled = x / denom
```

**5. Safe division** — `a / b.replace(0.0, np.nan)` then `fillna` if needed

**6. TA-Lib** — `pd.Series(talib.FUNC(arr), index=df_copy.index)`

---

## Operator Catalog

### Column

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `col` | OHLCV column reference | `df_copy["close"]` etc. | |

### Arithmetic

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `abs` | Absolute magnitude | `x.abs()` | |
| `log` | Log transform | `np.log(x.clip(lower=1e-12))` | |
| `exp` | Exponential | `np.exp(x.clip(upper=10))` | |
| `sqrt` | Square root | `np.sqrt(x.clip(lower=0))` | |
| `sin` | Sine transform | `np.sin(x)` | |
| `cos` | Cosine transform | `np.cos(x)` | |
| `sign` | Direction sign | `np.sign(x)` | |
| `reverse` | Negate value | `-x` | |
| `inverse` | Reciprocal | `1.0 / x.replace(0, np.nan)` | |
| `power` | Raise to power | `np.power(x, y)` | |
| `signed_power` | Signed power | `np.sign(x) * np.abs(x) ** y` | |
| `add` | Sum features | `a + b` | |
| `multiply` | Product / scale | `a * b` | |
| `subtract` | Spread / difference | `a - b` | |
| `max` | Element-wise max | `np.maximum(a, b)` | |
| `min` | Element-wise min | `np.minimum(a, b)` | |
| `divide` | Ratio | `a / b.replace(0, np.nan)` | |

### Logical

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `and_` | Logical and | `a & b` (boolean) | |
| `or_` | Logical or | `a \| b` (boolean) | |
| `not_` | Logical not | `~cond` | |
| `if_else` | Regime switch | `np.where(cond, a, b)` | |
| `is_nan` | Missing flag | `x.isna()` | |
| `lt` | Less than | `a < b` | |
| `le` | Less or equal | `a <= b` | |
| `eq` | Equal | `a == b` | |
| `gt` | Greater than | `a > b` | |
| `ge` | Greater or equal | `a >= b` | |
| `ne` | Not equal | `a != b` | |

### Time-Series

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `ts_mean` | Rolling mean | `.rolling(d).mean()` | |
| `ts_std_dev` / `ts_std` | Rolling volatility | `.rolling(d).std(ddof=1)` | |
| `ts_max` | Rolling high | `.rolling(d).max()` | |
| `ts_min` | Rolling low | `.rolling(d).min()` | |
| `ts_delay` / `delay` | Lag d bars | `x.shift(d)`, d>0 | risky |
| `ts_delta` | Change over d bars | `x - x.shift(d)` | |
| `ts_sum` | Rolling sum | `.rolling(d).sum()` | |
| `ts_product` | Rolling product | on `(1+ret)` window | risky |
| `ts_av_diff` | Deviation from mean | `x - x.rolling(d).mean()` | |
| `ts_zscore` | Time-series z-score | pattern §4 | |
| `ts_rank` | Time-series percentile | `.rolling(d).rank(pct=True)` | |
| `ts_scale` | Scale by abs sum | `x / x.abs().rolling(d).sum()` | |
| `ts_count_nans` | NaN count in window | `.rolling(d).apply(np.isnan).sum()` | |
| `ts_backfill` | Forward-fill from past | `x.ffill()` | |
| `ts_arg_max` | Bars since window max | `.rolling(d).apply(np.argmax)` | |
| `ts_arg_min` | Bars since window min | `.rolling(d).apply(np.argmin)` | |
| `ts_decay_linear` | Linear weighted mean | weighted rolling sum | risky |
| `ts_corr` | Rolling correlation | `a.rolling(d).corr(b)` | |
| `ts_covariance` | Rolling covariance | `a.rolling(d).cov(b)` | |
| `ts_regression` | Rolling OLS slope | `cov/var` or lstsq | risky |
| `ts_quantile` | TS rank to distribution | rank pct → `stats.norm.ppf` | |
| `days_from_last_change` | Bars since value changed | `changed.cumcount()` | |
| `kth_element` | k-th order stat in window | `rolling.apply` sort | risky |
| `last_diff_value` | Last distinct past value | shift + mask fill | risky |
| `ts_skew` | Rolling skewness | `.rolling(d).skew()` | |
| `ts_kurt` | Rolling kurtosis | `.rolling(d).kurt()` | |
| `ts_step` | Periodic step counter | anchor-based cumsum | risky |
| `hump` | Limit bar-to-bar move | `x.diff().clip(±h)` approx | risky |

### Cleaning

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `pasteurize` | Replace inf/invalid | `replace(inf,nan).fillna(0)` | |
| `tail` | Trim distribution tails | `x.clip(lo, hi)` quantiles | |
| `protected_div` | Safe ratio | `a / b.replace(0, np.nan)` | |
| `protected_log` | Safe log | `np.log(x.clip(lower=eps))` | |
| `protected_sqrt` | Safe sqrt | `np.sqrt(x.clip(lower=0))` | |

### Technical — Moving Averages

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `ts_sma` | Simple MA | `talib.SMA(close, d)` | |
| `ts_ema` | Exponential MA | `talib.EMA(close, d)` | |
| `ts_dema` | Double EMA | `talib.DEMA(close, d)` | |
| `ts_wma` | Weighted MA | `talib.WMA(close, d)` | |
| `ts_kama` | Adaptive MA | `talib.KAMA(close, d)` | |
| `ts_tema` | Triple EMA | `talib.TEMA(close, d)` | |
| `ts_trima` | Triangular MA | `talib.TRIMA(close, d)` | |
| `ts_t3` | T3 MA | `talib.T3(close, d)` | |

### Technical — Oscillators

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `ts_rsi` | RSI momentum | `talib.RSI(close, d)` | |
| `ts_macd` | MACD line | `talib.MACD(close)` | |
| `ts_cci` | Commodity Channel Index | `talib.CCI(H,L,C,d)` | |
| `ts_stoch` | Stochastic %K | `talib.STOCH(H,L,C)` | |
| `ts_stochf` | Fast stochastic | `talib.STOCHF(H,L,C)` | |
| `ts_stochrsi` | StochRSI | `talib.STOCHRSI(close, d)` | |
| `ts_willr` | Williams %R | `talib.WILLR(H,L,C,d)` | |
| `ts_cmo` | Chande momentum | `talib.CMO(close, d)` | |
| `ts_ppo` | PPO oscillator | `talib.PPO(close)` | |
| `ts_apo` | APO oscillator | `talib.APO(close)` | |
| `ts_ultosc` | Ultimate oscillator | `talib.ULTOSC(H,L,C)` | |
| `ts_mfi` | Money Flow Index | `talib.MFI(H,L,C,V,d)` | |
| `ts_trix` | TRIX momentum | `talib.TRIX(close, d)` | |
| `ts_bop` | Balance of power | `talib.BOP(H,L,C)` | |

### Technical — Channels & Volatility

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `ts_bbands` | Bollinger bands | `talib.BBANDS(close, d)` | |
| `ts_trange` | True range | `talib.TRANGE(H,L,C)` | |
| `ts_atr` | Average true range | `talib.ATR(H,L,C,d)` | |
| `ts_natr` | Normalized ATR | `talib.NATR(H,L,C,d)` | |
| `ts_donchian` | Donchian channel | `talib.MAX/MIN` on H/L | |
| `ts_keltner` | Keltner channel | EMA ± ATR multiple | |
| `ts_ma_envelope` | MA envelope bands | SMA ± pct band | |

### Technical — Trend & Volume

| name | meaning | impl hint | note |
|------|---------|-----------|------|
| `ts_roc` | Rate of change | `talib.ROC(close, d)` | |
| `ts_rocr` | ROC ratio | `talib.ROCR(close, d)` | |
| `ts_rocr100` | ROC ratio × 100 | `talib.ROCR100(close, d)` | |
| `ts_mom` | Momentum | `talib.MOM(close, d)` | |
| `ts_obv` | On-balance volume | `talib.OBV(close, volume)` | |
| `ts_ad` | Chaikin A/D line | `talib.AD(H,L,C,V)` | |
| `ts_adosc` | Chaikin oscillator | `talib.ADOSC(H,L,C,V)` | |
| `ts_adx` | Trend strength ADX | `talib.ADX(H,L,C,d)` | |
| `ts_adxr` | ADX rating | `talib.ADXR(H,L,C,d)` | |
| `ts_dx` | Directional movement | `talib.DX(H,L,C,d)` | |
| `ts_aroon` | Aroon up/down | `talib.AROON(H,L,d)` | |
| `ts_sar` | Parabolic SAR | `talib.SAR(H,L)` | |
| `ts_linearreg_slope` | Regression slope | `talib.LINEARREG_SLOPE(close,d)` | |
| `ts_linearreg_angle` | Regression angle | `talib.LINEARREG_ANGLE(close,d)` | |

---

## Pitfalls

- `ts_delay` / `delay`: `d` must be **positive** — negative shift leaks future data.
- `ts_rank` / `ts_zscore`: time-series over the stock's own history, not a multi-stock slice.
- `ts_product`: apply to returns or bounded ratios, not raw prices.
- `hump` / `ts_step`: WQ semantics are stateful; use vectorized approximations only.
- Keep factors **raw** — no final cross-sectional standardization or neutralization in alpha code.
