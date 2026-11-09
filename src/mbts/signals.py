"""Hypotheses H1-H5 as event and position rules.

Thresholds are given as percentiles of the IN-SAMPLE distribution (fixed before
the event study). Each event function returns a boolean Series (True = event on
that close). Positions are {0, 1} at the close of t; the backtest applies the
one-day execution lag.
"""
from __future__ import annotations

import pandas as pd

from .indicators import zscore


def decluster(events: pd.Series, cooldown: int = 20) -> pd.Series:
    """Keep the first event; drop events within `cooldown` days after a kept one."""
    out = pd.Series(False, index=events.index)
    last = -10**9
    for i, flag in enumerate(events.to_numpy()):
        if flag and i - last > cooldown:
            out.iloc[i] = True
            last = i
    return out


def h1_washout(df: pd.DataFrame, l1_thr: float) -> pd.Series:
    """H1: share of members with RSI<30 at an extreme -> buy."""
    return df["pct_rsi_lt30"] >= l1_thr


def h2_rsi_below_band(df: pd.DataFrame, lower: pd.Series) -> pd.Series:
    """H2: index RSI crosses below its lower envelope -> mean reversion (buy)."""
    below = df["rsi14"] < lower
    return below & ~below.shift(1, fill_value=False)


def h3_divergence(df: pd.DataFrame, r1_window: int = 10, lookback: int = 252) -> pd.Series:
    """H3: index within 1% of its 252d high while 10d avg R1 is below its 252d median -> risk-off."""
    near_high = df["close"] >= 0.99 * df["close"].rolling(lookback).max()
    r1_avg = df["pct_rsi_gt70"].rolling(r1_window).mean()
    weak = r1_avg < r1_avg.rolling(lookback).median()
    return near_high & weak


def h4_capitulation(df: pd.DataFrame, l1_thr: float, vol_z: float = 2.0) -> pd.Series:
    """H4: volume spike (63d z-score > vol_z) together with an L1 extreme -> buy."""
    return (zscore(df["volume"], 63) > vol_z) & (df["pct_rsi_lt30"] >= l1_thr)


def h5_thrust(df: pd.DataFrame, l1_thr: float, r1_thr: float, window: int = 10) -> pd.Series:
    """H5: breadth thrust: L1 extreme in the last `window` days, now R1 high -> buy."""
    recent_wash = (df["pct_rsi_lt30"] >= l1_thr).rolling(window, min_periods=1).max().astype(bool)
    return recent_wash & (df["pct_rsi_gt70"] >= r1_thr)


def trend_filter(close: pd.Series, n: int = 200) -> pd.Series:
    """Benchmark rule: long when close > n-day SMA, else cash."""
    sma = close.rolling(n, min_periods=n).mean()
    return (close > sma).astype(float).where(sma.notna(), 1.0)
