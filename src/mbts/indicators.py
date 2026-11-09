"""Technical and breadth indicators. Every function here is causal: the value at
date t uses only data up to and including t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (smoothing alpha = 1/n), as used by Bloomberg RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return rsi


def causal_envelope(
    x: pd.Series, ma_period: int = 21, vol_period: int = 21, n_std: float = 2.0
) -> pd.DataFrame:
    """Causal volatility envelope: trailing SMA +/- n_std * trailing std.

    This is NOT the Bloomberg VBE (El Saiid). The VBE formula is proprietary and
    its description mentions a *centered* MA, which would look ahead. We use this
    trailing version as a lookahead-free replacement with the same periods.
    """
    mid = x.rolling(ma_period, min_periods=ma_period).mean()
    sd = x.rolling(vol_period, min_periods=vol_period).std()
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd})


def breadth_from_prices(prices: pd.DataFrame, n: int = 14, hi: float = 70.0, lo: float = 30.0) -> pd.DataFrame:
    """Percent of members with RSI > hi and RSI < lo.

    prices: wide frame (dates x tickers). A ticker counts only on dates where it
    has a valid RSI (NaN outside its membership / trading history).
    """
    rsi = prices.apply(rsi_wilder, n=n)
    valid = rsi.notna().sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(
        {
            "pct_rsi_gt70": 100.0 * (rsi > hi).sum(axis=1) / valid,
            "pct_rsi_lt30": 100.0 * (rsi < lo).sum(axis=1) / valid,
            "n_members": valid,
        }
    )


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std()
    return (x - m) / s
