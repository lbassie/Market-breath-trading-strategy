"""Vectorised long/cash backtest."""
from __future__ import annotations

import pandas as pd


def run(
    position: pd.Series,
    asset_ret: pd.Series,
    cash_ret: pd.Series | float = 0.0,
    cost_bps: float = 10.0,
    lag: int = 1,
    ter_diff: float = 0.0,
) -> pd.DataFrame:
    """Daily P&L of a {0..1} position.

    position[t] is decided at the close of t. It earns asset_ret from t+lag onward
    (lag=1: trade at the next close). cost_bps is charged per side on |change|.
    ter_diff is an annual fee charged on the invested part (e.g. 0.002).
    """
    pos = position.reindex(asset_ret.index).ffill().fillna(0.0).shift(lag).fillna(0.0)
    if not isinstance(cash_ret, pd.Series):
        cash_ret = pd.Series(cash_ret, index=asset_ret.index)
    cash_ret = cash_ret.reindex(asset_ret.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps / 1e4
    fee = pos * ter_diff / 252.0
    ret = pos * asset_ret + (1.0 - pos) * cash_ret - cost - fee
    return pd.DataFrame({"position": pos, "turnover": turnover, "ret": ret, "equity": (1 + ret).cumprod()})
