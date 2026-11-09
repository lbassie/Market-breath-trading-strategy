from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mbts import backtest, indicators, io, metrics

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "MXWO_Market_Breath_data.xlsx"


def test_rsi_bounds_and_trend():
    up = pd.Series(np.linspace(100, 200, 100))
    assert indicators.rsi_wilder(up).iloc[-1] == 100.0
    rng = np.random.default_rng(1)
    rw = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    r = indicators.rsi_wilder(rw).dropna()
    assert r.between(0, 100).all()


def test_indicators_are_causal():
    rng = np.random.default_rng(2)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
    full = indicators.causal_envelope(indicators.rsi_wilder(x))
    cut = indicators.causal_envelope(indicators.rsi_wilder(x.iloc[:200]))
    pd.testing.assert_frame_equal(full.iloc[:200], cut)


def test_backtest_buy_and_hold_matches_asset():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 250), index=pd.bdate_range("2020-01-01", periods=250))
    bt = backtest.run(pd.Series(1.0, index=r.index), r, cost_bps=0.0, lag=0)
    assert np.allclose(bt["ret"], r)


def test_backtest_lag_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=idx)
    pos = pd.Series([1, 1, 0, 0, 1], index=idx, dtype=float)
    bt = backtest.run(pos, r, cost_bps=10.0, lag=1)
    # executed position: 0,1,1,0,0 -> returns: 0, 0.02-0.001, -0.01, 0-0.001, 0
    assert np.allclose(bt["ret"].values, [0, 0.019, -0.01, -0.001, 0])


def test_lookahead_changes_pnl():
    """A signal that peeks at today's return must look perfect with lag 0 and not with lag 1."""
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    cheat = (r > 0).astype(float)
    assert metrics.sharpe(backtest.run(cheat, r, cost_bps=0, lag=0)["ret"]) > 10
    assert abs(metrics.sharpe(backtest.run(cheat, r, cost_bps=0, lag=1)["ret"])) < 2


@pytest.mark.skipif(not DATA.exists(), reason="Bloomberg file not in data/raw")
def test_rsi_matches_bloomberg():
    df = io.load_bloomberg(DATA)
    ours = indicators.rsi_wilder(df["close"]).iloc[100:]
    assert ours.corr(df["rsi14"].iloc[100:]) > 0.99
