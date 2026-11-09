import numpy as np
import pandas as pd

from mbts import io, signals as sig


def _frame(n=700, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "pct_rsi_gt70": rng.uniform(0, 30, n),
        "pct_rsi_lt30": rng.uniform(0, 30, n),
    }, index=idx)


def test_phase2_rules_are_causal():
    df = _frame()
    for f in (lambda d: sig.t1_washout_reentry(d, 200, 0.95, 20),
              lambda d: sig.t2_confirmed_exit(d, 200),
              lambda d: sig.t3_combined(d, 200, 0.95, 20)):
        full, cut = f(df), f(df.iloc[:500])
        pd.testing.assert_series_equal(full.iloc[:500], cut)


def test_expanding_threshold_uses_past_only():
    x = pd.Series(np.arange(300, dtype=float))
    thr = sig.expanding_threshold(x, 0.5, min_periods=10)
    assert thr.iloc[100] == x.iloc[:100].quantile(0.5)


def test_parse_msci_json():
    payload = {"index_variant_type": "NETR", "indexes": {"INDEX_LEVELS": [
        {"level_eod": 2.0, "calc_date": 20060103}, {"level_eod": 1.0, "calc_date": 20060102}]}}
    s = io.parse_msci_json(payload)
    assert list(s.values) == [1.0, 2.0] and s.index[0] == pd.Timestamp("2006-01-02")


def test_parse_fred_and_cash_return():
    s = io.parse_fred_csv("observation_date,DTB3\n2020-01-01,3.65\n2020-01-02,\n2020-01-03,3.65\n")
    assert len(s) == 2
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    c = io.daily_cash_return(s, idx)
    assert c.iloc[0] == 0.0
    assert np.isclose(c.iloc[2], 1.0365 ** (3 / 365) - 1)
