"""Event study, bootstrap and trial log."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRIAL_LOG = Path(__file__).resolve().parents[2] / "outputs" / "trial_log.csv"


def forward_returns(close: pd.Series, horizons=(5, 20, 60)) -> pd.DataFrame:
    """Return from close t to close t+h (log-free, simple)."""
    return pd.DataFrame({f"fwd{h}": close.shift(-h) / close - 1 for h in horizons})


def event_study(
    close: pd.Series, events: pd.Series, horizons=(5, 20, 60), n_boot: int = 5000, seed: int = 0
) -> pd.DataFrame:
    """Mean forward return after (declustered) events vs the unconditional mean.

    Reports: number of events, mean, hit rate, excess vs all days, a bootstrap
    95% CI of the excess (resampling events), and a Newey-West t-stat from an
    all-days regression fwd_h = a + b*event (lags = h, for overlapping windows).
    """
    rng = np.random.default_rng(seed)
    fwd = forward_returns(close, horizons)
    rows = []
    for h in horizons:
        col = fwd[f"fwd{h}"]
        valid = col.notna()
        ev = events.reindex(col.index).fillna(False).astype(bool) & valid
        x = col[ev].to_numpy()
        base = col[valid].mean()
        if len(x) < 3:
            rows.append({"h": h, "n": len(x)})
            continue
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1) - base
        y = col[valid]
        X = sm.add_constant(ev[valid].astype(float))
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        rows.append({
            "h": h, "n": len(x), "mean": x.mean(), "hit": (x > 0).mean(), "uncond": base,
            "excess": x.mean() - base, "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
            "nw_t": res.tvalues.iloc[1],
        })
    return pd.DataFrame(rows).set_index("h")


def block_bootstrap(r: pd.Series, stat, block: int = 21, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Stationary-block bootstrap distribution of stat(r)."""
    rng = np.random.default_rng(seed)
    x = r.dropna().to_numpy()
    n = len(x)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + block, n)))
        out[b] = stat(pd.Series(x[idx[:n]]))
    return out


def log_trial(name: str, params: dict, metrics: dict, sample: str = "in-sample", path: Path = TRIAL_LOG) -> None:
    """Append one tried variant. The count of trials feeds the Deflated Sharpe Ratio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "name", "sample", "params", "metrics"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), name, sample, params, metrics])
