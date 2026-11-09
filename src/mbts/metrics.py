"""Performance and overfitting statistics."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ANN = 252


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (ANN / len(r)) - 1)


def vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(ANN))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = (r - rf).dropna()
    return float(ex.mean() / ex.std() * np.sqrt(ANN)) if ex.std() > 0 else np.nan


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    down = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / down * np.sqrt(ANN)) if down > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return cagr(r) / abs(mdd) if mdd < 0 else np.nan


def switches_per_year(position: pd.Series) -> float:
    return float(position.diff().abs().gt(0).sum() / (len(position) / ANN))


def alpha_tstat(r: pd.Series, bench: pd.Series, nw_lags: int = 5) -> tuple[float, float]:
    """Annualised alpha and its Newey-West t-stat from r = a + b*bench."""
    d = pd.concat([r, bench], axis=1).dropna()
    res = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    return float(res.params.iloc[0] * ANN), float(res.tvalues.iloc[0])


def summary(r: pd.Series, position: pd.Series | None = None) -> dict:
    out = {
        "CAGR": cagr(r), "Vol": vol(r), "Sharpe": sharpe(r), "Sortino": sortino(r),
        "MaxDD": max_drawdown(r), "Calmar": calmar(r),
    }
    if position is not None:
        out["TimeInMkt"] = float(position.mean())
        out["Switches/yr"] = switches_per_year(position)
    return out


def deflated_sharpe(sr_daily: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_trials_var: float) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014). Inputs are per-period (daily) values.

    kurt is the raw (non-excess) kurtosis. sr_trials_var is the variance of the
    daily Sharpe ratios of all trials tried.
    """
    emc = 0.5772156649
    sr0 = np.sqrt(sr_trials_var) * (
        (1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials) + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    num = (sr_daily - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily**2)
    return float(stats.norm.cdf(num / den))


def pbo_cscv(trial_returns: pd.DataFrame, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

    trial_returns: dates x trials (daily returns). Returns the share of
    combinations where the in-sample best trial ranks below the median out of sample.
    """
    blocks = np.array_split(np.arange(len(trial_returns)), n_splits)
    x = trial_returns.to_numpy()
    logits = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_idx])
        sr_is = x[is_rows].mean(0) / x[is_rows].std(0)
        sr_oos = x[oos_rows].mean(0) / x[oos_rows].std(0)
        best = int(np.nanargmax(sr_is))
        rank = stats.rankdata(sr_oos)[best] / (len(sr_oos) + 1)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))
