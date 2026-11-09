"""Phase 3: test the frozen T2 rule on long US history (Ken French 49 industries).

Out-of-sample period: 1928-2005 (before the 2006-2023 in-sample). Data after
2023-10-31 is cut off (hold-out stays locked).
Usage: python scripts/run_phase3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mbts import backtest, indicators as ind, io, metrics, signals as sig, validation as val  # noqa: E402

TAG = "2026-11-09"
OUT = ROOT / "outputs" / f"{TAG}_long-history-test"
OUT.mkdir(parents=True, exist_ok=True)
OOS = ("1928-01-01", "2005-12-31")
OVERLAP = ("2006-01-01", "2023-10-31")
DECADES = [(f"{y}s", f"{y}-01-01", f"{y + 9}-12-31") for y in range(1930, 2010, 10)]


def build():
    inds = io.fetch_french_industries_daily(49, "value").loc[: io.IN_SAMPLE_END]
    ff = io.fetch_french_factors_daily().loc[: io.IN_SAMPLE_END]
    prices = (1 + inds).cumprod().where(inds.notna())
    br = ind.breadth_from_prices(prices)
    mkt = ff["Mkt-RF"] + ff["RF"]
    df = pd.DataFrame({"close": (1 + mkt).cumprod(), "pct_rsi_gt70": br["pct_rsi_gt70"],
                       "pct_rsi_lt30": br["pct_rsi_lt30"], "n_members": br["n_members"]})
    return df, mkt, ff["RF"]


def evaluate(pos, asset, cash, a, b, cost=10.0):
    bt = backtest.run(pos, asset, cash, cost_bps=cost).loc[a:b]
    m = metrics.summary(bt["ret"], bt["position"])
    m["Sharpe"] = metrics.sharpe(bt["ret"], cash.loc[a:b])
    return bt, m


def calibration(df):
    bbg = io.split_in_sample(io.load_bloomberg(ROOT / "data" / "raw" / "MXWO_Market_Breath_data.xlsx"))
    j = pd.concat([df[["pct_rsi_gt70", "pct_rsi_lt30"]].add_prefix("ff_"),
                   bbg[["pct_rsi_gt70", "pct_rsi_lt30"]].add_prefix("bb_")], axis=1, join="inner").loc[slice(*OVERLAP)]

    def weak(r1):
        x = r1.rolling(10).mean()
        return x < x.rolling(252, min_periods=252).median()
    wf, wb = weak(j["ff_pct_rsi_gt70"]), weak(j["bb_pct_rsi_gt70"])
    ok = wf.notna() & wb.notna() & j["bb_pct_rsi_gt70"].rolling(262).count().ge(262)
    ntr = io.fetch_msci_levels("NETR")
    ff_yr = (1 + io.fetch_french_factors_daily()[["Mkt-RF", "RF"]].sum(axis=1)).cumprod().resample("YE").last().pct_change()
    ms_yr = ntr.resample("YE").last().pct_change()
    yr = pd.concat([ff_yr, ms_yr], axis=1).loc["2007":"2022"]
    res = {
        "R1 level corr": j["ff_pct_rsi_gt70"].corr(j["bb_pct_rsi_gt70"]),
        "L1 level corr": j["ff_pct_rsi_lt30"].corr(j["bb_pct_rsi_lt30"]),
        "weak-flag agreement": float((wf[ok] == wb[ok]).mean()),
        "annual ret corr US mkt vs MSCI World NTR (2007-22)": yr.corr().iloc[0, 1],
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(j["bb_pct_rsi_gt70"], j["ff_pct_rsi_gt70"], s=2, alpha=0.3)
    ax[0].set_xlabel("Bloomberg MXWO R1 (% members RSI>70)")
    ax[0].set_ylabel("French 49 ind. R1")
    ax[0].set_title(f"R1, 2006-2023, corr {res['R1 level corr']:.2f}", fontsize=9)
    ax[1].scatter(j["bb_pct_rsi_lt30"], j["ff_pct_rsi_lt30"], s=2, alpha=0.3, color="#c0392b")
    ax[1].set_xlabel("Bloomberg MXWO L1 (% members RSI<30)")
    ax[1].set_ylabel("French 49 ind. L1")
    ax[1].set_title(f"L1, 2006-2023, corr {res['L1 level corr']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_calibration.png", dpi=130)
    plt.close(fig)
    return res


def main():
    df, mkt, rf = build()
    cal = calibration(df)
    members = df["n_members"].resample("10YE").min()

    rules = {
        "T0 SMA200": sig.trend_filter(df["close"], 200),
        "T2 confirmed exit": sig.t2_confirmed_exit(df, 200),
        "T1 washout re-entry": sig.t1_washout_reentry(df, 200, 0.95, 20),
        "T3 = T1+T2": sig.t3_combined(df, 200, 0.95, 20),
        "T0 SMA150": sig.trend_filter(df["close"], 150),
        "T2 SMA150": sig.t2_confirmed_exit(df, 150),
        "T0 SMA250": sig.trend_filter(df["close"], 250),
        "T2 SMA250": sig.t2_confirmed_exit(df, 250),
    }
    bh = pd.Series(1.0, index=df.index)
    tables, bts = {}, {}
    for label, (a, b) in {"1928-2005 (OOS)": OOS, "2006-2023 (overlap)": OVERLAP}.items():
        rows = {"Buy & hold US mkt": evaluate(bh, mkt, rf, a, b, cost=0)[1]}
        for name, pos in rules.items():
            bt, m = evaluate(pos, mkt, rf, a, b)
            rows[name] = m
            bts[(label, name)] = bt
            val.log_trial(f"phase3 {name}", {"period": label, "cost_bps": 10}, {k: round(v, 4) for k, v in m.items()},
                          sample="long-history US (French)")
        tables[label] = pd.DataFrame(rows).T
    cost25 = {n: evaluate(rules[n], mkt, rf, *OOS, cost=25)[1] for n in ("T0 SMA200", "T2 confirmed exit")}

    # decade Sharpe
    dec = {}
    for name in ("T0 SMA200", "T2 confirmed exit"):
        r = bts[("1928-2005 (OOS)", name)]["ret"]
        dec[name] = {lab: metrics.sharpe(r.loc[a:b], rf.loc[a:b]) for lab, a, b in DECADES}
    r_bh = backtest.run(bh, mkt, rf, cost_bps=0)["ret"]
    dec["Buy & hold"] = {lab: metrics.sharpe(r_bh.loc[a:b], rf.loc[a:b]) for lab, a, b in DECADES}
    dec = pd.DataFrame(dec)
    dec["T2 - T0"] = dec["T2 confirmed exit"] - dec["T0 SMA200"]

    # paired bootstrap T2 vs T0 (OOS)
    ex = pd.concat([bts[("1928-2005 (OOS)", n)]["ret"] - rf.loc[OOS[0]:OOS[1]] for n in ("T2 confirmed exit", "T0 SMA200")], axis=1)

    def sr_diff(x):
        s = x.mean() / x.std() * np.sqrt(252)
        return s.iloc[0] - s.iloc[1]
    boot = val.block_bootstrap_paired(ex, sr_diff, block=63, n_boot=2000)
    bstat = {"diff": sr_diff(ex), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
             "p(diff<=0)": float((boot <= 0).mean())}

    oos = tables["1928-2005 (OOS)"]
    gates = {
        "a  Sharpe T2 > T0 and bootstrap p < 0.05": bool(oos.loc["T2 confirmed exit", "Sharpe"] > oos.loc["T0 SMA200", "Sharpe"]
                                                       and bstat["p(diff<=0)"] < 0.05),
        "b  T2 > T0 in >= 5 of 8 decades": int((dec["T2 - T0"] > 0).sum()) >= 5,
        "c  same direction at SMA150 and SMA250": bool(oos.loc["T2 SMA150", "Sharpe"] > oos.loc["T0 SMA150", "Sharpe"]
                                                     and oos.loc["T2 SMA250", "Sharpe"] > oos.loc["T0 SMA250", "Sharpe"]),
        "d  switches <= 12/yr": bool(oos.loc["T2 confirmed exit", "Switches/yr"] <= 12),
    }

    # charts
    fig, ax = plt.subplots(figsize=(10, 4.5))
    e = backtest.run(bh, mkt, rf, cost_bps=0)["equity"].loc[OOS[0]:OOS[1]]
    ax.semilogy(e / e.iloc[0], color="#888", label="Buy & hold US market")
    for name, c in (("T0 SMA200", "#2c3e50"), ("T2 confirmed exit", "#2980b9")):
        q = bts[("1928-2005 (OOS)", name)]["equity"]
        ax.semilogy(q / q.iloc[0], color=c, lw=1, label=name)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("US market 1928-2005 (out of sample), T-bill cash, 10 bp per side")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_equity-1928-2005.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cols = ["#27ae60" if v > 0 else "#c0392b" for v in dec["T2 - T0"]]
    ax.bar(dec.index, dec["T2 - T0"], color=cols)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title("Sharpe difference T2 - T0 (SMA200) by decade")
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_decades.png", dpi=130)
    plt.close(fig)

    for k, t in tables.items():
        t.round(4).to_csv(OUT / f"{TAG}_results_{k.split()[0]}.csv")
    dec.round(4).to_csv(OUT / f"{TAG}_decades.csv")

    pd.set_option("display.width", 200)
    print("## calibration", {k: round(v, 3) for k, v in cal.items()})
    print("## min industries with data per decade", members.astype(int).to_dict())
    for k, t in tables.items():
        print(f"\n## {k}\n", t.round(3).to_string())
    print("\n## OOS at 25 bp\n", pd.DataFrame(cost25).T.round(3).to_string())
    print("\n## decades\n", dec.round(3).to_string())
    print("\n## bootstrap T2 - T0 (OOS, 63d blocks)", {k: round(v, 3) for k, v in bstat.items()})
    print("\n## gates", gates)


if __name__ == "__main__":
    main()
