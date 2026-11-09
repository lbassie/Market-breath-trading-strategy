"""Phase 2: breadth as a filter on the SMA trend rule. In-sample only (<= 2023-10-31).

Usage: python scripts/run_phase2.py
Data: Bloomberg breadth file (data/raw), MSCI World NTR/PRICE USD (msci.com), 3m T-bill (FRED DTB3).
Writes tables and charts to outputs/2026-11-09_trend-breadth-study/.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mbts import backtest, io, metrics, signals as sig, validation as val  # noqa: E402

TAG = "2026-11-09"
OUT = ROOT / "outputs" / f"{TAG}_trend-breadth-study"
OUT.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2007-01-02")  # after 252-day warm-up of the causal thresholds
PERIODS = {"2007-12": ("2007-01-01", "2012-12-31"), "2013-19": ("2013-01-01", "2019-12-31"),
           "2020-23": ("2020-01-01", "2023-10-31")}
GRID_N, GRID_Q, GRID_H = (150, 200, 250), (0.90, 0.95, 0.975), (10, 20, 40)
DEFAULT = dict(n=200, q=0.95, hold=20)


def load():
    bbg = io.load_bloomberg(ROOT / "data" / "raw" / "MXWO_Market_Breath_data.xlsx")
    ntr = io.fetch_msci_levels("NETR")
    prc = io.fetch_msci_levels("STRD")
    tbill = io.fetch_fred("DTB3")
    try:
        io.fetch_ecb_estr()  # cached for a later EUR check; not used in phase 2
    except Exception as e:  # noqa: BLE001
        print("ECB fetch failed (not needed for phase 2):", e)
    idx = bbg.index.intersection(ntr.index)
    df = io.split_in_sample(bbg.loc[idx])
    idx = df.index
    asset = ntr.reindex(idx).pct_change().fillna(0.0)
    cash = io.daily_cash_return(tbill, idx)
    return bbg, ntr, prc, df, asset, cash


def data_checks(bbg, ntr, prc) -> dict:
    common = bbg.index.intersection(prc.index)
    rb, rm = bbg["close"].reindex(common).pct_change(), prc.reindex(common).pct_change()
    yr_ntr = ntr.resample("YE").last().pct_change()
    yr_prc = prc.resample("YE").last().pct_change()
    return {
        "common_dates_bbg_msci": len(common),
        "bbg_dates_missing_in_msci": len(bbg.index.difference(prc.index)),
        "ret_corr_bbg_vs_msci_price": rb.corr(rm),
        "max_abs_level_diff_pct": float((bbg["close"].reindex(common) / prc.reindex(common) - 1).abs().max() * 100),
        "avg_annual_ntr_minus_price_pct": float((yr_ntr - yr_prc).loc["2007":"2023"].mean() * 100),
    }


def evaluate(pos, asset, cash, cost=10.0):
    bt = backtest.run(pos, asset, cash, cost_bps=cost)
    bt = bt.loc[START:]
    ex = bt["ret"] - cash.loc[START:]
    m = metrics.summary(bt["ret"], bt["position"])
    m["Sharpe"] = metrics.sharpe(bt["ret"], cash.loc[START:])  # excess over T-bill
    return bt, ex, m


def rules(df):
    out = {}
    for n in GRID_N:
        out[("T0", n, None, None)] = sig.trend_filter(df["close"], n)
        out[("T2", n, None, None)] = sig.t2_confirmed_exit(df, n)
        for q, h in product(GRID_Q, GRID_H):
            out[("T1", n, q, h)] = sig.t1_washout_reentry(df, n, q, h)
            out[("T3", n, q, h)] = sig.t3_combined(df, n, q, h)
    return out


def main():
    bbg, ntr, prc, df, asset, cash = load()
    checks = data_checks(bbg, ntr, prc)

    # --- all trials --------------------------------------------------------
    pos_all = rules(df)
    res, ex_all, bts = {}, {}, {}
    for key, pos in pos_all.items():
        bt, ex, m = evaluate(pos, asset, cash)
        res[key], ex_all[key], bts[key] = m, ex, bt
        val.log_trial(key[0], {"n": key[1], "q": key[2], "hold": key[3], "cost_bps": 10}, {
            k: round(v, 4) for k, v in m.items()}, sample="in-sample phase 2")
    grid = pd.DataFrame(res).T
    grid.index.names = ["rule", "n", "q", "hold"]
    grid.round(4).to_csv(OUT / f"{TAG}_grid.csv")

    bh_bt, bh_ex, bh = evaluate(pd.Series(1.0, index=df.index), asset, cash, cost=0.0)

    # --- pre-registered defaults ---------------------------------------------
    d = DEFAULT
    keys = {"T0 SMA200": ("T0", 200, None, None), "T1 washout re-entry": ("T1", 200, d["q"], d["hold"]),
            "T2 confirmed exit": ("T2", 200, None, None), "T3 = T1+T2": ("T3", 200, d["q"], d["hold"])}
    rows = {"Buy & hold NTR": bh}
    rows25 = {}
    for name, k in keys.items():
        rows[name] = res[k]
        rows25[name] = evaluate(pos_all[k], asset, cash, cost=25.0)[2]
    table = pd.DataFrame(rows).T
    table25 = pd.DataFrame(rows25).T
    table.round(4).to_csv(OUT / f"{TAG}_defaults.csv")

    # sub-periods (Sharpe excess over cash)
    sub = {}
    for name, k in [("Buy & hold NTR", None)] + list(keys.items()):
        r = bh_bt["ret"] if k is None else bts[k]["ret"]
        sub[name] = {p: metrics.sharpe(r.loc[a:b], cash.loc[a:b]) for p, (a, b) in PERIODS.items()}
    sub = pd.DataFrame(sub).T

    # paired bootstrap: Sharpe(Tx) - Sharpe(T0)
    def sr_diff(x):
        s = x.mean() / x.std() * np.sqrt(252)
        return s.iloc[0] - s.iloc[1]
    boot = {}
    for name, k in list(keys.items())[1:]:
        pair = pd.concat([ex_all[k], ex_all[keys["T0 SMA200"]]], axis=1)
        b = val.block_bootstrap_paired(pair, sr_diff, n_boot=2000)
        boot[name] = {"diff": sr_diff(pair), "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5),
                      "p(diff<=0)": float((b <= 0).mean())}
    boot = pd.DataFrame(boot).T

    # parameter stability: neighbours of the default in the grid
    stab = {}
    for rule in ("T1", "T3"):
        neigh = grid.loc[rule]["Sharpe"]
        stab[rule] = {"default": neigh.loc[(200, d["q"], d["hold"])], "min_grid": neigh.min(),
                      "max_grid": neigh.max(), "share_grid_above_T0": float((neigh > res[keys["T0 SMA200"]]["Sharpe"]).mean())}
    stab = pd.DataFrame(stab).T

    # DSR for the best breadth trial, PBO over all phase-2 trials
    ex_df = pd.DataFrame({str(k): v for k, v in ex_all.items()})
    best_key = max((k for k in res if k[0] != "T0"), key=lambda k: res[k]["Sharpe"])
    best_ex = ex_all[best_key]
    daily_srs = ex_df.mean() / ex_df.std()
    n_trials = sum(1 for _ in open(val.TRIAL_LOG)) - 1
    dsr = metrics.deflated_sharpe(float(best_ex.mean() / best_ex.std()), len(best_ex), float(best_ex.skew()),
                                  float(best_ex.kurt() + 3), n_trials, float(daily_srs.var()))
    pbo = metrics.pbo_cscv(ex_df, n_splits=16)

    # gates (in-sample) for each default rule
    t0 = res[keys["T0 SMA200"]]
    gates = {}
    for name in list(keys)[1:]:
        m = rows[name]
        gates[name] = {
            "1 Sharpe>=BH+0.15 or MDD 30% smaller": (m["Sharpe"] >= bh["Sharpe"] + 0.15)
            or (m["Sharpe"] >= bh["Sharpe"] and abs(m["MaxDD"]) <= 0.7 * abs(bh["MaxDD"])),
            "2 Sharpe > T0": m["Sharpe"] > t0["Sharpe"],
            "5 beats T0 in >=2 of 3 sub-periods": int((sub.loc[name] > sub.loc["T0 SMA200"]).sum()) >= 2,
            "6 switches <= 12/yr": m["Switches/yr"] <= 12,
            "bootstrap p(diff<=0) < 0.05": boot.loc[name, "p(diff<=0)"] < 0.05,
        }
    gates = pd.DataFrame(gates).T
    gates["3 DSR > 0.95 (best trial)"] = dsr > 0.95
    gates["4 PBO < 0.2"] = pbo < 0.2

    # --- charts ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(bh_bt["equity"] / bh_bt["equity"].iloc[0], label="Buy & hold NTR", color="#888")
    for name, c in [("T0 SMA200", "#2c3e50"), ("T2 confirmed exit", "#2980b9"), ("T3 = T1+T2", "#c0392b"), ("T1 washout re-entry", "#27ae60")]:
        e = bts[keys[name]]["equity"]
        ax.semilogy(e / e.iloc[0], label=name, color=c, lw=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("In-sample equity, MSCI World NTR USD, T-bill cash, 10 bp per side (2007-2023)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_equity.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, rule in zip(axes, ("T1", "T3")):
        h = grid.sort_index().loc[(rule, 200)]["Sharpe"].astype(float).unstack("hold")
        im = ax.imshow(h.values, cmap="RdYlGn", vmin=t0["Sharpe"] - 0.15, vmax=t0["Sharpe"] + 0.15)
        ax.set_xticks(range(len(h.columns)), h.columns)
        ax.set_yticks(range(len(h.index)), h.index)
        ax.set_xlabel("hold days")
        ax.set_ylabel("L1 percentile")
        ax.set_title(f"{rule} Sharpe, SMA200 (T0 = {t0['Sharpe']:.2f})", fontsize=9)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                ax.text(j, i, f"{h.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / f"{TAG}_heatmaps.png", dpi=130)
    plt.close(fig)

    # --- print -------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("## data checks", {k: round(v, 5) if isinstance(v, float) else v for k, v in checks.items()})
    print("\n## defaults, 10 bp\n", table.round(3).to_string())
    print("\n## defaults, 25 bp\n", table25.round(3).to_string())
    print("\n## sub-period Sharpe\n", sub.round(3).to_string())
    print("\n## bootstrap Sharpe diff vs T0\n", boot.round(3).to_string())
    print("\n## grid stability\n", stab.round(3).to_string())
    print(f"\n## best breadth trial {best_key}: Sharpe {res[best_key]['Sharpe']:.3f}; "
          f"trials logged {n_trials}; DSR {dsr:.3f}; PBO {pbo:.3f}")
    print("\n## gates\n", gates.to_string())
    print("\n## grid top 10\n", grid.sort_values("Sharpe", ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
