"""Phase 4: the single, pre-registered hold-out run of the SMA200 filter (T0).

Hold-out: 2023-11-01 to the last date in the data. Rule frozen: long MSCI World if the
price index (USD) closes above its 200-day SMA, else cash; trade at the next close; 10 bp.
EUR version: MSCI World NTR EUR minus 0.20%/yr TER (IWDA proxy), cash = ESTR.
USD version: MSCI World NTR USD, cash = 3m T-bill.
Usage: python scripts/run_phase4_holdout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mbts import backtest, io, metrics, signals as sig, validation as val  # noqa: E402

TAG = "2026-11-09"
OUT = ROOT / "outputs" / f"{TAG}_final-strategy-note"
OUT.mkdir(parents=True, exist_ok=True)
HO_START = pd.Timestamp("2023-11-01")


def worst_12m_gap(strat: pd.Series, bh: pd.Series) -> float:
    s = (1 + strat).rolling(252).apply(np.prod, raw=True) - 1
    b = (1 + bh).rolling(252).apply(np.prod, raw=True) - 1
    return float((s - b).min())


def run_version(close, asset_lvl, cash_rate, ter, label):
    idx = close.index.intersection(asset_lvl.index)
    close = close.reindex(idx)
    asset = asset_lvl.reindex(idx).pct_change().fillna(0.0)
    cash = io.daily_cash_return(cash_rate, idx)
    pos = sig.trend_filter(close, 200)
    bt = backtest.run(pos, asset, cash, cost_bps=10.0, ter_diff=ter).loc[HO_START:]
    bh = backtest.run(pd.Series(1.0, index=idx), asset, cash, cost_bps=0.0, ter_diff=ter).loc[HO_START:]
    c = cash.loc[HO_START:]
    rows = {}
    for name, b in (("Buy & hold", bh), ("SMA200 filter", bt)):
        m = metrics.summary(b["ret"], b["position"])
        m["Sharpe"] = metrics.sharpe(b["ret"], c)
        m["Total return"] = float(b["equity"].iloc[-1] / b["equity"].iloc[0] * (1 + b["ret"].iloc[0]) - 1)
        rows[name] = m
    t = pd.DataFrame(rows).T
    t["Switches (count)"] = [0, int(bt["position"].diff().abs().gt(0).sum())]
    t.loc["SMA200 filter", "Worst 12m vs B&H"] = worst_12m_gap(bt["ret"], bh["ret"])
    bm, sm = rows["Buy & hold"], rows["SMA200 filter"]
    passed = (sm["Sharpe"] >= bm["Sharpe"] + 0.15) or (
        sm["Sharpe"] >= bm["Sharpe"] and abs(sm["MaxDD"]) <= 0.7 * abs(bm["MaxDD"]))
    val.log_trial("T0 SMA200", {"version": label, "cost_bps": 10, "ter": ter},
                  {k: round(v, 4) for k, v in sm.items()}, sample="HOLD-OUT")
    return t, passed, bt, bh


def main():
    bbg = io.load_bloomberg(ROOT / "data" / "raw" / "MXWO_Market_Breath_data.xlsx")
    close = bbg["close"]

    # causality check: signal up to the hold-out start is the same without hold-out data
    full = sig.trend_filter(close, 200).loc[: io.IN_SAMPLE_END]
    cut = sig.trend_filter(close.loc[: io.IN_SAMPLE_END], 200)
    assert full.equals(cut), "signal changed when hold-out data was appended"

    ntr_eur = io.fetch_msci_levels("NETR", currency="EUR")
    ntr_usd = io.fetch_msci_levels("NETR")
    estr = io.fetch_ecb_estr()
    tbill = io.fetch_fred("DTB3")

    t_eur, pass_eur, bt_eur, bh_eur = run_version(close, ntr_eur, estr, 0.002, "EUR IWDA proxy")
    t_usd, pass_usd, _, _ = run_version(close, ntr_usd, tbill, 0.0, "USD")
    t_eur.round(4).to_csv(OUT / f"{TAG}_holdout_eur.csv")
    t_usd.round(4).to_csv(OUT / f"{TAG}_holdout_usd.csv")

    sma = close.rolling(200).mean()
    pos = sig.trend_filter(close, 200)
    flips = pos.loc[HO_START:].diff().fillna(0)
    flip_dates = flips[flips != 0]
    last = close.index[-1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(bh_eur["equity"] / bh_eur["equity"].iloc[0], color="#888", label="Buy & hold (IWDA proxy, EUR)")
    ax[0].plot(bt_eur["equity"] / bt_eur["equity"].iloc[0], color="#2c3e50", label="SMA200 filter (EUR, ESTR cash)")
    ax[0].set_title("Hold-out 2023-11 to 2026-11, single pre-registered run")
    ax[0].legend(fontsize=8)
    w = close.loc[HO_START:]
    ax[1].plot(w, color="#222", lw=0.8, label="MSCI World price (USD)")
    ax[1].plot(sma.loc[HO_START:], color="#c0392b", lw=1, label="200-day SMA")
    ax[1].fill_between(w.index, w.min(), w.max(), where=pos.loc[HO_START:] == 0, color="#c0392b", alpha=0.1,
                       label="signal = cash")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_holdout.png", dpi=130)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("## causality check passed")
    print("\n## EUR (IWDA proxy, ESTR cash)\n", t_eur.round(4).to_string(), "\npass:", pass_eur)
    print("\n## USD (NTR USD, T-bill cash)\n", t_usd.round(4).to_string(), "\npass:", pass_usd)
    print("\n## signal changes in hold-out\n", flip_dates.map({1.0: "to LONG", -1.0: "to CASH"}).to_string())
    print(f"\n## current signal {last.date()}: close {close.iloc[-1]:.2f}, SMA200 {sma.iloc[-1]:.2f}, "
          f"distance {100 * (close.iloc[-1] / sma.iloc[-1] - 1):.2f}% -> {'LONG' if pos.iloc[-1] else 'CASH'}")
    print("ESTR range in hold-out:", estr.loc[HO_START:].min(), estr.loc[HO_START:].max(),
          "| NTR EUR last date:", ntr_eur.index[-1].date())


if __name__ == "__main__":
    main()
