"""Phase 1: data audit + in-sample event study + baseline backtests.

Usage: python scripts/run_phase1.py [path/to/MXWO_Market_Breath_data.xlsx]
Writes tables and charts to outputs/2026-11-09_breadth-study/.
Uses ONLY in-sample data (<= 2023-10-31) for the event study and baselines.
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
from mbts import backtest, indicators as ind, io, metrics, signals as sig, validation as val  # noqa: E402

OUT = ROOT / "outputs" / "2026-11-09_breadth-study"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160)


def audit(df: pd.DataFrame) -> dict:
    r = df["close"].pct_change()
    gaps = df.index.to_series().diff().dt.days
    res = {
        "rows": len(df), "start": df.index.min().date(), "end": df.index.max().date(),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "gaps_gt_4_days": gaps[gaps > 4].to_dict(),
        "missing": df.isna().sum().to_dict(),
        "abs_ret_gt_7pct": r[r.abs() > 0.07].round(4).to_dict(),
        "volume_zero": int((df["volume"] == 0).sum()),
        "describe": df.describe().T.round(2),
    }
    # RSI check
    ours = ind.rsi_wilder(df["close"])
    m = ours.iloc[100:]
    res["rsi_corr"] = m.corr(df["rsi14"].iloc[100:])
    res["rsi_mae"] = (m - df["rsi14"].iloc[100:]).abs().mean()
    # VBE lookahead test
    mid = (df["vbe_upper"] + df["vbe_lower"]) / 2
    trailing = df["rsi14"].rolling(21).mean()
    centered = trailing.shift(-10)
    res["vbe_mid_corr_rsi"] = mid.corr(df["rsi14"])
    res["vbe_mid_corr_close"] = mid.corr(df["close"])
    res["vbe_mid_corr_trailing_sma21"] = mid.corr(trailing)
    res["vbe_mid_corr_centered_sma21"] = mid.corr(centered)
    res["vbe_mid_mae_trailing"] = (mid - trailing).abs().mean()
    res["vbe_mid_mae_centered"] = (mid - centered).abs().mean()
    res["vbe_lead_lag"] = pd.Series({k: mid.corr(df["rsi14"].shift(k)) for k in range(-10, 11)})
    # chart
    fig, ax = plt.subplots(figsize=(10, 4))
    w = slice("2022-01-01", "2022-12-31")
    ax.plot(df.loc[w, "rsi14"], lw=0.8, label="RSI 14", color="#444")
    ax.plot(mid.loc[w], lw=1.2, label="Bloomberg VBE mid", color="#c0392b")
    ax.plot(centered.loc[w], lw=1, ls="--", label="Centered SMA21 of RSI (uses future)", color="#2980b9")
    ax.plot(trailing.loc[w], lw=1, ls=":", label="Trailing SMA21 of RSI (causal)", color="#27ae60")
    ax.set_title("VBE mid tracks a centered (look-ahead) average, not a trailing one - 2022")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "2026-11-09_vbe-lookahead.png", dpi=130)
    plt.close(fig)
    return res


def event_studies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ins = io.split_in_sample(df)
    thr = {
        "L1_p95": ins["pct_rsi_lt30"].quantile(0.95),
        "L1_p90": ins["pct_rsi_lt30"].quantile(0.90),
        "R1_p90": ins["pct_rsi_gt70"].quantile(0.90),
    }
    env = ind.causal_envelope(ins["rsi14"], 21, 21, 2.0)
    raw = {
        "H1 washout (L1>=p95)": sig.h1_washout(ins, thr["L1_p95"]),
        "H2 RSI < causal lower env": sig.h2_rsi_below_band(ins, env["lower"]),
        "H2b RSI < Bloomberg VBE lower (LOOK-AHEAD)": sig.h2_rsi_below_band(ins, ins["vbe_lower"]),
        "H3 divergence (risk-off)": sig.h3_divergence(ins),
        "H4 capitulation (vol z>2 & L1>=p90)": sig.h4_capitulation(ins, thr["L1_p90"]),
        "H5 thrust (L1>=p95 then R1>=p90)": sig.h5_thrust(ins, thr["L1_p95"], thr["R1_p90"]),
    }
    rows = []
    for name, ev in raw.items():
        ev = sig.decluster(ev.fillna(False), cooldown=20)
        es = val.event_study(ins["close"], ev)
        es.insert(0, "hypothesis", name)
        rows.append(es.reset_index())
        val.log_trial(name, {"thresholds": {k: round(v, 2) for k, v in thr.items()}, "cooldown": 20},
                      {f"excess{h}": round(float(es.loc[h, 'excess']), 4) for h in es.index if "excess" in es},
                      sample="in-sample event study")
    return pd.concat(rows, ignore_index=True), thr


def baselines(df: pd.DataFrame) -> pd.DataFrame:
    ins = io.split_in_sample(df)
    r = ins["close"].pct_change().fillna(0.0)
    out = {}
    bh = backtest.run(pd.Series(1.0, index=r.index), r, cost_bps=0)
    out["Buy & hold (price)"] = metrics.summary(bh["ret"], bh["position"])
    for n in (200,):
        pos = sig.trend_filter(ins["close"], n)
        bt = backtest.run(pos, r, cost_bps=10)
        out[f"SMA{n} trend filter, 10bp, cash=0%"] = metrics.summary(bt["ret"], bt["position"])
        val.log_trial(f"SMA{n} baseline", {"n": n, "cost_bps": 10}, out[f"SMA{n} trend filter, 10bp, cash=0%"])
    return pd.DataFrame(out).T


def breadth_chart(df: pd.DataFrame) -> None:
    ins = io.split_in_sample(df)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].semilogy(ins["close"], color="#222", lw=0.8)
    ax[0].set_title("MSCI World (price) and breadth, in-sample 2006-2023")
    ax[1].plot(ins["pct_rsi_gt70"], color="#27ae60", lw=0.5, label="% members RSI>70")
    ax[1].plot(-ins["pct_rsi_lt30"], color="#c0392b", lw=0.5, label="-% members RSI<30")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-11-09_breadth-overview.png", dpi=130)
    plt.close(fig)


def main(path: str) -> None:
    df = io.load_bloomberg(path)
    a = audit(df)
    es, thr = event_studies(df)
    bl = baselines(df)
    breadth_chart(df)
    es.round(4).to_csv(OUT / "2026-11-09_event-study.csv", index=False)
    bl.round(4).to_csv(OUT / "2026-11-09_baselines.csv")
    for k, v in a.items():
        print(f"\n## {k}\n{v}")
    print("\n## thresholds", {k: round(v, 2) for k, v in thr.items()})
    print("\n## event study\n", es.round(4).to_string())
    print("\n## baselines\n", bl.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw" / "MXWO_Market_Breath_data.xlsx"))
