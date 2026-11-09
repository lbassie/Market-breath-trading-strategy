# Final strategy note: MSCI World trend filter (SMA200)

Date: 2026-11-09. Author: research repo `market-breath-trading-strategy` (phases 1–4).

## Verdict
The MSCI World breadth data (RSI bands, % of members with RSI > 70 / < 30, volume) gives **no robust timing edge**. The only rule that survives all tests is the plain **200-day moving-average filter** on MSCI World. It does not beat buy-and-hold on return. Its value is risk control: a much smaller drawdown for a similar or higher Sharpe ratio over long periods. In bull markets it lags, as in the 2023–26 hold-out.

## Rules
1. Each trading day after the US close, compare the MSCI World price index (USD, msci.com end-of-day) with its 200-day simple moving average.
2. Close above the SMA200 → hold MSCI World (IWDA). Close below → hold cash (EUR money market, €STR).
3. Trade at the next EU close (t+1). There are no other parameters.

## Evidence
| Sample | Data | B&H Sharpe | SMA200 Sharpe | B&H MaxDD | SMA200 MaxDD | SMA200 CAGR vs B&H |
|---|---|---|---|---|---|---|
| MSCI World 2007–2023 (in-sample) | NTR USD, T-bill | 0.35 | 0.49 | −58% | −17% | 5.4% vs 5.6% |
| US market 1928–2005 (out of sample) | CRSP (French), T-bill | 0.41 | 0.68 | −84% | −34% | 10.9% vs 9.2% |
| **Hold-out 2023-11 → 2026-11, EUR** | NTR EUR − 0.20% TER, €STR | 1.16 | 1.17 | −20.3% | −14.2% | 15.7% vs 18.1% |
| Hold-out, USD | NTR USD, T-bill | 1.21 | 1.01 | −16.6% | −11.1% | 15.3% vs 20.3% |

All results are net of 10 bp per side and include the cash yield.

**Hold-out result (single, pre-registered run).** EUR version: **pass, marginal.** The Sharpe is equal and the max drawdown is 30.4% smaller (gate: ≥ 30%). USD version: **fail.** The Sharpe is lower and the drawdown is 33% smaller. This is what we expected for a bull market. Over 3 years the filter gave up 10 (EUR) to 22 (USD) percentage points of total return. The worst 12-month gap vs B&H was −14.5% (EUR) and −18.8% (USD). There were 15 switches, 8 of them in March–May 2025 (whipsaw around the SMA).

## What was rejected and why
- **Bloomberg VBE bands:** they are calculated from a centered average, so the historical values use future data. They give a false edge (t = 5.0) that disappears with a causal band (phase 1).
- **Breadth extremes on their own** (washout, divergence, capitulation, thrust): no significant forward returns (phase 1).
- **Washout re-entry on the trend rule (T1/T3):** lower Sharpe in every parameter set. It buys too early in falling markets (phase 2).
- **Breadth-confirmed exit (T2):** better in 2007–2023 (Sharpe 0.58 vs 0.49), but worse in 6 of 8 decades in 1928–2005 and −74% drawdown in 1929–32 (phase 3).

## Implementation
- **Data:** msci.com end-of-day (free, identical to the Bloomberg close). Run `scripts/run_phase4_holdout.py` or a daily version of `signals.trend_filter`.
- **Instruments:** IWDA (iShares Core MSCI World, TER 0.20%) and a EUR overnight-rate money-market ETF, e.g. XEON [VERIFY ticker, TER and that it tracks €STR].
- **Timing:** the MSCI World close is after the EU close. Check the signal in the evening and trade at the next EU close.
- **Costs:** about 5–7 switches per year on average, with clusters when the price is near the SMA. A spread + commission of 10 bp per side is assumed [VERIFY with your broker].
- **Tax and administration:** depends on your own setup (box 3, BV or fund) [VERIFY]. Frequent switches can have an effect.

## Risks
- **Whipsaw:** when the price is near the SMA, the signal can flip many times in a few weeks (March 2025: 8 switches).
- **Lag in V-shaped recoveries:** in 2020 and 2025 the filter re-entered after a large part of the rebound.
- **Long underperformance in bull markets:** 12-month gaps of −15% to −19% vs B&H in the hold-out. This is the price of the drawdown protection, and it needs discipline.
- **Regime change:** the rule has worked over 95 years, but there is no guarantee for the future.

## Current signal
On 2026-11-06 the MSCI World close was 4,739.93 and the SMA200 was 4,741.71 (−0.04%). The signal is **CASH**, just below the line. This is a typical whipsaw situation.

## Monitoring
- Track live results against B&H every month.
- Review triggers: live drawdown > 20%, or underperformance vs B&H > 20% over 12 months (larger than the worst seen in the hold-out).
- Possible later test, not part of this note: a band around the SMA (e.g. ±1–2%) to cut whipsaw. This is a new parameter, so it needs its own out-of-sample test on the 1928–2005 data before use.
