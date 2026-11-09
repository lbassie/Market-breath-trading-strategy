# Phase 2: breadth as a filter on the SMA200 trend rule

Date: 2026-11-09. Scope: in-sample only, 2007-01 to 2023-10 (2006 is used as warm-up). The hold-out is still locked.
Data (all free): MSCI World NTR and price, USD, daily, from msci.com. 3m T-bill (FRED DTB3) as the cash yield. Breadth (R1/L1) from the Bloomberg file.
Costs: 10 bp per side (25 bp as a sensitivity test). Signal at the close of t, trade at the close of t+1. Sharpe = excess over the T-bill.
Reproduce: `python scripts/run_phase2.py`. Tables: `2026-11-09_defaults.csv`, `2026-11-09_grid.csv`. Trial log: 67 trials in total.

## Summary
1. **T2 is the only candidate.** T2 = go to cash only if the price is below the SMA200 *and* breadth is weak. It has Sharpe 0.58 vs 0.49 for T0 (SMA200) and 0.35 for buy-and-hold, and CAGR 6.8% vs 5.4%. It beats T0 in all three sub-periods, for all three SMA lengths (150/200/250), and at 25 bp costs.
2. **The edge is not proven.** The bootstrap 95% interval of the Sharpe difference vs T0 is −0.15 to +0.30 (p = 0.23). PBO is 0.51 (gate < 0.2). DSR is 0.954, just above the 0.95 gate. Max drawdown is −26% vs −17% for T0: T2 gets a higher return because it stays invested longer (77% vs 70% of days).
3. **Buying the washout does not work.** T1 (re-enter after an L1 extreme) lowers the Sharpe in every one of the 27 grid cells. It buys too early in falling markets (2008, March 2020). The same applies to T3. This agrees with the phase-1 event study.
4. **Free data works.** The MSCI price series from msci.com is identical to the Bloomberg close (return correlation 1.000, max level difference 0.006%). NTR adds on average 2.1%/yr over price.

## Pre-registered default rules (SMA200, L1 p95, hold 20 days), 10 bp
| Rule | CAGR | Vol | Sharpe | MaxDD | Calmar | Time in mkt | Switches/yr |
|---|---|---|---|---|---|---|---|
| Buy & hold NTR | 5.6% | 17.1% | 0.35 | −57.8% | 0.10 | 100% | 0 |
| T0 SMA200 | 5.4% | 9.5% | 0.49 | −17.4% | 0.31 | 70% | 6.9 |
| T1 washout re-entry | 5.5% | 14.7% | 0.37 | −42.1% | 0.13 | 84% | 6.5 |
| **T2 confirmed exit** | **6.8%** | 10.6% | **0.58** | −26.4% | 0.26 | 77% | 5.6 |
| T3 = T1 + T2 | 6.7% | 15.3% | 0.44 | −43.8% | 0.15 | 90% | 4.7 |

At 25 bp: T0 Sharpe 0.38, T2 0.50.

## Sub-period Sharpe
| | 2007–12 | 2013–19 | 2020–23 |
|---|---|---|---|
| Buy & hold | 0.08 | 0.89 | 0.30 |
| T0 SMA200 | 0.25 | 0.89 | 0.32 |
| T2 | **0.33** | **1.03** | **0.38** |

## Robustness (Sharpe over the grid)
T2 vs T0 by SMA length: 150: 0.54 vs 0.50 · 200: 0.58 vs 0.49 · 250: 0.57 vs 0.43. None of the 27 T1 cells and none of the 27 T3 cells beat T0 (heatmaps: `2026-11-09_heatmaps.png`).

## Gates (in-sample, the bar before the hold-out)
| Gate | T2 |
|---|---|
| 1. Sharpe ≥ B&H + 0.15 | Pass (0.58 vs 0.35) |
| 2. Sharpe > T0 | Pass |
| 3. DSR > 0.95 (67 trials) | Pass, marginal (0.954) |
| 4. PBO < 0.2 | **Fail (0.51)** |
| 5. Beats T0 in ≥2 of 3 sub-periods | Pass (3 of 3) |
| 6. ≤12 switches/yr | Pass (5.6) |
| Bootstrap p < 0.05 vs T0 | **Fail (0.23)** |

The PBO is computed over all 60 phase-2 trials. 54 of those are T1/T3 variants, which are close to each other. This makes the ranking noisy, but the gate was fixed in advance, so we report it as a fail.

## Verdict
On the pre-registered criteria, no rule passes all in-sample gates. Thus we do not open the hold-out. T2 is a plausible candidate: it is simple, has one breadth condition, and is consistent across sub-periods and SMA lengths. But 17 years with about 4 bear markets do not give enough statistical power to separate it from the plain SMA200 rule.

## Options for the next step (decision for Lars)
1. **Open the hold-out for T2 vs T0 only** (one run, pre-registered now). This accepts that the PBO gate failed. The hold-out has only 3 years and probably no bear market, so it will not give much power.
2. **Get more history before we decide:** rebuild breadth from free data for a longer period (e.g. US S&P 500 members back to the 1990s) and test T2 out of sample on that.
3. **Stop here:** keep the SMA200 rule as the practical result and use breadth only as a discretionary input.
