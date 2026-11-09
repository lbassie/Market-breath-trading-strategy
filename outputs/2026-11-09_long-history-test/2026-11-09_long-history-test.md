# Phase 3: long-history test of the T2 rule (US, 1928–2005)

Date: 2026-11-09. Data: Ken French daily data (CRSP). Members = the 49 value-weighted industry portfolios. Market = Mkt-RF + RF. Cash = RF (1-month T-bill).
Rules are **frozen** from phase 2 (no new tuning). Out-of-sample period 1928–2005. Data after 2023-10-31 is cut off, so the hold-out stays locked.
Costs 10 bp per side (25 bp sensitivity), t+1 execution. Reproduce: `python scripts/run_phase3.py`.

## Verdict
**T2 fails the out-of-sample test and is rejected.** Over 1928–2005, T2 has Sharpe 0.55 vs 0.68 for the plain SMA200 rule. It is worse in 6 of 8 decades and at all three SMA lengths. It only helps in the 1990s and 2000s, which overlap with the period where we found it. This is the typical sign of a rule that fits one period. The practical result of the project is the **plain SMA200 trend filter**. Breadth adds nothing to it.

## Calibration: is the French breadth a good proxy? (2006–2023, thresholds fixed in advance)
| Check | Result | Threshold | |
|---|---|---|---|
| R1 level correlation, French vs Bloomberg MXWO | 0.80 | ≥ 0.70 | pass |
| L1 level correlation | 0.86 | – | |
| T2 "weak breadth" flag agrees | 84% of days | ≥ 75% | pass |
| Annual return correlation, US market vs MSCI World NTR (2007–22) | 0.98 | – | |

There are 40 industries with data in the 1920s–30s, 43 in the 1940s–60s and 48–49 from the 1970s. The proxy is good enough to test the rule.

## Out-of-sample 1928–2005, 10 bp
| Rule | CAGR | Vol | Sharpe | MaxDD | Time in mkt | Switches/yr |
|---|---|---|---|---|---|---|
| Buy & hold US market | 9.2% | 16.6% | 0.41 | −84% | 100% | 0 |
| **T0 SMA200** | **10.9%** | 11.1% | **0.68** | **−34%** | 71% | 5.3 |
| T2 confirmed exit | 10.3% | 13.5% | 0.55 | −74% | 82% | 5.4 |
| T1 washout re-entry | 9.3% | 14.8% | 0.45 | −71% | 84% | 4.9 |
| T3 = T1 + T2 | 9.0% | 15.9% | 0.41 | −84% | 93% | 3.9 |

SMA150: T0 0.69 vs T2 0.53. SMA250: T0 0.67 vs T2 0.56. At 25 bp: T0 0.61 vs T2 0.49.
T2 has a −74% drawdown because in 1929–32 the industries did not all fall at the same time. Breadth did not show "weak", so T2 stayed invested while the price was far below the SMA200.

## Sharpe by decade
| | 1930s | 1940s | 1950s | 1960s | 1970s | 1980s | 1990s | 2000s |
|---|---|---|---|---|---|---|---|---|
| T0 SMA200 | 0.43 | 0.97 | 1.50 | 0.93 | 0.44 | 0.81 | 0.89 | −0.23 |
| T2 | 0.25 | 0.81 | 1.43 | 0.92 | 0.27 | 0.79 | 0.91 | −0.08 |
| T2 − T0 | −0.18 | −0.16 | −0.06 | −0.01 | −0.17 | −0.02 | +0.03 | +0.16 |

Paired block bootstrap (63-day blocks), Sharpe T2 − T0: −0.13, 95% CI −0.31 to +0.02.

## Pre-registered gates
| Gate | Result |
|---|---|
| a. T2 Sharpe > T0, bootstrap p < 0.05 | **Fail** (T2 is lower) |
| b. T2 > T0 in ≥ 5 of 8 decades | **Fail** (2 of 8) |
| c. Same direction at SMA150 and SMA250 | **Fail** |
| d. ≤ 12 switches/yr | Pass |

## The overlap 2006–2023 (French data, for information only)
On US data, T2 is a little better than T0 here too (Sharpe 0.51 vs 0.47), in line with the MSCI World result. This shows that the in-sample edge is real *for that period*, but it does not hold over 80 years.

## What the project shows
1. The Bloomberg VBE bands use future data. They must not be used for backtests.
2. Breadth extremes (washouts, divergences, capitulation) do not predict MSCI World returns on their own (phase 1).
3. Breadth as a filter on the trend rule looked good in 2007–2023 but fails on 1928–2005.
4. The SMA200 trend filter is robust in both samples. It gives a lower drawdown (−34% vs −84% US, −17% vs −58% MSCI World) and a higher Sharpe, with about 5–7 switches per year.

Possible next step (not started): test the plain SMA200 rule once on the locked hold-out (2023-11 to 2026-11), as an IWDA long/€STR cash implementation, and write it up as the final strategy note.
