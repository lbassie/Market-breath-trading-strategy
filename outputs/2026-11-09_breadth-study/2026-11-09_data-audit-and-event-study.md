# Phase 1: data audit and in-sample event study

Date: 2026-11-09. Data: Bloomberg MXWO export, 2006-01-23 to 2026-11-06 (5,425 rows, price index).
Scope: in-sample only (to 2023-10-31). The hold-out (2023-11-01 to 2026-11-06) is not used.
Reproduce: `python scripts/run_phase1.py`. Tables: `2026-11-09_event-study.csv`, `2026-11-09_baselines.csv`. Trial log: `outputs/trial_log.csv` (7 trials so far).

## Summary
1. **The Bloomberg VBE bands look ahead. Do not use them in a backtest.** The band midpoint correlates 0.98 with a *centered* 21-day average of the RSI (mean absolute error 1.7 RSI points). With a *trailing* 21-day average, the correlation is 0.79 (error 5.4). The lead/lag correlation of the band midpoint with the RSI is symmetric around lag 0. A causal average must lag the RSI. This agrees with the Bloomberg description ("centered moving average envelopes"). The last values in the series are probably estimates that change when new data arrives (repaint).
2. **The look-ahead makes a false edge.** "RSI below the Bloomberg lower band" gives +1.1% excess return over 5 days (Newey-West t = 5.0, 75% hit rate). The same rule with a causal envelope gives −0.2% (t = −0.7). The bands alone create all of the apparent edge.
3. **No causal breadth signal has a significant edge in the in-sample data.** All bootstrap 95% intervals include zero and all |t| < 1.4 (table below). The best result is H4 capitulation (+0.9% over 5 days, 22 events, t = 1.3). This is not significant and the sample is small.
4. **The benchmark to beat is strong.** A 200-day SMA filter (10 bp per side, 0% cash yield) has the same CAGR as buy-and-hold (4.3% price return) with Sharpe 0.49 vs 0.33 and max drawdown −20% vs −59%.

## Data audit
| Check | Result |
|---|---|
| Duplicate dates / gaps > 4 days | 0 / 0 |
| Missing values | VBE bands 42 (warm-up, 63 − 21), volume 6 |
| Our Wilder RSI(14) vs Bloomberg | corr 0.99999997, MAE 0.002. The rebuild is exact. |
| L1 sign | Stored as negative. The loader converts it to a positive %. |
| Daily moves > 7% | 7 days (Oct–Dec 2008, Mar 2020). All are real market moves. |
| Index type | Price return. P&L in later phases must use net total return (msci.com) or IWDA. |
| Band scale | VBE is on the RSI series (corr with RSI 0.90, with index level 0.12). |

## Event study (in-sample, events declustered with a 20-day cooldown)
Thresholds are in-sample percentiles, fixed before the test: L1 p95 = 14.6%, L1 p90 = 8.6%, R1 p90 = 13.5%.

| Hypothesis | Events | Excess 5d | Excess 20d | Excess 60d | NW t (5d / 20d / 60d) |
|---|---|---|---|---|---|
| H1 washout (L1 ≥ p95) | 39 | +0.42% | +0.14% | −0.19% | 0.7 / 0.1 / −0.2 |
| H2 RSI < causal lower envelope | 89 | −0.16% | −0.09% | +0.33% | −0.7 / −0.2 / 0.6 |
| H2b RSI < Bloomberg VBE lower (**look-ahead**) | 118 | +1.06% | +1.12% | +1.20% | 5.0 / 3.1 / 2.7 |
| H3 divergence (risk-off) | 44 | +0.03% | −0.05% | −1.02% | 0.2 / −0.1 / −1.0 |
| H4 capitulation (vol z > 2 & L1 ≥ p90) | 22 | +0.89% | +0.99% | −1.38% | 1.3 / 0.8 / −0.8 |
| H5 thrust (L1 ≥ p95, then R1 ≥ p90 within 10d) | 0 | – | – | – | – |

H5 had no events with the thresholds fixed in advance. We did not loosen the thresholds after we saw the result, because that would be data snooping.

## Baselines (in-sample, price return, cash = 0%)
| | CAGR | Vol | Sharpe | MaxDD | Calmar | Time in mkt | Switches/yr |
|---|---|---|---|---|---|---|---|
| Buy & hold | 4.3% | 16.8% | 0.33 | −59.1% | 0.07 | 100% | – |
| SMA200 filter, 10 bp | 4.3% | 9.5% | 0.49 | −19.6% | 0.22 | 72% | 6.6 |

## Verdict and next steps
On this data, the breadth fields alone do not show an edge that we can trade. The trend filter is the bar to beat. Proposed next steps:
1. Remove the Bloomberg VBE columns from all signal work. Use only causal envelopes.
2. Test breadth as a **conditioning variable on the trend filter**, not as a standalone signal. Example: stay long under the SMA200 when L1 shows a washout. Fewer than 3 parameters, logged as new trials.
3. Get net total return and cash data (msci.com NTR, ECB €STR), so that the baselines are realistic. With a cash yield, the trend filter gets better.
4. Rebuild R1/L1 from EODHD constituents (needs subscription and `EODHD_API_KEY`).
5. Keep the hold-out locked until a rule passes the in-sample validation (walk-forward, DSR, PBO).
