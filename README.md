# Market-breadth trading strategy (MSCI World, long/cash)

This project tests if MSCI World breadth data (index RSI, RSI envelopes, % of members with RSI > 70 / < 30, volume) gives a timing edge that survives costs. The strategy holds a long position in an MSCI World ETF or holds cash.

**Status:** complete (phases 1–4). Verdict: breadth adds no robust edge. The plain SMA200 trend filter is the practical result. See the **final strategy note**: `outputs/2026-11-09_final-strategy-note/2026-11-09_final-strategy-note.md`.
- Phase 1 (data audit, event study): `outputs/2026-11-09_breadth-study/2026-11-09_data-audit-and-event-study.md`
- Phase 2 (breadth as a filter on SMA200, free data): `outputs/2026-11-09_trend-breadth-study/2026-11-09_trend-breadth-study.md`
- Phase 3 (long US history 1928–2005, Ken French 49 industries): `outputs/2026-11-09_long-history-test/2026-11-09_long-history-test.md`

## Setup
```bash
pip install -e ".[dev]"
# put the Bloomberg export at data/raw/MXWO_Market_Breath_data.xlsx (gitignored)
pytest -q
python scripts/run_phase1.py
python scripts/run_phase2.py   # downloads MSCI NTR/price (msci.com) and T-bill (FRED) into data/raw
python scripts/run_phase3.py   # downloads Ken French daily industry and factor data
python scripts/run_phase4_holdout.py   # the single hold-out run (2023-11 onward); do not rerun to tune
```

## Layout
- `src/mbts/io.py`: loaders for Bloomberg xlsx, msci.com (STRD/NETR/GRTR), FRED, ECB €STR, Ken French data library and the EODHD API (`EODHD_API_KEY`)
- `src/mbts/indicators.py`: Wilder RSI, causal envelope, breadth from constituent prices
- `src/mbts/signals.py`: hypotheses H1–H5 and the SMA trend benchmark
- `src/mbts/backtest.py`: long/cash engine with t+1 execution, costs, cash yield and TER
- `src/mbts/metrics.py`: Sharpe, Sortino, MDD, Calmar, alpha t-stat, Deflated Sharpe, PBO (CSCV)
- `src/mbts/validation.py`: event study, block bootstrap, trial log

## Research rules
- In-sample: 2006-01 to 2023-10. The hold-out (2023-11 onward) was used once, on 2026-11-09, for the SMA200 rule only.
- Every variant we try goes into `outputs/trial_log.csv`. The Deflated Sharpe Ratio uses this count.
- Do not use the Bloomberg VBE band columns. They are calculated from a centered average and therefore use future data.
