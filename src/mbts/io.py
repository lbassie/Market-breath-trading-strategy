"""Data loaders. All loaders return a DataFrame with an ascending DatetimeIndex."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

BLOOMBERG_COLUMNS = {
    "MSCI World Index": "close",
    "14-day RSI": "rsi14",
    "Upper Band (MXWO)": "vbe_upper",
    "Lower Band (MXWO)": "vbe_lower",
    "Percent of members with RSI >70 (R1)": "pct_rsi_gt70",
    "Percent of members with RSI <30 (L1)": "pct_rsi_lt30",
    "Volume": "volume",
}

IN_SAMPLE_END = pd.Timestamp("2023-10-31")  # hold-out starts 2023-11-01 (locked)


def load_bloomberg(path: str | Path) -> pd.DataFrame:
    """Load the Bloomberg MXWO breadth export.

    The file has an empty first row; headers are on row 2. L1 is stored as a
    negative number by Bloomberg; we store it as a positive percentage.
    """
    df = pd.read_excel(path, header=1)
    df = df.rename(columns=BLOOMBERG_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[list(BLOOMBERG_COLUMNS.values())]
    df["pct_rsi_lt30"] = df["pct_rsi_lt30"].abs()
    return df


def split_in_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the in-sample part. The hold-out must not be used before step 7."""
    return df.loc[:IN_SAMPLE_END]


def load_msci_csv(path: str | Path) -> pd.Series:
    """Load an end-of-day index history exported from msci.com (Date, Level)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col, level_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[level_col].astype(str).str.replace(",", ""), errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=level_col,
    )
    return s.sort_index().dropna()


RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
MSCI_URL = "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph"


def parse_msci_json(payload: dict) -> pd.Series:
    levels = payload["indexes"]["INDEX_LEVELS"]
    s = pd.Series(
        [x["level_eod"] for x in levels],
        index=pd.to_datetime([str(x["calc_date"]) for x in levels], format="%Y%m%d"),
        name=payload.get("index_variant_type", "level"),
    )
    return s.sort_index()


def fetch_msci_levels(variant: str = "NETR", code: str = "990100", currency: str = "USD",
                      start: str = "20060101", end: str = "20261231", cache: bool = True) -> pd.Series:
    """Daily MSCI index levels from msci.com. variant: STRD (price), NETR or GRTR. 990100 = MSCI World."""
    path = RAW / f"msci_{code}_{variant}_{currency}.csv"
    if cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    r = requests.get(MSCI_URL, params={
        "currency_symbol": currency, "index_variant": variant, "start_date": start, "end_date": end,
        "data_frequency": "DAILY", "index_codes": code}, timeout=60)
    r.raise_for_status()
    s = parse_msci_json(r.json())
    RAW.mkdir(parents=True, exist_ok=True)
    s.to_csv(path)
    return s


def parse_fred_csv(text: str) -> pd.Series:
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=df.columns[1])
    return s.dropna()


def fetch_fred(series_id: str = "DTB3", cache: bool = True) -> pd.Series:
    """FRED series in percent per year (e.g. DTB3 = 3m T-bill)."""
    path = RAW / f"fred_{series_id}.csv"
    if cache and path.exists():
        return parse_fred_csv(path.read_text())
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=60)
    r.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return parse_fred_csv(r.text)


def fetch_ecb_estr(cache: bool = True) -> pd.Series:
    """Euro short-term rate (%STR), percent per year, from Oct 2019. EONIA history is not loaded."""
    path = RAW / "ecb_estr.csv"
    if not (cache and path.exists()):
        r = requests.get("https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT",
                         params={"format": "csvdata", "startPeriod": "2019-10-01"}, timeout=120)
        r.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    df = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE"])
    return pd.Series(df["OBS_VALUE"].values, index=pd.to_datetime(df["TIME_PERIOD"]), name="ESTR").sort_index()


def daily_cash_return(rate_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Cash return per row of `index`: the rate known at t-1, accrued over the calendar days to t."""
    rate = rate_pct.reindex(rate_pct.index.union(index)).ffill().reindex(index).shift(1)
    days = index.to_series().diff().dt.days
    return ((1 + rate / 100) ** (days / 365.0) - 1).fillna(0.0)


def load_eodhd(ticker: str, start: str = "2005-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Daily end-of-day prices from EODHD. The key is read from EODHD_API_KEY."""
    api_key = api_key or os.environ["EODHD_API_KEY"]
    r = requests.get(
        f"https://eodhd.com/api/eod/{ticker}",
        params={"api_token": api_key, "from": start, "fmt": "json"},
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()
