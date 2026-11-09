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
