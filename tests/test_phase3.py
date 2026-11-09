import numpy as np
import pandas as pd

from mbts import indicators, io

SAMPLE = """This file was created using the CRSP database.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   0.56,  -0.07, -99.99
19260702,   0.29,   0.06, -999

  Average Equal Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   1.00,   2.00,   3.00
"""


def test_parse_french_sections_and_missing():
    vw = io.parse_french_section(SAMPLE, "Average Value Weighted Returns -- Daily")
    assert list(vw.columns) == ["Agric", "Food", "Soda"]
    assert len(vw) == 2 and np.isclose(vw.iloc[0, 0], 0.0056)
    assert vw["Soda"].isna().all()
    ew = io.parse_french_section(SAMPLE, "Average Equal Weighted Returns -- Daily")
    assert len(ew) == 1 and np.isclose(ew.iloc[0, 2], 0.03)


def test_breadth_counts_only_members_with_data():
    idx = pd.bdate_range("2000-01-01", periods=60)
    up = pd.Series(np.linspace(100, 200, 60), index=idx)
    prices = pd.DataFrame({"a": up, "b": up, "c": np.nan}, index=idx)
    br = indicators.breadth_from_prices(prices)
    assert br["n_members"].iloc[-1] == 2
    assert br["pct_rsi_gt70"].iloc[-1] == 100.0
