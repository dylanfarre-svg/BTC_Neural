"""The key test: every feature at date t must be identical whether or not the future exists.

We rebuild features on data truncated at several cutoffs and compare the last row with the
full-history computation. Any centred window, full-sample normalisation, or hindsight
cycle date would make these differ.
"""

import numpy as np
import pandas as pd
import pytest

from btc_neural import db
from btc_neural.data import daily_prices
from btc_neural.features import build_features

CUTOFFS = ["2012-03-15", "2014-01-10", "2017-12-17", "2019-06-26", "2021-11-08", "2024-03-13"]


@pytest.fixture(scope="module")
def data():
    daily = daily_prices()
    if len(daily) < 3000:
        pytest.skip("database not populated; run `btcn ingest` first")
    series = db.read_series()
    lags = db.series_lags()
    h4 = db.read_candles("4h")
    full = build_features(daily, series, lags, h4)
    return daily, series, lags, h4, full


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_are_causal(data, cutoff):
    daily, series, lags, h4, full = data
    t = pd.Timestamp(cutoff)
    trunc = build_features(
        daily.loc[:t],
        series.loc[:t],
        lags,
        h4.loc[: t + pd.Timedelta(hours=23, minutes=59)],
    )
    a = full.loc[t]
    b = trunc.loc[t]
    both_nan = a.isna() & b.isna()
    close = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9) | both_nan
    bad = a.index[~close]
    assert len(bad) == 0, f"features differ at {cutoff} (lookahead!): " + ", ".join(
        f"{c}: full={a[c]} trunc={b[c]}" for c in bad[:10])
