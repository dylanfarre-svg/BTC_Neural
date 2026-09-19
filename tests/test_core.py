"""Unit tests on synthetic data: labels, calibration, extremes detection, simulator, RSI."""

import numpy as np
import pandas as pd

from btc_neural.calibrate import apply_calibration, apply_trailing, fit_calibration
from btc_neural.data import Extreme
from btc_neural.features import confirmed_extremes, htf_rsi
from btc_neural.labels import cycle_label
from btc_neural.backtest.simulator import simulate


def synthetic_cycle(days=1600):
    idx = pd.date_range("2015-01-01", periods=days, freq="D")
    t = np.arange(days)
    # low at 0, top at 1000, low at 1400
    lp = np.where(t <= 1000, np.log(100) + t / 1000 * np.log(50), np.log(5000) - (t - 1000) / 400 * np.log(12.5))
    lp = np.where(t > 1400, np.log(400) + (t - 1400) / 1000, lp)
    return pd.Series(np.exp(lp), index=idx)


def test_cycle_label_extremes_and_cutoff():
    c = synthetic_cycle()
    ex = [Extreme("low", c.index[0], c.iloc[0]), Extreme("top", c.index[1000], c.iloc[1000]),
          Extreme("low", c.index[1400], c.iloc[1400])]
    y = cycle_label(c, ex, c.index[-1])
    assert y.iloc[0] == 1 and y.iloc[1000] == -1 and y.iloc[1400] == 1
    assert y.iloc[1500:].isna().all()  # after the last extreme: unlabeled
    # Cutoff before the last low is confirmed: the falling leg must be unlabeled
    y2 = cycle_label(c, ex, c.index[1450])
    assert y2.iloc[1001:].isna().all()
    assert y2.index[-1] == c.index[1450]


def test_calibration_budget():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=1460 * 4), index=pd.date_range("2010-01-01", periods=1460 * 4))
    cal = fit_calibration(s, {10: 10, 5: 100})
    st = apply_calibration(s, cal)
    assert abs((st == 10).sum() - 40) <= 2
    assert abs((st == -10).sum() - 40) <= 2
    assert ((st.abs() >= 5).sum() - 800) <= 4


def test_trailing_is_causal():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=3000), index=pd.date_range("2010-01-01", periods=3000))
    full = apply_trailing(s, {10: 10}, window=1095)
    part = apply_trailing(s.iloc[:2000], {10: 10}, window=1095)
    assert (full.iloc[:2000] == part).all()


def test_confirmed_extremes_are_causal():
    c = synthetic_cycle()
    full = confirmed_extremes(c)
    part = confirmed_extremes(c.iloc[:1200])
    pd.testing.assert_frame_equal(full.iloc[:1200], part)
    # the top at day 1000 is confirmed only once price halves (~day 1110)
    assert np.isnan(full["days_since_top1"].iloc[1050])
    assert full["days_since_top1"].iloc[1300] == 300


def test_htf_rsi_uses_partial_bar_only():
    c = synthetic_cycle()
    full = htf_rsi(c, "W-SUN")
    part = htf_rsi(c.iloc[:900], "W-SUN")
    assert np.allclose(full.iloc[:900].fillna(-1), part.fillna(-1))


def test_simulator_dca_and_tax():
    c = synthetic_cycle(1500)
    zero = pd.Series(0, index=c.index)
    fx = pd.Series(15.0, index=c.index)
    dca = simulate(c, zero, fx, c.index[0], c.index[-1], {"fee_rate": 0.0}, dca_only=True)
    months = len(pd.period_range(c.index[0], c.index[-1], freq="M"))
    assert dca.summary["contributed_usd"] == 1000 * months
    assert dca.summary["n_sells"] == 0 and dca.summary["tax_paid_usd"] == 0
    # Sell hard around the top: must produce taxable gains and keep the core floor
    s = pd.Series(0, index=c.index)
    s.iloc[980:1000] = -10
    strat = simulate(c, s, fx, c.index[0], c.index[-1], {"fee_rate": 0.0, "core_floor_fraction": 0.25})
    assert strat.summary["n_sells"] == 20
    assert strat.summary["tax_paid_usd"] + strat.summary["tax_accrued_usd"] > 0
    peak = strat.equity["btc"].max()
    assert strat.equity["btc"].iloc[1000] >= 0.25 * peak - 1e-9


def test_forecast_uses_only_earlier_cycles():
    """A hindcast must not change when later extremes are added to the list."""
    from btc_neural.backtest.walkforward import dataset
    from btc_neural.data import cycle_extremes
    from btc_neural.forecast import forecast_next
    import pytest
    daily, feats = dataset()
    if len(daily) < 3000:
        pytest.skip("database not populated")
    ex = cycle_extremes(daily["close"])
    asof = ex[6].date + pd.Timedelta(days=180)   # after the 2021 top, before the 2022 low
    a = forecast_next(daily, feats, ex[:7], asof=asof)
    b = forecast_next(daily, feats, ex, asof=asof)
    assert a.kind == "low" and a.date_mu == b.date_mu and a.logp_mu == b.logp_mu
    assert a.date_sd >= 30 and a.logp_sd >= 0.15
