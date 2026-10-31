"""Trading-signal tests: barrier labels on known paths, causality, purging, simulator, append-only log."""

import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from btc_neural import db
from btc_neural.trade.folds import blocks, train_mask
from btc_neural.trade.horizons import horizon
from btc_neural.trade.labels import barrier_outcomes, uniqueness_weights


def flat_market(days=60, price=100.0):
    """Flat daily prices (so the target width is the 5% floor) on a 4h grid we can edit."""
    d_idx = pd.date_range("2020-01-01", periods=days, freq="D")
    daily = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0}, index=d_idx)
    h_idx = pd.date_range("2020-01-01", periods=days * 6, freq="4h")
    h4 = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0}, index=h_idx)
    return daily, h4


def bar(h4, ts, **kw):
    for k, v in kw.items():
        h4.loc[pd.Timestamp(ts), k] = v


HZ = replace(horizon("swing"), floor=0.05, stop_ratio=0.6, hold_max_days=28)


def test_long_target_and_short_stop():
    daily, h4 = flat_market()
    # decision at the close of 2020-01-20 (after 15+ days of vol history); path starts 01-21 00:00
    bar(h4, "2020-01-22 08:00", high=103.5)          # short stop (103) hit; long untouched
    bar(h4, "2020-01-23 12:00", high=105.5)          # long target (105) hit later
    o = barrier_outcomes(daily, h4, HZ)
    lo, sh = o["long"].loc["2020-01-20"], o["short"].loc["2020-01-20"]
    assert lo["tw"] == pytest.approx(0.05)
    assert lo["reason"] == "target" and lo["exit_price"] == pytest.approx(105.0) and lo["gross"] == pytest.approx(0.05)
    assert lo["exit_time"] == pd.Timestamp("2020-01-23 16:00")
    assert sh["reason"] == "stop" and sh["exit_price"] == pytest.approx(103.0) and sh["gross"] == pytest.approx(-0.03)
    assert sh["R"] == pytest.approx(-0.6)


def test_same_bar_assumes_stop_and_gaps_fill_at_open():
    daily, h4 = flat_market()
    bar(h4, "2020-01-22 00:00", high=106, low=96)    # both long barriers inside one bar
    o = barrier_outcomes(daily, h4, HZ)
    assert o["long"].loc["2020-01-20", "reason"] == "stop"
    daily, h4 = flat_market()
    bar(h4, "2020-01-22 00:00", open=94, high=95, low=93, close=94)   # gaps through the long stop (97)
    o = barrier_outcomes(daily, h4, HZ)
    assert o["long"].loc["2020-01-20", "exit_price"] == pytest.approx(94.0)


def test_deadline_and_unresolved_tail():
    daily, h4 = flat_market(days=70)
    o = barrier_outcomes(daily, h4, HZ)["long"]
    assert o.loc["2020-01-20", "reason"] == "deadline" and o.loc["2020-01-20", "hold_days"] == pytest.approx(28.0)
    assert np.isnan(o.loc["2020-03-05", "R"])       # window runs past the data, nothing touched yet


def test_uniqueness_weights_are_inverse_overlap():
    daily, h4 = flat_market(days=80)
    o = barrier_outcomes(daily, h4, HZ)["long"]
    w = uniqueness_weights(o).dropna()
    assert ((w > 0) & (w <= 1)).all()
    assert w.iloc[len(w) // 2] < 0.1   # 28-day windows overlap ~28-fold in the middle


def test_blocks_are_purged():
    hz = horizon("swing")
    bl = blocks(hz, hz.warmup_start, pd.Timestamp("2020-01-01"))
    assert bl and all(b.train_end == b.start - pd.Timedelta(days=hz.purge_days) for b in bl)
    dates = pd.date_range("2012-01-01", "2020-01-01")
    for b in bl[::10]:
        m = train_mask(dates, b, hz)
        last_train = dates[m.to_numpy()].max()
        assert last_train + pd.Timedelta(days=hz.hold_max_days) < b.start   # its label ends before the block


@pytest.fixture(scope="module")
def real():
    from btc_neural.data import daily_prices
    daily = daily_prices()
    if len(daily) < 3000:
        pytest.skip("database not populated")
    return daily, db.read_series(), db.series_lags(), db.read_candles("4h")


@pytest.mark.parametrize("cutoff", ["2014-06-15", "2018-03-01", "2021-11-08", "2024-03-13"])
def test_trade_features_are_causal(real, cutoff):
    from btc_neural.trade.features import build_trade_features
    daily, series, lags, h4 = real
    t = pd.Timestamp(cutoff)
    full = build_trade_features(daily, series, lags, h4)
    trunc = build_trade_features(daily.loc[:t], series.loc[:t], lags, h4.loc[: t + pd.Timedelta(hours=23, minutes=59)])
    a, b = full.loc[t], trunc.loc[t]
    ok = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9) | (a.isna() & b.isna())
    assert ok.all(), f"lookahead at {cutoff}: " + ", ".join(f"{c}: {a[c]} vs {b[c]}" for c in a.index[~ok][:8])


def test_simulator_equity_matches_trades(real):
    from btc_neural.trade.simulator import after_tax_equity, build_market, simulate
    daily, _, _, h4 = real
    hz = horizon("swing")
    m = build_market(daily, h4, hz)
    n = len(m.days)
    rng = np.random.default_rng(1)
    sig = {"long": rng.random(n) < 0.2, "short": rng.random(n) < 0.2}
    pref = {"long": rng.random(n), "short": rng.random(n)}
    res = simulate(m, sig, pref, "2019-01-01", "2024-12-31")
    assert len(res.trades) > 20
    assert (1 + res.daily).prod() == pytest.approx((1 + res.trades.net_ret).prod(), rel=1e-9)
    # no trade starts while another is open, and the cooldown holds
    gaps = (res.trades.entry_date.iloc[1:].to_numpy() - res.trades.exit_day.iloc[:-1].to_numpy()) / np.timedelta64(1, "D")
    assert (gaps >= hz.cooldown_days).all()
    pre = (1 + res.daily).cumprod().iloc[-1]
    cap, rev = after_tax_equity(res, hz.capital_rate).iloc[-1], after_tax_equity(res, hz.revenue_rate).iloc[-1]
    assert (cap <= pre and rev <= cap) if pre > 1 else (cap == pytest.approx(pre) and rev == pytest.approx(pre))


def test_trade_signals_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "path", lambda key: tmp_path / "t.db")
    db.execute("INSERT INTO trade_signals (ts, horizon, action) VALUES ('2026-01-01', 'swing', 'flat')")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE trade_signals SET action='long'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("DELETE FROM trade_signals")


def test_trade_alerts_never_reach_ntfy(monkeypatch):
    """The user's instruction: no swing signals on ntfy. Enforced in dispatch itself."""
    from btc_neural import notify
    sent = []
    monkeypatch.setattr(notify, "send_ntfy", lambda a: sent.append(("ntfy", a.kind)))
    monkeypatch.setattr(notify, "send_macos", lambda a: sent.append(("macos", a.kind)))
    monkeypatch.setattr(notify, "already_sent", lambda d, k: False)
    monkeypatch.setattr(notify.db, "log_notification", lambda *a, **k: None)
    alerts = [notify.Alert(k, "t", "m") for k in ("trade_enter", "trade_stop", "trade", "strong")]
    out = notify.dispatch("2026-01-01", alerts, force=True)
    assert out == ["strong"]
    assert all(kind == "strong" for _, kind in sent)
