"""Short-horizon features for the trading signals.

Same contract as `btc_neural.features`: every value at date t uses only information available at
the daily close of t (tests/test_trade.py checks this by truncation). Kept in their own registry so
the cycle model's feature frame and feature sets are untouched.

Publication timing. Coin Metrics flows and Fear & Greed are shifted one day: the daily job runs just
after midnight UTC, before either has reliably published the day just closed. Funding (sum of the
day's settlements), the Bybit open-interest snapshot and DVOL's daily close are known at the close.
Macro series use their configured publication lags, as the cycle features do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import rolling_pct, rsi, sma

TRADE_GROUPS: dict[str, str] = {}

# Earliest date each group's inputs exist. A feature set is only *tested* from
# max(group starts) + one year, so no test block is scored by a model fed years of NaN.
GROUP_START = {
    "fast_price": "2010-09-01", "fast_vol": "2010-09-01", "fast_struct": "2010-09-01",
    "fast_volume": "2011-09-01", "fast_intraday": "2011-09-01", "fast_flow": "2011-06-01",
    "fast_macro": "2010-09-01", "cycle_context": "2012-11-28", "fast_sentiment": "2018-02-01",
    "funding": "2018-11-15", "open_interest": "2020-08-05", "implied_vol": "2021-03-24",
}

TRADE_FEATURE_SETS: dict[str, list[str]] = {
    # valid from 2015 (the primary population)
    "fast_core": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday"],
    "fast_flow": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "fast_flow"],
    "fast_macro": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "fast_macro"],
    "fast_cycle": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "cycle_context"],
    "fast_all": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "fast_flow",
                 "fast_macro", "cycle_context"],
    # valid from 2019+ only (a separate population, never ranked against the sets above)
    "fast_sent": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "fast_sentiment"],
    "fast_funding": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "funding"],
    "fast_deriv": ["fast_price", "fast_vol", "fast_struct", "fast_volume", "fast_intraday", "funding",
                   "open_interest", "implied_vol"],
}


def _reg(f: pd.DataFrame, name: str, group: str, values) -> None:
    TRADE_GROUPS[name] = group
    f[name] = values


def valid_from(feature_set: str) -> pd.Timestamp:
    return max(pd.Timestamp(GROUP_START[g]) for g in TRADE_FEATURE_SETS[feature_set])


def trade_columns(feature_set: str, available: list[str]) -> list[str]:
    groups = TRADE_FEATURE_SETS[feature_set]
    return [c for c in available if TRADE_GROUPS.get(c) in groups]


def _z(x: pd.Series, n: int, minp: int | None = None) -> pd.Series:
    m = x.rolling(n, min_periods=minp or n // 2)
    return (x - m.mean()) / m.std().replace(0, np.nan)


def _on_days(x: pd.Series, idx: pd.DatetimeIndex, shift_days: int = 0) -> pd.Series:
    """Daily observation series re-dated by `shift_days`, forward-filled onto the price calendar."""
    x = x.dropna()
    if x.empty:
        return pd.Series(np.nan, index=idx)
    x.index = pd.DatetimeIndex(x.index).normalize() + pd.Timedelta(days=shift_days)
    x = x[~x.index.duplicated(keep="last")]
    return x.reindex(x.index.union(idx)).ffill().reindex(idx)


def price_vol_features(d: pd.DataFrame) -> pd.DataFrame:
    c, h, lo, o = d["close"], d["high"], d["low"], d["open"]
    lc = np.log(c)
    r = lc.diff()
    sd = r.rolling(30, min_periods=15).std()
    f = pd.DataFrame(index=d.index)
    for n in (1, 3, 5, 10, 20):
        ret = lc - lc.shift(n)
        _reg(f, f"ret_{n}d", "fast_price", ret)
        _reg(f, f"ret_{n}d_z", "fast_price", ret / (sd * np.sqrt(n)))
    for n in (20, 50, 100):
        _reg(f, f"ma{n}_dist_z", "fast_price", (lc - np.log(sma(c, n))) / (sd * np.sqrt(n / 4)))
    _reg(f, "rsi_14", "fast_price", rsi(c, 14))
    _reg(f, "rsi_7", "fast_price", rsi(c, 7))
    for n in (20, 55):
        hh, ll = h.rolling(n, min_periods=n // 2).max(), lo.rolling(n, min_periods=n // 2).min()
        _reg(f, f"range_pos_{n}", "fast_price", (c - ll) / (hh - ll).replace(0, np.nan))
    up = np.sign(r).fillna(0)
    run = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    _reg(f, "streak", "fast_price", (run * up).clip(-10, 10))

    rv10 = r.rolling(10, min_periods=7).std() * np.sqrt(365)
    rv30 = r.rolling(30, min_periods=15).std() * np.sqrt(365)
    _reg(f, "rv_10", "fast_vol", rv10)
    _reg(f, "rv_30", "fast_vol", rv30)
    _reg(f, "rv_ratio", "fast_vol", np.log(rv10 / rv30))
    hl = np.log(h / lo)
    _reg(f, "parkinson_14", "fast_vol", np.sqrt((hl ** 2).rolling(14, min_periods=7).mean() / (4 * np.log(2))) * np.sqrt(365))
    co = np.log(c / o)
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    _reg(f, "garman_klass_14", "fast_vol", np.sqrt(gk.rolling(14, min_periods=7).mean().clip(lower=0)) * np.sqrt(365))
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    _reg(f, "atr_pct_14", "fast_vol", tr.rolling(14, min_periods=7).mean() / c)
    _reg(f, "vol_of_vol_60", "fast_vol", np.log(rv10).diff(5).rolling(60, min_periods=30).std())
    _reg(f, "rv30_pct_1y", "fast_vol", rolling_pct(rv30, 365, 180))

    prior_hi = c.shift(1).rolling(20, min_periods=10).max()
    prior_lo = c.shift(1).rolling(20, min_periods=10).min()
    _reg(f, "breakout_20", "fast_struct", (c > prior_hi).astype(float).where(prior_hi.notna()))
    _reg(f, "breakdown_20", "fast_struct", (c < prior_lo).astype(float).where(prior_lo.notna()))
    failed = ((h > prior_hi) & (c < prior_hi)).astype(float)
    _reg(f, "failed_breakout_5", "fast_struct", failed.rolling(5, min_periods=1).max().where(prior_hi.notna()))
    pos = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
    hi_at = pos.where(c >= c.rolling(60, min_periods=20).max()).ffill()
    lo_at = pos.where(c <= c.rolling(60, min_periods=20).min()).ffill()
    _reg(f, "days_since_60d_high", "fast_struct", (pos - hi_at) / 60)
    _reg(f, "days_since_60d_low", "fast_struct", (pos - lo_at) / 60)
    _reg(f, "dd_60d_high", "fast_struct", lc - np.log(c.rolling(60, min_periods=20).max()))
    _reg(f, "rally_60d_low", "fast_struct", lc - np.log(c.rolling(60, min_periods=20).min()))

    v = d["volume"].where(d["volume"] > 0)
    lv = np.log(v)
    _reg(f, "volume_z_20", "fast_volume", _z(lv, 20))
    _reg(f, "volume_z_5_mean", "fast_volume", _z(lv, 60).rolling(5, min_periods=3).mean())
    upv = v.where(r > 0, 0.0)
    _reg(f, "up_volume_share_20", "fast_volume", upv.rolling(20, min_periods=10).sum() / v.rolling(20, min_periods=10).sum())
    _reg(f, "volume_price_div", "fast_volume", np.sign(f["ret_5d"]) * f["volume_z_5_mean"])
    return f


def intraday_features(d: pd.DataFrame, h4: pd.DataFrame | None) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    if h4 is None or h4.empty:
        return f
    c4 = h4["close"]
    r4 = np.log(c4).diff()
    day = h4.index.normalize()
    rv = (r4 ** 2).groupby(day).sum()
    _reg(f, "rv4h_7", "fast_intraday", np.sqrt(rv.reindex(d.index).rolling(7, min_periods=4).mean() * 365))
    r = np.log(d["close"]).diff()
    _reg(f, "rv4h_vs_daily", "fast_intraday", np.log(f["rv4h_7"] / (r.rolling(10, min_periods=7).std() * np.sqrt(365))))
    rs4 = rsi(c4, 14)
    _reg(f, "rsi4h_max_3d", "fast_intraday", rs4.groupby(day).max().reindex(d.index).rolling(3, min_periods=1).max())
    _reg(f, "rsi4h_min_3d", "fast_intraday", rs4.groupby(day).min().reindex(d.index).rolling(3, min_periods=1).min())
    hour = h4.index.hour
    covered = d.index >= day.min()  # no 4h history before this: leave NaN, don't invent zeros
    for name, hours in (("asia", (0, 4)), ("europe", (8, 12)), ("us", (16, 20))):
        seg = r4[np.isin(hour, hours)].groupby(day[np.isin(hour, hours)]).sum()
        seg = seg.reindex(d.index).fillna(0).where(covered)
        _reg(f, f"session_{name}_5d", "fast_intraday", seg.rolling(5, min_periods=3).mean())
    last = r4[hour == 20].groupby(day[hour == 20]).sum()
    _reg(f, "last_4h_ret", "fast_intraday", last.reindex(d.index))
    return f


def flow_sentiment_macro(d: pd.DataFrame, s: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    idx = d.index
    f = pd.DataFrame(index=idx)

    def col(name: str, shift: int) -> pd.Series:
        return _on_days(s[name], idx, shift) if name in s else pd.Series(np.nan, index=idx)

    fin, fout, mcap = col("FlowInExUSD", 1), col("FlowOutExUSD", 1), col("CapMrktCurUSD", 1)
    net = (fin - fout) / mcap
    _reg(f, "netflow_z_90", "fast_flow", _z(net, 90))
    _reg(f, "netflow_7d_z", "fast_flow", _z(net.rolling(7, min_periods=4).sum(), 180))
    _reg(f, "netflow_30d_z", "fast_flow", _z(net.rolling(30, min_periods=15).sum(), 365))
    sx = col("SplyExNtv", 1)
    _reg(f, "ex_supply_chg_7", "fast_flow", np.log(sx / sx.shift(7)))
    _reg(f, "ex_supply_chg_30", "fast_flow", np.log(sx / sx.shift(30)))
    _reg(f, "fees_z_90", "fast_flow", _z(np.log(col("FeeTotNtv", 1).where(lambda x: x > 0)), 90))
    adr = np.log(col("AdrActCnt", 1).where(lambda x: x > 0))
    _reg(f, "active_addr_z", "fast_flow", _z(adr.rolling(7, min_periods=4).mean(), 90))

    fg = col("FearGreed", 1)
    _reg(f, "fear_greed", "fast_sentiment", fg / 100)
    _reg(f, "fear_greed_chg_7", "fast_sentiment", (fg - fg.shift(7)) / 100)
    _reg(f, "fear_greed_z_90", "fast_sentiment", _z(fg, 90))

    def macro(name: str) -> pd.Series:
        return col(name, lags.get(name, 0))

    dxy, y10, curve, real = macro("DTWEXBGS"), macro("DGS10"), macro("T10Y2Y"), macro("DFII10")
    _reg(f, "dxy_chg_5", "fast_macro", np.log(dxy / dxy.shift(5)))
    _reg(f, "dxy_chg_20", "fast_macro", np.log(dxy / dxy.shift(20)))
    _reg(f, "us10y_chg_5", "fast_macro", y10 - y10.shift(5))
    _reg(f, "us10y_chg_20", "fast_macro", y10 - y10.shift(20))
    _reg(f, "curve_chg_20", "fast_macro", curve - curve.shift(20))
    _reg(f, "real_yield_chg_20", "fast_macro", real - real.shift(20))
    return f


def derivatives_features(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    idx = d.index
    f = pd.DataFrame(index=idx)

    def obs(name: str) -> pd.Series:
        """Same-day observation, NOT forward-filled across its own start (NaN before history)."""
        if name not in s:
            return pd.Series(np.nan, index=idx)
        x = s[name].dropna()
        x.index = pd.DatetimeIndex(x.index).normalize()
        return x.reindex(idx).where(idx >= x.index.min()).ffill(limit=3)

    fund = obs("FUND_BINANCE").combine_first(obs("FUND_BYBIT"))
    _reg(f, "funding_1d", "funding", fund)
    _reg(f, "funding_7d", "funding", fund.rolling(7, min_periods=4).mean())
    _reg(f, "funding_z_90", "funding", _z(fund.rolling(3, min_periods=2).mean(), 90))
    _reg(f, "funding_pct_1y", "funding", rolling_pct(fund.rolling(7, min_periods=4).mean(), 365, 120))

    oi = np.log(obs("OI_BYBIT").where(lambda x: x > 0))
    lc = np.log(d["close"])
    _reg(f, "oi_chg_7", "open_interest", oi - oi.shift(7))
    _reg(f, "oi_chg_30", "open_interest", oi - oi.shift(30))
    _reg(f, "oi_price_div_7", "open_interest", (oi - oi.shift(7)) - (lc - lc.shift(7)))
    _reg(f, "oi_z_90", "open_interest", _z(oi, 90))

    dvol = obs("DVOL") / 100
    rv30 = np.log(d["close"]).diff().rolling(30, min_periods=15).std() * np.sqrt(365)
    _reg(f, "dvol", "implied_vol", dvol)
    _reg(f, "iv_rv_spread", "implied_vol", dvol - rv30)
    _reg(f, "dvol_chg_7", "implied_vol", dvol - dvol.shift(7))
    return f


def cycle_context(idx: pd.DatetimeIndex, oos: pd.DataFrame | None) -> pd.DataFrame:
    """The cycle model's walk-forward OUT-OF-SAMPLE score/strength - never the production model's
    in-sample score, which would leak its fit of the whole history into the swing model."""
    f = pd.DataFrame(index=idx)
    if oos is None or oos.empty:
        _reg(f, "cycle_score", "cycle_context", np.nan)
        _reg(f, "cycle_strength", "cycle_context", np.nan)
        return f
    o = oos.reindex(idx)
    _reg(f, "cycle_score", "cycle_context", o["score"])
    _reg(f, "cycle_strength", "cycle_context", o["strength"] / 10)
    return f


def build_trade_features(daily: pd.DataFrame, series: pd.DataFrame, lags: dict[str, int],
                         h4: pd.DataFrame | None = None, cycle_oos: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = [
        price_vol_features(daily),
        intraday_features(daily, h4),
        flow_sentiment_macro(daily, series, lags),
        derivatives_features(daily, series),
        cycle_context(daily.index, cycle_oos),
    ]
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def cycle_oos_predictions(cycle_feats: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Out-of-sample cycle score and strength for every date it exists.

    Walk-forward predictions of the run the production cycle model was built from (each epoch
    predicted by a model trained before it), extended past that run's last date with the
    production model's own output - but only after its `trained_through` cutoff, where it too is
    out of sample. Caveat: *which* run became production was chosen on all folds, a mild
    selection effect that no per-date construction can remove.
    """
    from .. import db
    from ..pipeline import all_roles, signal_history
    roles = all_roles()
    prod = roles[roles.role == "production"]
    if prod.empty or not prod.source_run_id.iloc[0]:
        return None
    df = db.query("SELECT date, score, strength FROM predictions WHERE run_id=? ORDER BY date",
                  (prod.source_run_id.iloc[0],))
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    oos = df.set_index("date")[["score", "strength"]]
    try:
        h = signal_history("production", cycle_feats)
    except Exception:  # noqa: BLE001 - the extension is optional; the walk-forward part stands alone
        h = None
    if h is not None:
        cutoff = pd.Timestamp(h.attrs["meta"].get("trained_through", oos.index[-1]))
        tail = h.loc[h.index > max(cutoff, oos.index[-1]), ["score", "strength"]]
        oos = pd.concat([oos, tail])
    return oos
