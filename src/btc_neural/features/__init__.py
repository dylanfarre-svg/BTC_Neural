"""Causal, scale-free features.

Every feature at date t uses only information available at the end of day t
(tests/test_no_lookahead.py enforces this). No raw prices or raw dates are used:
everything is a log ratio, oscillator, percentile, or cycle-relative clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import halvings

GENESIS = pd.Timestamp("2009-01-03")
EPOCH_DAYS = 1460.0

# feature name -> group. Filled by the builders below.
FEATURE_GROUPS: dict[str, str] = {}

FEATURE_SETS: dict[str, list[str]] = {
    "core": ["price", "onchain", "clock"],
    "core_macro": ["price", "onchain", "clock", "macro"],
    "core_calendar": ["price", "onchain", "clock", "calendar"],
    "core_intraday": ["price", "onchain", "clock", "intraday"],
    "price_clock": ["price", "clock"],
    "no_clock": ["price", "onchain"],
    # Only amplitude-free features (4-year percentiles, oscillators, cycle clocks), to cope with
    # each cycle being weaker than the last.
    "cyclenorm": ["cyclenorm"],
    # production feature set plus Ether (ETH/BTC and ETH/USD); ETH history starts 2015-08
    "core_eth": ["price", "onchain", "clock", "eth"],
    "cyclenorm_eth": ["cyclenorm", "eth"],
}

CYCLENORM = [
    "mayer_pct4y", "p_200w_pct4y", "rsi_m_pct4y", "pi_cycle_pct4y", "p_2yma_pct4y", "logreg_resid_pct4y",
    "ret_365_pct4y", "vol_90_pct4y", "mvrv_pct4y", "puell_pct4y", "mvrv_z_pct4y", "rsi_d", "rsi_w", "rsi_m",
    "rsi_m_div", "dd_ath", "days_since_ath", "vol_ratio", "hash_ribbon", "days_since_halving",
    "halving_phase_sin", "halving_phase_cos", "days_since_top1", "days_since_top2", "days_since_low1",
    "days_since_low2", "vs_last_top", "vs_last_low", "top_minus_low_recency",
]

# Features whose higher values mean "more overheated" (push toward SELL). Used for
# monotone constraints in LightGBM and for the rule-based composite.
SELL_MONOTONE = [
    "mayer", "p_200w", "p_2yma", "pi_cycle", "logreg_resid", "mvrv", "mvrv_z", "nupl", "puell",
    "rsi_m", "rsi_w", "mayer_pct4y", "mvrv_pct4y", "puell_pct4y", "rsi_m_pct4y", "p_200w_pct4y",
    "p_2yma_pct4y", "logreg_resid_pct4y", "mvrv_z_pct4y", "pi_cycle_pct4y",
]

# Components of the rule-based composite baseline (no training).
RULE_COMPONENTS = ["mayer_pctx", "p_200w_pctx", "mvrv_pctx", "puell_pctx", "rsi_m_pctx", "pi_cycle_pctx"]


# --- helpers ---------------------------------------------------------------------

def sma(x: pd.Series, n: int, minp: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=minp or n).mean()


def expanding_pct(x: pd.Series, minp: int = 365) -> pd.Series:
    """Percentile of x_t among all values up to and including t (causal)."""
    a = x.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    hist = []
    import bisect
    for i, v in enumerate(a):
        if np.isnan(v):
            continue
        bisect.insort(hist, v)
        if len(hist) >= minp:
            out[i] = bisect.bisect_right(hist, v) / len(hist)
    return pd.Series(out, index=x.index)


def rolling_pct(x: pd.Series, window: int = 1460, minp: int = 365) -> pd.Series:
    return x.rolling(window, min_periods=minp).rank(pct=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    ch = close.diff()
    ag = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def htf_rsi(close: pd.Series, period: str, n: int = 14) -> pd.Series:
    """Higher-timeframe RSI as seen on each day: the current (partial) bar closes at today's price.

    This is what a chart shows for 'current weekly/monthly RSI' and uses no future data.
    """
    per = close.index.to_period(period)
    pc = close.groupby(per).last()
    ch = pc.diff()
    a = 1 / n
    ag = ch.clip(lower=0).ewm(alpha=a, adjust=False).mean()
    al = (-ch.clip(upper=0)).ewm(alpha=a, adjust=False).mean()
    prev_c = pc.shift(1).reindex(per).to_numpy()
    prev_ag = ag.shift(1).reindex(per).to_numpy()
    prev_al = al.shift(1).reindex(per).to_numpy()
    change = close.to_numpy() - prev_c
    g = (1 - a) * prev_ag + a * np.clip(change, 0, None)
    l_ = (1 - a) * prev_al + a * np.clip(-change, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + g / l_)
    return pd.Series(out, index=close.index)


def confirmed_extremes(close: pd.Series, drawdown: float = 0.5, rise: float = 1.0, min_days: int = 60) -> pd.DataFrame:
    """Causal cycle top/low detection.

    A top is confirmed once price falls `drawdown` below the running max; a low is
    confirmed once price rises `rise` above the running min and `min_days` have passed.
    Returns, per day, the dates/prices of the last two confirmed tops and lows *as known
    on that day*.
    """
    dates = close.index
    c = close.to_numpy()
    mode = "top"
    ext_i = 0
    tops: list[int] = []
    lows: list[int] = []
    rec = np.full((len(c), 4), -1, dtype=int)
    for i in range(len(c)):
        if mode == "top":
            if c[i] >= c[ext_i]:
                ext_i = i
            elif c[i] <= (1 - drawdown) * c[ext_i]:
                tops.append(ext_i)
                mode, ext_i = "low", i
        else:
            if c[i] <= c[ext_i]:
                ext_i = i
            elif c[i] >= (1 + rise) * c[ext_i] and i - ext_i >= min_days:
                lows.append(ext_i)
                mode, ext_i = "top", i
        rec[i] = [tops[-1] if tops else -1, tops[-2] if len(tops) > 1 else -1,
                  lows[-1] if lows else -1, lows[-2] if len(lows) > 1 else -1]
    out = pd.DataFrame(index=dates)
    idx = np.arange(len(c))
    for j, name in enumerate(["top1", "top2", "low1", "low2"]):
        r = rec[:, j]
        valid = r >= 0
        days = np.where(valid, idx - r, np.nan)
        price = np.where(valid, c[np.clip(r, 0, None)], np.nan)
        out[f"days_since_{name}"] = days
        out[f"price_{name}"] = price
    return out


def _reg(name: str, group: str) -> str:
    FEATURE_GROUPS[name] = group
    return name


# --- builders ----------------------------------------------------------------------

def price_features(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=d.index)
    f[_reg("mayer", "price")] = lc - np.log(sma(c, 200, 100))
    f[_reg("p_200w", "price")] = lc - np.log(sma(c, 1400, 350))
    f[_reg("p_2yma", "price")] = lc - np.log(sma(c, 730, 250))
    f[_reg("pi_cycle", "price")] = np.log(sma(c, 111)) - np.log(2 * sma(c, 350, 200))
    ema_fast, ema_slow = c.ewm(span=84, adjust=False).mean(), c.ewm(span=182, adjust=False).mean()
    f[_reg("macd_w", "price")] = (ema_fast - ema_slow) / c
    for n in (30, 90, 180, 365):
        f[_reg(f"ret_{n}", "price")] = lc - lc.shift(n)
    for n in (30, 90, 365):
        f[_reg(f"vol_{n}", "price")] = r.rolling(n, min_periods=n // 2).std() * np.sqrt(365)
    f[_reg("vol_ratio", "price")] = np.log(f["vol_30"] / f["vol_365"])
    ath = c.cummax()
    f[_reg("dd_ath", "price")] = lc - np.log(ath)
    ath_idx = pd.Series(np.where(c >= ath, np.arange(len(c)), np.nan), index=c.index).ffill()
    f[_reg("days_since_ath", "price")] = (np.arange(len(c)) - ath_idx) / EPOCH_DAYS
    lo2 = c.rolling(730, min_periods=180).min()
    f[_reg("rise_2y_low", "price")] = lc - np.log(lo2)
    f[_reg("rsi_d", "price")] = rsi(c, 14)
    f[_reg("rsi_w", "price")] = htf_rsi(c, "W-SUN")
    f[_reg("rsi_m", "price")] = htf_rsi(c, "M")
    f[_reg("rsi_m_div", "price")] = f["rsi_m"] - f["rsi_m"].rolling(365, min_periods=90).max()
    f[_reg("bb_width", "price")] = np.log((c.rolling(140, min_periods=70).std() * 4) / sma(c, 140, 70))
    f[_reg("logreg_resid", "price")] = log_regression_residual(c)

    # Cycle-normalized versions: percentile of history (expanding) and of the last 4 years.
    for base in ("mayer", "p_200w", "rsi_m", "pi_cycle"):
        f[_reg(f"{base}_pctx", "price")] = expanding_pct(f[base])
    for base in ("mayer", "p_200w", "rsi_m", "pi_cycle", "p_2yma", "logreg_resid", "ret_365", "vol_90"):
        f[_reg(f"{base}_pct4y", "price")] = rolling_pct(f[base])
    return f


def log_regression_residual(c: pd.Series, refit: str = "MS") -> pd.Series:
    """Residual from ln(price) ~ a + b*ln(days since genesis), refit monthly on past data only."""
    x = np.log((c.index - GENESIS).days.to_numpy(dtype=float))
    y = np.log(c.to_numpy())
    out = np.full(len(c), np.nan)
    months = c.index.to_period("M")
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    bounds = list(starts) + [len(c)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if s < 365:
            continue
        b, a = np.polyfit(x[:s], y[:s], 1)
        out[s:e] = y[s:e] - (a + b * x[s:e])
    return pd.Series(out, index=c.index)


def onchain_features(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    s = s.reindex(d.index).ffill(limit=5)
    if "CapMVRVCur" not in s:
        return f
    mvrv = s["CapMVRVCur"]
    mcap = s["CapMrktCurUSD"]
    rcap = mcap / mvrv
    f[_reg("mvrv", "onchain")] = np.log(mvrv)
    f[_reg("mvrv_z", "onchain")] = (mcap - rcap) / mcap.expanding(min_periods=365).std()
    f[_reg("nupl", "onchain")] = 1 - 1 / mvrv
    iss = s["IssTotUSD"]
    f[_reg("puell", "onchain")] = np.log(iss / sma(iss, 365, 180))
    h = s["HashRate"]
    f[_reg("hash_ribbon", "onchain")] = np.log(sma(h, 30) / sma(h, 60))
    f[_reg("hash_growth", "onchain")] = np.log(sma(h, 30) / sma(h, 30).shift(365))
    f[_reg("adr_growth", "onchain")] = np.log(sma(s["AdrActCnt"], 30) / sma(s["AdrActCnt"], 30).shift(365))
    f[_reg("tx_growth", "onchain")] = np.log(sma(s["TxCnt"], 30) / sma(s["TxCnt"], 30).shift(365))
    for base in ("mvrv", "puell", "mvrv_z"):
        f[_reg(f"{base}_pctx", "onchain")] = expanding_pct(f[base])
        f[_reg(f"{base}_pct4y", "onchain")] = rolling_pct(f[base])
    return f


def clock_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    hs = [GENESIS] + halvings()
    hs_arr = np.array(hs, dtype="datetime64[ns]")
    k = np.searchsorted(hs_arr, d.index.values, side="right") - 1
    last_h = pd.DatetimeIndex(hs_arr[k])
    f[_reg("days_since_halving", "clock")] = (d.index - last_h).days / EPOCH_DAYS
    ph = 2 * np.pi * f["days_since_halving"]
    f[_reg("halving_phase_sin", "clock")] = np.sin(ph)
    f[_reg("halving_phase_cos", "clock")] = np.cos(ph)
    ce = confirmed_extremes(d["close"])
    lc = np.log(d["close"])
    for name in ("top1", "top2", "low1", "low2"):
        f[_reg(f"days_since_{name}", "clock")] = ce[f"days_since_{name}"] / EPOCH_DAYS
    f[_reg("vs_last_top", "clock")] = lc - np.log(ce["price_top1"])
    f[_reg("vs_last_low", "clock")] = lc - np.log(ce["price_low1"])
    f[_reg("top_minus_low_recency", "clock")] = f["days_since_top1"] - f["days_since_low1"]
    return f


def macro_features(d: pd.DataFrame, s: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)

    def avail(name: str) -> pd.Series:
        """Series re-dated to when it was actually published, then forward-filled daily."""
        if name not in s:
            return pd.Series(np.nan, index=d.index)
        x = s[name].dropna()
        x.index = x.index + pd.Timedelta(days=lags.get(name, 0))
        x = x[~x.index.duplicated(keep="last")]
        return x.reindex(x.index.union(d.index)).ffill().reindex(d.index)

    if "M2SL" in s:
        m2 = s["M2SL"].dropna()
        yoy = np.log(m2 / m2.shift(12))
        yoy.index = yoy.index + pd.Timedelta(days=lags.get("M2SL", 60))
        f[_reg("m2_yoy", "macro")] = yoy.reindex(yoy.index.union(d.index)).ffill().reindex(d.index)
    walcl, dxy, y10 = avail("WALCL"), avail("DTWEXBGS"), avail("DGS10")
    f[_reg("fed_bs_yoy", "macro")] = np.log(walcl / walcl.shift(365))
    f[_reg("dxy_chg_90", "macro")] = np.log(dxy / dxy.shift(90))
    f[_reg("dxy_chg_365", "macro")] = np.log(dxy / dxy.shift(365))
    f[_reg("us10y", "macro")] = y10
    f[_reg("us10y_chg_180", "macro")] = y10 - y10.shift(180)
    return f


def calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    m = 2 * np.pi * (d.index.month - 1) / 12
    w = 2 * np.pi * (d.index.isocalendar().week.to_numpy() - 1) / 52
    f[_reg("month_sin", "calendar")] = np.sin(m)
    f[_reg("month_cos", "calendar")] = np.cos(m)
    f[_reg("week_sin", "calendar")] = np.sin(w)
    f[_reg("week_cos", "calendar")] = np.cos(w)
    return f


def intraday_features(d: pd.DataFrame, h4: pd.DataFrame | None) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    if h4 is None or h4.empty:
        return f
    lr = np.log(h4["close"]).diff()
    rv = (lr ** 2).groupby(h4.index.normalize()).sum().reindex(d.index)
    iv30 = np.sqrt(rv.rolling(30, min_periods=15).mean() * 365)
    f[_reg("intraday_vol_30", "intraday")] = iv30
    f[_reg("intraday_vs_close_vol", "intraday")] = np.log(iv30 / d["close"].pipe(np.log).diff().rolling(30, min_periods=15).std() / np.sqrt(365))
    r4 = rsi(h4["close"], 14)
    f[_reg("rsi_4h_max_7d", "intraday")] = r4.groupby(h4.index.normalize()).max().reindex(d.index).rolling(7, min_periods=1).max()
    f[_reg("rsi_4h_min_7d", "intraday")] = r4.groupby(h4.index.normalize()).min().reindex(d.index).rolling(7, min_periods=1).min()
    return f


def eth_features(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Ether as a second market: ETH/BTC and ETH/USD (history starts 2015-08, so NaN before that).

    ETH/BTC tends to lead risk appetite within the crypto cycle, which BTC-only inputs cannot see.
    """
    f = pd.DataFrame(index=d.index)
    s = s.reindex(d.index).ffill(limit=5)
    if "ETH_PriceUSD" not in s:
        return f
    eth = s["ETH_PriceUSD"]
    ethbtc = eth / d["close"]
    lr = np.log(ethbtc)
    f[_reg("ethbtc", "eth")] = lr - np.log(sma(ethbtc, 365, 180))
    f[_reg("ethbtc_ret_90", "eth")] = lr - lr.shift(90)
    f[_reg("ethbtc_ret_365", "eth")] = lr - lr.shift(365)
    f[_reg("ethbtc_dd_ath", "eth")] = lr - np.log(ethbtc.cummax())
    f[_reg("ethbtc_pct4y", "eth")] = rolling_pct(lr)
    le = np.log(eth)
    f[_reg("eth_vs_sma200", "eth")] = le - np.log(sma(eth, 200, 100))
    f[_reg("eth_ret_90", "eth")] = le - le.shift(90)
    f[_reg("eth_ret_365", "eth")] = le - le.shift(365)
    f[_reg("eth_vol_ratio", "eth")] = np.log(
        le.diff().rolling(90, min_periods=45).std() / np.log(d["close"]).diff().rolling(90, min_periods=45).std())
    if "ETH_CapMrktCurUSD" in s and "CapMrktCurUSD" in s:
        share = np.log(s["ETH_CapMrktCurUSD"] / s["CapMrktCurUSD"])
        f[_reg("eth_btc_mcap", "eth")] = share
        f[_reg("eth_btc_mcap_pct4y", "eth")] = rolling_pct(share)
    if "ETH_AdrActCnt" in s:
        a = sma(s["ETH_AdrActCnt"], 30)
        f[_reg("eth_adr_growth", "eth")] = np.log(a / a.shift(365))
    return f


def sentiment_features(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    if "FearGreed" in s:
        fg = s["FearGreed"].reindex(d.index)
        f[_reg("fear_greed_30", "sentiment")] = fg.rolling(30, min_periods=10).mean()
    return f


def build_features(daily: pd.DataFrame, series: pd.DataFrame, lags: dict[str, int],
                   h4: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = [
        price_features(daily),
        onchain_features(daily, series),
        clock_features(daily),
        macro_features(daily, series, lags),
        calendar_features(daily),
        intraday_features(daily, h4),
        eth_features(daily, series),
        sentiment_features(daily, series),
    ]
    f = pd.concat(parts, axis=1)
    return f.replace([np.inf, -np.inf], np.nan)


def columns_for(feature_set: str, available: list[str]) -> list[str]:
    groups = FEATURE_SETS[feature_set]
    if "cyclenorm" in groups:
        extra = [c for c in available if FEATURE_GROUPS.get(c) in groups and c not in CYCLENORM]
        return [c for c in CYCLENORM if c in available] + extra
    return [c for c in available if FEATURE_GROUPS.get(c) in groups]


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (daily prices, features) from the database."""
    from .. import db
    from ..data import daily_prices

    daily = daily_prices()
    series = db.read_series()
    h4 = db.read_candles("4h")
    feats = build_features(daily, series, db.series_lags(), h4)
    return daily, feats
