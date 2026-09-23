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
    # production set plus explicit 20-week / 50-week moving-average ratios and structure
    "core_ma": ["price", "onchain", "clock", "ma"],
    # exchange flows + fee pressure (behavioural on-chain)
    "core_flow": ["price", "onchain", "clock", "flow"],
    # parabola-exhaustion measures derived from price
    "core_exhaust": ["price", "onchain", "clock", "exhaust"],
    # both of the above together
    "core_tier1": ["price", "onchain", "clock", "flow", "exhaust"],
    # production set plus the 40-week and 50-week moving-average ratios only
    "core_ma4050": ["price", "onchain", "clock", "+p_40w", "+p_50w", "+p_40w_pct4y", "+p_50w_pct4y"],
    "core_exhaust_ma4050": ["price", "onchain", "clock", "exhaust",
                            "+p_40w", "+p_50w", "+p_40w_pct4y", "+p_50w_pct4y"],
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
    "ex_supply_chg_90", "ex_flow_ratio", "ex_inflow_vs_mcap", "ex_inflow_pct4y", "fee_pct4y",
    "accel_90_365", "accel_30_90", "ret_30_z", "ret_30_pct4y", "rsi_m_hot_365",
    "mayer", "p_200w", "p_2yma", "pi_cycle", "logreg_resid", "mvrv", "mvrv_z", "nupl", "puell",
    "rsi_m", "rsi_w", "mayer_pct4y", "mvrv_pct4y", "puell_pct4y", "rsi_m_pct4y", "p_200w_pct4y",
    "p_2yma_pct4y", "logreg_resid_pct4y", "mvrv_z_pct4y", "pi_cycle_pct4y",
    "p_20w", "p_40w", "p_50w", "ma20w_vs_50w", "p_20w_pct4y", "p_40w_pct4y", "p_50w_pct4y",
    "ma20w_vs_50w_pct4y",
]

# Components of the rule-based composite baseline (no training).
RULE_COMPONENTS = ["mayer_pctx", "p_200w_pctx", "mvrv_pctx", "puell_pctx", "rsi_m_pctx", "pi_cycle_pctx"]


# One-line description per feature, shown in the Feature explorer and rendered into the README.
# "log ratio" values are natural logs: 0 = equal, +0.69 = double, -0.69 = half.
FEATURE_DOCS: dict[str, str] = {
    # --- price / technical -------------------------------------------------
    "mayer": "Mayer Multiple in logs: price vs its 200-day average. >0 above, historically ~+0.9 at tops.",
    "p_200w": "Price vs its 200-week average (log). The 200w average has acted as the bear-market floor: near 0 at cycle lows.",
    "p_2yma": "Price vs its 2-year average (log).",
    "pi_cycle": "Pi Cycle Top: log gap between the 111-day average and twice the 350-day average. Crossing 0 flagged the 2013/2017/2021 tops.",
    "macd_w": "Weekly-style MACD: 12-week minus 26-week EMA, divided by price.",
    "ret_30": "30-day log return.",
    "ret_90": "90-day log return.",
    "ret_180": "180-day log return.",
    "ret_365": "1-year log return.",
    "vol_30": "30-day realised volatility, annualised.",
    "vol_90": "90-day realised volatility, annualised.",
    "vol_365": "1-year realised volatility, annualised.",
    "vol_ratio": "Log ratio of 30-day to 1-year volatility: >0 means the market is more volatile than usual.",
    "dd_ath": "Drawdown from the all-time high (log). 0 = at a new high, -1.4 = 75% below.",
    "days_since_ath": "Days since the last all-time high, in 4-year units (1.0 = 1460 days).",
    "rise_2y_low": "How far price has risen above its lowest point in the last 2 years (log).",
    "rsi_d": "Daily RSI(14).",
    "rsi_w": "Weekly RSI(14), with this week's bar still forming - what a live chart shows.",
    "rsi_m": "Monthly RSI(14), with this month's bar still forming. Historically >90 at cycle tops.",
    "rsi_m_div": "Monthly RSI minus its own highest value of the past year: 0 = at a new RSI high, negative = momentum diverging.",
    "bb_width": "Bollinger-band width over 140 days (log): high = wide/volatile, low = a squeeze.",
    "logreg_resid": "Distance above/below the long-term log-log growth trend, refit monthly on past data only.",
    "mayer_pctx": "Mayer Multiple as a percentile of its whole history to date (0-1).",
    "p_200w_pctx": "Price/200-week average as a percentile of its whole history to date.",
    "rsi_m_pctx": "Monthly RSI as a percentile of its whole history to date.",
    "pi_cycle_pctx": "Pi Cycle gap as a percentile of its whole history to date.",
    "mayer_pct4y": "Mayer Multiple as a percentile of the last 4 years: 1.0 = hottest in a cycle. Cycle-relative, so it still reaches extremes as amplitudes shrink.",
    "p_200w_pct4y": "Price/200-week average as a percentile of the last 4 years.",
    "rsi_m_pct4y": "Monthly RSI as a percentile of the last 4 years.",
    "pi_cycle_pct4y": "Pi Cycle gap as a percentile of the last 4 years.",
    "p_2yma_pct4y": "Price/2-year average as a percentile of the last 4 years.",
    "logreg_resid_pct4y": "Trend residual as a percentile of the last 4 years.",
    "ret_365_pct4y": "1-year return as a percentile of the last 4 years.",
    "vol_90_pct4y": "90-day volatility as a percentile of the last 4 years.",
    # --- moving averages ---------------------------------------------------
    "p_20w": "Price vs its 20-week (140-day) average (log) - the classic bull-market support line.",
    "p_40w": "Price vs its 40-week (280-day) average (log).",
    "p_50w": "Price vs its 50-week (350-day) average (log).",
    "p_20w_pct4y": "Price/20-week average as a percentile of the last 4 years.",
    "p_40w_pct4y": "Price/40-week average as a percentile of the last 4 years.",
    "p_50w_pct4y": "Price/50-week average as a percentile of the last 4 years.",
    "ma20w_slope": "Slope of the 20-week average: its 90-day log change. >0 = rising.",
    "ma50w_slope": "Slope of the 50-week average: its 180-day log change.",
    "ma20w_vs_50w": "Log gap between the 20-week and 50-week averages: >0 after a golden cross, <0 after a death cross.",
    "ma20w_vs_50w_pct4y": "That 20W/50W gap as a percentile of the last 4 years.",
    "weeks_since_ma_cross": "Time since the 20-week average last crossed the 50-week, in 4-year units.",
    # --- on-chain ----------------------------------------------------------
    "mvrv": "MVRV in logs: market value vs realised value (what all coins last moved at). High = large unrealised profits.",
    "mvrv_z": "MVRV Z-score: (market cap - realised cap) / expanding std of market cap. A classic top marker, though it peaks lower each cycle.",
    "nupl": "Net Unrealised Profit/Loss: share of market value that is paper profit. <0 = the average coin is under water.",
    "puell": "Puell Multiple (log): daily miner issuance value vs its 1-year average. High = miner revenue unusually rich.",
    "hash_ribbon": "Hash ribbon: log ratio of 30-day to 60-day hash rate. <0 = miner capitulation, historically near lows.",
    "hash_growth": "1-year growth in hash rate (log).",
    "adr_growth": "1-year growth in active addresses (log).",
    "tx_growth": "1-year growth in transaction count (log).",
    "mvrv_pctx": "MVRV as a percentile of its whole history to date.",
    "mvrv_pct4y": "MVRV as a percentile of the last 4 years.",
    "puell_pctx": "Puell Multiple as a percentile of its whole history to date.",
    "puell_pct4y": "Puell Multiple as a percentile of the last 4 years.",
    "mvrv_z_pctx": "MVRV Z-score as a percentile of its whole history to date.",
    "mvrv_z_pct4y": "MVRV Z-score as a percentile of the last 4 years.",
    # --- cycle clock -------------------------------------------------------
    "days_since_halving": "Time since the last halving, in 4-year units. Tops have come ~0.36-0.38 (525-550 days) after one.",
    "halving_phase_sin": "Sine of the halving phase - lets the model treat the 4-year clock as circular.",
    "halving_phase_cos": "Cosine of the halving phase.",
    "days_since_top1": "Time since the most recent *confirmed* cycle top, in 4-year units. Confirmation is causal: a top counts only once price has halved from it.",
    "days_since_top2": "Time since the confirmed top before that.",
    "days_since_low1": "Time since the most recent confirmed cycle low (confirmed once price has doubled off it).",
    "days_since_low2": "Time since the confirmed low before that.",
    "vs_last_top": "Price vs the last confirmed top (log): 0 = back at that level.",
    "vs_last_low": "Price vs the last confirmed low (log).",
    "top_minus_low_recency": "Which confirmed extreme is more recent, and by how much: >0 = the low came more recently (we are rising).",
    # --- exhaustion --------------------------------------------------------
    "accel_90_365": "Acceleration: daily pace of the last 90 days minus that of the last year. >0 = the advance is speeding up.",
    "accel_30_90": "Short-term acceleration: 30-day pace minus 90-day pace.",
    "ret_30_z": "The 30-day move in standard deviations of daily returns - how stretched the last month is.",
    "ret_30_pct4y": "30-day return as a percentile of the last 4 years.",
    "dd_90d_high": "Drawdown from the highest price of the last 90 days (log).",
    "days_above_20w": "Unbroken run of days above the 20-week average, in 4-year units.",
    "rsi_m_hot_365": "Share of the last year that monthly RSI spent above 70 (0-1) - how long the market has been hot.",
    # --- exchange flows, fees, addresses ------------------------------------
    "ex_supply_share": "Share of all coins sitting on exchanges (log). A falling share = accumulation into self-custody.",
    "ex_supply_chg_90": "90-day change in that exchange share. >0 = coins moving onto exchanges.",
    "ex_supply_chg_365": "1-year change in the exchange share.",
    "ex_supply_pct4y": "Exchange share as a percentile of the last 4 years.",
    "ex_flow_ratio": "Log ratio of 30-day exchange deposits to withdrawals. >0 = more coins arriving than leaving (selling pressure).",
    "ex_inflow_vs_mcap": "30-day exchange deposits relative to market cap (log).",
    "ex_inflow_pct4y": "That deposit intensity as a percentile of the last 4 years.",
    "ex_inflow_spike": "Log ratio of 7-day to 1-year average deposits - a sudden rush to exchanges.",
    "fee_vs_issuance": "Total transaction fees vs miner issuance (log, 30-day). High = blockspace demand is intense, as in Dec 2017.",
    "fee_per_tx": "Average fee per transaction in BTC (log, 30-day).",
    "fee_pct4y": "Fee-to-issuance ratio as a percentile of the last 4 years.",
    "addr_base_growth": "1-year growth in the number of addresses holding a balance (log) - adoption.",
    # --- Ether -------------------------------------------------------------
    "ethbtc": "ETH/BTC vs its own 1-year average (log). Rising = risk appetite rotating into altcoins.",
    "ethbtc_ret_90": "90-day log change in ETH/BTC.",
    "ethbtc_ret_365": "1-year log change in ETH/BTC.",
    "ethbtc_dd_ath": "ETH/BTC drawdown from its all-time high (log).",
    "ethbtc_pct4y": "ETH/BTC (vs its year average) as a percentile of the last 4 years.",
    "eth_vs_sma200": "ETH price vs its 200-day average (log).",
    "eth_ret_90": "90-day log return of ETH/USD.",
    "eth_ret_365": "1-year log return of ETH/USD.",
    "eth_vol_ratio": "Log ratio of ETH to BTC 90-day volatility: how much more violently ETH is moving.",
    "eth_btc_mcap": "ETH market cap vs BTC market cap (log).",
    "eth_btc_mcap_pct4y": "That market-cap ratio as a percentile of the last 4 years.",
    "eth_adr_growth": "1-year growth in ETH active addresses (log).",
    # --- macro (each lagged to its publication date) -------------------------
    "m2_yoy": "US M2 money supply, 1-year log growth.",
    "m2_yoy_chg_365": "Change in that M2 growth rate over a year - liquidity accelerating or tightening.",
    "fed_bs_yoy": "Fed balance sheet, 1-year log growth. >0 = QE, <0 = QT.",
    "dxy_chg_90": "Broad dollar index, 90-day log change. A rising dollar is usually a headwind.",
    "dxy_chg_365": "Broad dollar index, 1-year log change.",
    "us10y": "US 10-year Treasury yield, percent.",
    "us10y_chg_180": "Change in the 10-year yield over 180 days, percentage points.",
    "fed_funds": "Effective fed funds rate, percent - the policy rate itself.",
    "fed_funds_chg_180": "Change in the policy rate over 180 days: >0 hiking, <0 cutting.",
    "fed_funds_chg_365": "Change in the policy rate over a year.",
    "fed_funds_pct4y": "Policy rate as a percentile of the last 4 years: 1.0 = tightest in a cycle.",
    "real_yield_10y": "US 10-year real (inflation-linked) yield, percent - the opportunity cost of holding a non-yielding asset.",
    "yield_curve": "10-year minus 2-year Treasury spread. <0 = inverted.",
    # --- intraday (from 4h candles) -----------------------------------------
    "intraday_vol_30": "30-day average of volatility measured within each day from 4-hour bars, annualised.",
    "intraday_vs_close_vol": "Log ratio of intraday to close-to-close volatility.",
    "rsi_4h_max_7d": "Highest 4-hour RSI of the past week.",
    "rsi_4h_min_7d": "Lowest 4-hour RSI of the past week.",
    # --- calendar (ablation only) -------------------------------------------
    "month_sin": "Sine of the month-of-year - included only to test whether seasonality helps (it does not).",
    "month_cos": "Cosine of the month-of-year.",
    "week_sin": "Sine of the week-of-year.",
    "week_cos": "Cosine of the week-of-year.",
    # --- sentiment (display only) -------------------------------------------
    "fear_greed_30": "Crypto Fear & Greed index, 30-day average (2018+). Not in any training set - history is too short.",
}


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


def ma_features(d: pd.DataFrame) -> pd.DataFrame:
    """Classic weekly moving averages as explicit ratios, plus MA structure (slope, 20W/50W gap,
    time since the cross) which a single price/MA ratio cannot express."""
    c = d["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=d.index)
    ma20w, ma40w, ma50w = sma(c, 140, 70), sma(c, 280, 140), sma(c, 350, 175)
    f[_reg("p_20w", "ma")] = lc - np.log(ma20w)
    f[_reg("p_40w", "ma")] = lc - np.log(ma40w)
    f[_reg("p_50w", "ma")] = lc - np.log(ma50w)
    f[_reg("p_40w_pct4y", "ma")] = rolling_pct(f["p_40w"])
    f[_reg("p_50w_pct4y", "ma")] = rolling_pct(f["p_50w"])
    f[_reg("ma20w_slope", "ma")] = np.log(ma20w / ma20w.shift(90))
    f[_reg("ma50w_slope", "ma")] = np.log(ma50w / ma50w.shift(180))
    gap = np.log(ma20w / ma50w)          # > 0 after a golden cross, < 0 after a death cross
    f[_reg("ma20w_vs_50w", "ma")] = gap
    crossed = np.sign(gap).diff().fillna(0) != 0
    idx = pd.Series(np.where(crossed, np.arange(len(c)), np.nan), index=c.index).ffill()
    f[_reg("weeks_since_ma_cross", "ma")] = (np.arange(len(c)) - idx) / EPOCH_DAYS
    f[_reg("p_20w_pct4y", "ma")] = rolling_pct(f["p_20w"])
    f[_reg("ma20w_vs_50w_pct4y", "ma")] = rolling_pct(gap)
    return f


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


def flow_features(d: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Behavioural on-chain data: coins moving to/from exchanges, fee pressure, address base.

    Valuation metrics (MVRV, Puell) say whether price is stretched; these say what holders are
    actually doing — deposits spike when people intend to sell, fees spike when the crowd arrives.
    """
    f = pd.DataFrame(index=d.index)
    s = s.reindex(d.index).ffill(limit=5)
    if "SplyExNtv" in s and "SplyCur" in s:
        share = np.log(s["SplyExNtv"] / s["SplyCur"])
        f[_reg("ex_supply_share", "flow")] = share
        f[_reg("ex_supply_chg_90", "flow")] = share - share.shift(90)
        f[_reg("ex_supply_chg_365", "flow")] = share - share.shift(365)
        f[_reg("ex_supply_pct4y", "flow")] = rolling_pct(share)
    if "FlowInExUSD" in s and "FlowOutExUSD" in s:
        inflow, outflow = sma(s["FlowInExUSD"], 30, 15), sma(s["FlowOutExUSD"], 30, 15)
        f[_reg("ex_flow_ratio", "flow")] = np.log(inflow / outflow)
        if "CapMrktCurUSD" in s:
            rel = np.log(inflow / s["CapMrktCurUSD"])
            f[_reg("ex_inflow_vs_mcap", "flow")] = rel
            f[_reg("ex_inflow_pct4y", "flow")] = rolling_pct(rel)
        f[_reg("ex_inflow_spike", "flow")] = np.log(sma(s["FlowInExUSD"], 7, 4) / sma(s["FlowInExUSD"], 365, 180))
    if "FeeTotNtv" in s:
        fee_usd = s["FeeTotNtv"] * d["close"]
        f[_reg("fee_vs_issuance", "flow")] = np.log(sma(fee_usd, 30, 15) / sma(s["IssTotUSD"], 30, 15))
        f[_reg("fee_per_tx", "flow")] = np.log(sma(s["FeeTotNtv"], 30, 15) / sma(s["TxCnt"], 30, 15))
        f[_reg("fee_pct4y", "flow")] = rolling_pct(f["fee_vs_issuance"])
    if "AdrBalCnt" in s:
        a = sma(s["AdrBalCnt"], 30, 15)
        f[_reg("addr_base_growth", "flow")] = np.log(a / a.shift(365))
    return f


def exhaustion_features(d: pd.DataFrame) -> pd.DataFrame:
    """Parabola-exhaustion measures: is the advance accelerating into a blow-off, or just steep?

    Trend-following inputs (MAs, Mayer) stay bullish through an entire parabola; these are meant to
    separate mid-run from terminal.
    """
    c = d["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=d.index)
    f[_reg("accel_90_365", "exhaust")] = (lc - lc.shift(90)) / 90 - (lc - lc.shift(365)) / 365
    f[_reg("accel_30_90", "exhaust")] = (lc - lc.shift(30)) / 30 - (lc - lc.shift(90)) / 90
    vol90 = lc.diff().rolling(90, min_periods=45).std()
    f[_reg("ret_30_z", "exhaust")] = (lc - lc.shift(30)) / (vol90 * np.sqrt(30))
    f[_reg("ret_30_pct4y", "exhaust")] = rolling_pct(lc - lc.shift(30))
    f[_reg("dd_90d_high", "exhaust")] = lc - np.log(c.rolling(90, min_periods=45).max())
    above = (c > sma(c, 140, 70)).astype(float)
    streak = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    f[_reg("days_above_20w", "exhaust")] = streak / EPOCH_DAYS
    rsi_m = htf_rsi(c, "M")
    f[_reg("rsi_m_hot_365", "exhaust")] = (rsi_m > 70).rolling(365, min_periods=180).mean()
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
    ff, real10, curve = avail("DFF"), avail("DFII10"), avail("T10Y2Y")
    f[_reg("fed_bs_yoy", "macro")] = np.log(walcl / walcl.shift(365))
    f[_reg("dxy_chg_90", "macro")] = np.log(dxy / dxy.shift(90))
    f[_reg("dxy_chg_365", "macro")] = np.log(dxy / dxy.shift(365))
    f[_reg("us10y", "macro")] = y10
    f[_reg("us10y_chg_180", "macro")] = y10 - y10.shift(180)
    # policy rate: level, direction, and where it sits in its own 4-year range
    f[_reg("fed_funds", "macro")] = ff
    f[_reg("fed_funds_chg_180", "macro")] = ff - ff.shift(180)
    f[_reg("fed_funds_chg_365", "macro")] = ff - ff.shift(365)
    f[_reg("fed_funds_pct4y", "macro")] = rolling_pct(ff)
    f[_reg("real_yield_10y", "macro")] = real10
    f[_reg("yield_curve", "macro")] = curve
    f[_reg("m2_yoy_chg_365", "macro")] = f["m2_yoy"] - f["m2_yoy"].shift(365) if "m2_yoy" in f else np.nan
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
        ma_features(daily),
        flow_features(daily, series),
        exhaustion_features(daily),
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
    """A set is a list of group names, plus optional '+column' entries naming single features."""
    spec = FEATURE_SETS[feature_set]
    groups = [g for g in spec if not g.startswith("+")]
    singles = [g[1:] for g in spec if g.startswith("+")]
    if "cyclenorm" in groups:
        extra = [c for c in available if FEATURE_GROUPS.get(c) in groups and c not in CYCLENORM]
        cols = [c for c in CYCLENORM if c in available] + extra
    else:
        cols = [c for c in available if FEATURE_GROUPS.get(c) in groups]
    return cols + [c for c in singles if c in available and c not in cols]


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: (daily prices, features) from the database."""
    from .. import db
    from ..data import daily_prices

    daily = daily_prices()
    series = db.read_series()
    h4 = db.read_candles("4h")
    feats = build_features(daily, series, db.series_lags(), h4)
    return daily, feats
