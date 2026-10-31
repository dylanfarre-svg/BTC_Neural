"""Discrete round-trip simulator for the trading signals.

A separate sleeve (PLAN_SWING.md Q3): one position at a time, the whole sleeve per trade, 1x
notional, no leverage. Its lots never mix with the cycle holding's.

Per trade, with c = fee + slippage per side, G = 1 + gross and F = accumulated funding per unit of
entry notional: net = (1-c)^2 (G + F) - 1. Longs are spot (no funding); shorts are perps and
receive funding when it is positive. The daily equity path is built so it compounds to exactly
that net, which keeps Sharpe, drawdown and total return consistent with the per-trade numbers.

Tax (Q5), applied to a fresh copy of the equity path per treatment: realized P&L is netted per SA
tax year (March-February), losses carry forward, tax is paid at year end, and whatever has accrued
is paid at the end of the window (liquidation basis). The R40k annual exclusion is ignored: it
belongs to the person, and the cycle holding is assumed to use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import db
from .horizons import Horizon
from .labels import barrier_outcomes, day_end_prices, h4_grid

SIDE_MULT = {"long": -1.0, "short": 1.0}  # funding sign: longs pay positive funding, shorts receive


@dataclass
class Market:
    hz: Horizon
    days: pd.DatetimeIndex
    px: np.ndarray                     # day-end price (00:00 UTC next day)
    funding: np.ndarray                # daily sum of 8h funding rates
    out: dict[str, pd.DataFrame]       # barrier outcomes per side
    exit_day: dict[str, np.ndarray]    # index into `days` of the day each trade exits in
    net: dict[str, np.ndarray]         # net return per entry day, after costs and funding
    funding_paid: dict[str, np.ndarray]

    def pos(self, t) -> int:
        return int(self.days.searchsorted(pd.Timestamp(t)))


def funding_series(days: pd.DatetimeIndex, hz: Horizon) -> np.ndarray:
    s = db.read_series(["FUND_BINANCE", "FUND_BYBIT"])
    fb = 3 * hz.funding_fallback_8h
    if s.empty:
        return np.full(len(days), fb)
    f = s.get("FUND_BINANCE", pd.Series(dtype=float)).combine_first(s.get("FUND_BYBIT", pd.Series(dtype=float)))
    return f.reindex(days).fillna(fb).to_numpy()


def build_market(daily: pd.DataFrame, h4: pd.DataFrame, hz: Horizon,
                 out: dict[str, pd.DataFrame] | None = None) -> Market:
    grid = h4_grid(h4)
    days = daily.index
    px = day_end_prices(grid, days).ffill().to_numpy()
    out = out or barrier_outcomes(daily, h4, hz)
    fund = funding_series(days, hz)
    m = Market(hz, days, px, fund, out, {}, {}, {})
    reprice(m, hz)
    return m


def reprice(m: Market, hz: Horizon) -> None:
    """(Re)compute exit days, funding and net returns for a cost model - cheap, no relabelling."""
    m.hz = hz
    c = hz.fee_rate + hz.slippage
    S = np.concatenate([[0.0], np.cumsum(m.funding * m.px)])  # S[j+1] = sum_{<=j} rate*px
    for side, o in m.out.items():
        et = o["exit_time"]
        exit_date = (et - pd.Timedelta(seconds=1)).dt.normalize().fillna(m.days[-1] + pd.Timedelta(days=1))
        k = m.days.searchsorted(pd.DatetimeIndex(exit_date).as_unit(m.days.unit))
        k = np.where(o["gross"].notna().to_numpy(), k, -1)
        m.exit_day[side] = k
        i = np.arange(len(m.days))
        use_f = hz.funding_enabled and side in hz.funding_sides
        F = np.where(k >= 0, SIDE_MULT[side] * (S[np.clip(k, 0, None) + 1] - S[i + 1]) / o["entry"].to_numpy(), np.nan)
        F = F if use_f else np.where(k >= 0, 0.0, np.nan)
        m.funding_paid[side] = F
        m.net[side] = (1 - c) ** 2 * (1 + o["gross"].to_numpy() + F) - 1


@dataclass
class SimResult:
    trades: pd.DataFrame
    daily: pd.Series        # pre-tax daily returns of the sleeve
    start: pd.Timestamp
    end: pd.Timestamp


def _value(side: str, p: float, e: float) -> float:
    return p / e if side == "long" else 2 - p / e


def simulate(m: Market, sig: dict[str, np.ndarray], pref: dict[str, np.ndarray],
             start, end, sides: tuple[str, ...] = ("long", "short")) -> SimResult:
    """Walk the days in [start, end]; when flat and out of cooldown, take the signalled side
    (the higher `pref` if both fire). Trades still open at `end` are closed at that day's price."""
    hz = m.hz
    c = hz.fee_rate + hz.slippage
    i0, i1 = m.pos(start), min(m.pos(end), len(m.days) - 1)
    if m.days[min(i1, len(m.days) - 1)] > pd.Timestamp(end):
        i1 -= 1
    short_ok = m.days >= hz.short_start
    r = np.zeros(len(m.days))
    trades, i, tid = [], i0, 0
    while i <= i1:
        cand = [s for s in sides if sig[s][i] and (s != "short" or short_ok[i])]
        if not cand:
            i += 1
            continue
        side = max(cand, key=lambda s: pref[s][i])
        o = m.out[side].iloc[i]
        e, k = o["entry"], m.exit_day[side][i]
        if not np.isfinite(e):
            i += 1
            continue
        if k < 0 or k > i1:  # unresolved, or exits after the window: close at the window end
            reason = "open" if k < 0 else "window_end"
            k = i1
            exit_px, hold = m.px[k], float(k - i)
        else:
            exit_px, reason, hold = o["exit_price"], o["reason"], float(o["hold_days"])
        if k <= i:
            i += 1
            continue
        mult = SIDE_MULT[side] if (hz.funding_enabled and side in hz.funding_sides) else 0.0
        v_prev, F = 1.0, 0.0
        r[i] = (1 + r[i]) * (1 - c) - 1
        for d in range(i + 1, k + 1):
            F += mult * m.funding[d] * m.px[d] / e
            p = exit_px if d == k else m.px[d]
            v = _value(side, p, e) + F
            r[d] = (v * (1 - c) if d == k else v) / v_prev - 1
            v_prev = v
        net = (1 - c) ** 2 * (_value(side, exit_px, e) + F) - 1
        trades.append({"trade_id": tid, "side": side, "entry_date": m.days[i], "entry_price": e,
                       "exit_time": o["exit_time"] if reason in ("target", "stop", "deadline") else m.days[k] + pd.Timedelta(days=1),
                       "exit_price": exit_px, "exit_reason": reason, "target": o["tw"],
                       "gross_ret": (exit_px / e - 1) if side == "long" else (1 - exit_px / e), "cost": 1 - (1 - c) ** 2,
                       "funding": F, "net_ret": net, "hold_days": hold, "mae": o["mae"], "mfe": o["mfe"],
                       "exit_day": m.days[k]})
        tid += 1
        i = k + max(1, hz.cooldown_days)
    daily = pd.Series(r, index=m.days).iloc[i0:i1 + 1]
    cols = ["trade_id", "side", "entry_date", "entry_price", "exit_time", "exit_price", "exit_reason", "target",
            "gross_ret", "cost", "funding", "net_ret", "hold_days", "mae", "mfe", "exit_day"]
    return SimResult(pd.DataFrame(trades, columns=cols), daily, m.days[i0], m.days[i1])


def after_tax_equity(res: SimResult, rate: float, year_start_month: int = 3) -> pd.Series:
    """Equity path after tax at `rate` on net realized P&L per tax year (losses carried forward)."""
    eq_pre = (1 + res.daily).cumprod()
    if res.trades.empty:
        return eq_pre
    days = res.daily.index
    pnl_frac = {}
    for t in res.trades.itertuples():
        pnl_frac.setdefault(t.exit_day, []).append((t.entry_date, t.net_ret))
    E, out, realized, carry = 1.0, np.empty(len(days)), 0.0, 0.0
    pre_entry = {}
    for j, d in enumerate(days):
        E *= 1 + res.daily.iloc[j]
        out[j] = E
        if d in pnl_frac:
            for entry_date, net in pnl_frac[d]:
                base = pre_entry.get(entry_date)
                if base is None:
                    base = E / (1 + net)
                realized += base * net
        nxt = d + pd.Timedelta(days=1)
        year_end = nxt.month == year_start_month and nxt.day == 1
        if year_end or j == len(days) - 1:
            taxable = realized + carry
            tax = rate * taxable if taxable > 0 else 0.0
            carry = min(taxable, 0.0)
            realized = 0.0
            E -= tax
            out[j] = E
        # equity just before the next day's entries (the base for a trade's P&L)
        pre_entry[nxt] = E
    return pd.Series(out, index=days)
