"""Trade outcomes: triple-barrier labels walked on 4h bars, for long and short alike.

For a decision at the close of day t (00:00 UTC on t+1):
- entry = close of the 4h bar ending at that moment (Bitstamp, the same venue as the path);
- target distance `tw = max(floor, sigma_mult * sd_daily * sqrt(hold_max / 2))`, where sd_daily is
  the trailing `vol_window_days` daily log-return std known at t (14 days for the swing horizon);
- long:  target e*(1+tw), stop e*(1-stop_ratio*tw);  short: target e*(1-tw), stop e*(1+stop_ratio*tw);
- deadline after `hold_max_days`, exit at that bar's close.

Fills: a barrier fills at the barrier price, or at the bar's open if the bar gaps through it. When
both barriers fall inside one 4h bar the order is unknowable, so the stop is assumed (conservative).
A row whose window runs past the end of the data and has not yet touched a barrier is unresolved
(NaN) - it carries information only once the future arrives.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .horizons import Horizon

BARS_PER_DAY = 6  # 4h bars


def h4_grid(h4: pd.DataFrame) -> pd.DataFrame:
    """4h candles on a complete UTC grid: gaps become flat bars at the last close."""
    g = h4[["open", "high", "low", "close"]].copy()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="4h"))
    g["close"] = g["close"].ffill()
    for c in ("open", "high", "low"):
        g[c] = g[c].fillna(g["close"])
    return g


def day_end_prices(grid: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Price at each day's close (00:00 UTC next day) = close of the 20:00 bar."""
    return grid["close"].reindex(days + pd.Timedelta(hours=20))


def target_width(close: pd.Series, hz: Horizon) -> pd.Series:
    sd = np.log(close).diff().rolling(hz.vol_window_days, min_periods=hz.vol_window_days // 2).std()
    return np.maximum(hz.floor, hz.sigma_mult * sd * np.sqrt(hz.hold_max_days / 2))


def barrier_outcomes(daily: pd.DataFrame, h4: pd.DataFrame, hz: Horizon) -> dict[str, pd.DataFrame]:
    """Per decision day and side: exit time/price/reason, gross return, R multiple, MAE/MFE.

    Returns {"long": df, "short": df}, both indexed like `daily`, plus shared columns
    `entry` and `tw` on each.
    """
    grid = h4_grid(h4)
    days = daily.index
    L = int(round(hz.hold_max_days * BARS_PER_DAY))
    entry = day_end_prices(grid, days).to_numpy()
    tw = target_width(daily["close"], hz).to_numpy()
    # first path bar opens at t+1 00:00
    j0 = grid.index.get_indexer(days + pd.Timedelta(days=1))
    n, G = len(days), len(grid)
    idx = j0[:, None] + np.arange(L)[None, :]
    valid = (j0[:, None] >= 0) & (idx < G)
    safe = np.clip(idx, 0, G - 1)
    O, H, Lo, C = (np.where(valid, grid[c].to_numpy()[safe], np.nan) for c in ("open", "high", "low", "close"))
    complete = valid.all(axis=1)
    out = {}
    for side in ("long", "short"):
        if side == "long":
            tgt, stp = entry * (1 + tw), entry * (1 - hz.stop_ratio * tw)
            hit_t, hit_s = H >= tgt[:, None], Lo <= stp[:, None]
        else:
            tgt, stp = entry * (1 - tw), entry * (1 + hz.stop_ratio * tw)
            hit_t, hit_s = Lo <= tgt[:, None], H >= stp[:, None]
        kt = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), L)
        ks = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), L)
        k = np.minimum(kt, ks)
        stop_first = ks <= kt  # ties (same bar) -> stop
        hit = k < L
        resolved = hit | complete
        kk = np.where(hit, k, L - 1)
        rows = np.arange(n)
        o_k = O[rows, kk]
        if side == "long":
            px_stop = np.minimum(stp, o_k)       # gap down through the stop fills at the open
            px_tgt = np.maximum(tgt, o_k)        # gap up through the target fills at the open
        else:
            px_stop = np.maximum(stp, o_k)
            px_tgt = np.minimum(tgt, o_k)
        exit_px = np.where(hit, np.where(stop_first, px_stop, px_tgt), C[rows, kk])
        reason = np.where(hit, np.where(stop_first, "stop", "target"), "deadline")
        sign = 1.0 if side == "long" else -1.0
        gross = sign * (exit_px / entry - 1)
        # excursions up to and including the exit bar
        upto = np.arange(L)[None, :] <= kk[:, None]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # rows past the data end are all-NaN
            hmax = np.nanmax(np.where(upto, H, np.nan), axis=1)
            lmin = np.nanmin(np.where(upto, Lo, np.nan), axis=1)
        mfe = sign * ((hmax if side == "long" else lmin) / entry - 1)
        mae = sign * ((lmin if side == "long" else hmax) / entry - 1)
        exit_bar = np.where(j0 >= 0, j0 + kk, -1)
        exit_time = pd.DatetimeIndex(np.where(exit_bar >= 0, grid.index.to_numpy()[np.clip(exit_bar, 0, G - 1)],
                                              np.datetime64("NaT", "ns"))) + pd.Timedelta(hours=4)
        df = pd.DataFrame({
            "entry": entry, "tw": tw, "exit_time": exit_time, "exit_price": exit_px, "reason": reason,
            "gross": gross, "R": gross / tw, "hold_days": (kk + 1) / BARS_PER_DAY, "mae": mae, "mfe": mfe,
        }, index=days)
        bad = ~resolved | np.isnan(entry) | np.isnan(tw) | (j0 < 0)
        df.loc[bad, ["exit_price", "gross", "R", "hold_days", "mae", "mfe"]] = np.nan
        df.loc[bad, "reason"] = None
        df.loc[bad, "exit_time"] = pd.NaT
        out[side] = df
    return out


def training_target(o: pd.DataFrame, clip: float = 1.5) -> pd.Series:
    """What the models learn: the trade's R multiple (gross return in units of the target)."""
    return o["R"].clip(-clip, clip)


def uniqueness_weights(o: pd.DataFrame) -> pd.Series:
    """Average uniqueness of each label: 1/concurrency over the days its trade is open.

    Consecutive days share most of their forward path, so equal weights overstate the sample
    size several-fold; this down-weights rows in proportion to how much they overlap.
    """
    ok = o["R"].notna().to_numpy()
    n = len(o)
    span = np.where(ok, np.ceil(o["hold_days"].fillna(0).to_numpy()).astype(int), 0)
    start = np.arange(n) + 1
    end = np.minimum(start + span, n + 1)  # exclusive
    conc = np.zeros(n + 2)
    np.add.at(conc, start[ok], 1)
    np.add.at(conc, end[ok], -1)
    conc = np.cumsum(conc)
    inv = np.where(conc > 0, 1 / np.maximum(conc, 1), 0)
    csum = np.concatenate([[0], np.cumsum(inv)])
    w = np.where(ok & (span > 0), (csum[np.minimum(end, n + 1)] - csum[np.minimum(start, n + 1)]) / np.maximum(span, 1), np.nan)
    return pd.Series(w, index=o.index)
