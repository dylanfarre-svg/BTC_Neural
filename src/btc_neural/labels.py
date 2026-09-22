"""Training targets in [-1, +1]: +1 = cycle low (max buy), -1 = cycle top (max sell).

Labels use hindsight by definition, so they are always computed *as of a cutoff date*:
only prices up to the cutoff are used, and only extremes confirmed by the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .data import Extreme


def cycle_label(close: pd.Series, extremes: list[Extreme], cutoff: pd.Timestamp) -> pd.Series:
    """Log-position between the surrounding cycle low and top, mapped to [-1, 1].

    Rising leg (low -> top): y = 1 - 2 * ln(P/L) / ln(T/L)
    Falling leg (top -> low): same formula with the leg's own top and low.
    Rows after the last extreme usable at the cutoff are left unlabeled (NaN).
    """
    confirm = pd.Timedelta(days=load_config()["extreme_confirm_days"])
    usable = [e for e in extremes if e.date + confirm <= cutoff]
    y = pd.Series(np.nan, index=close.index)
    lc = np.log(close)
    for a, b in zip(usable[:-1], usable[1:]):
        top, low = (a, b) if a.kind == "top" else (b, a)
        seg = slice(a.date, b.date)
        pos = (lc.loc[seg] - np.log(low.price)) / (np.log(top.price) - np.log(low.price))
        y.loc[seg] = 1 - 2 * pos.clip(0, 1)
    return y.loc[:cutoff]


def window_label(close: pd.Series, cutoff: pd.Timestamp, half_days: int = 365) -> pd.Series:
    """Log-position within a centred window of +/- half_days, using prices up to the cutoff."""
    c = np.log(close.loc[:cutoff])
    hi = c[::-1].rolling(half_days + 1, min_periods=1).max()[::-1]
    lo = c[::-1].rolling(half_days + 1, min_periods=1).min()[::-1]
    hi = np.maximum(hi, c.rolling(half_days + 1, min_periods=1).max())
    lo = np.minimum(lo, c.rolling(half_days + 1, min_periods=1).min())
    pos = (c - lo) / (hi - lo).replace(0, np.nan)
    y = 1 - 2 * pos
    # need at least 180 days of forward data for a meaningful label
    y.iloc[-180:] = np.nan
    return y


def make_label(close: pd.Series, extremes: list[Extreme], cutoff: pd.Timestamp,
               cfg: dict | None = None) -> pd.Series:
    cfg = cfg or load_config()["label"]
    if cfg["kind"] == "window":
        return window_label(close, cutoff, cfg["window_half_days"])
    return cycle_label(close, extremes, cutoff)
