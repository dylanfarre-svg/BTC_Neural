"""Canonical daily price frame and cycle reference data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import db
from .config import halvings, load_config


def daily_prices() -> pd.DataFrame:
    """Daily OHLC indexed by UTC date.

    Close is the Coin Metrics reference rate (a multi-exchange price, robust in the thin
    2010-2012 markets), falling back to Bitstamp when Coin Metrics has not published yet.
    High/low come from Bitstamp, widened to contain the close.
    """
    d = db.read_candles("1d")
    cm = db.read_series(["PriceUSD"])
    df = d[["open", "high", "low", "close", "volume"]].copy()
    if not cm.empty:
        df = df.join(cm["PriceUSD"], how="outer")
        df["close"] = df["PriceUSD"].fillna(df["close"])
        df = df.drop(columns="PriceUSD")
    df["open"] = df["open"].fillna(df["close"].shift(1)).fillna(df["close"])
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df.dropna(subset=["close"])


@dataclass(frozen=True)
class Extreme:
    kind: str          # "top" | "low"
    date: pd.Timestamp
    price: float


def cycle_extremes(close: pd.Series) -> list[Extreme]:
    """Snap the configured (approximate) extreme dates to the actual max/min close nearby.

    HINDSIGHT DATA: only for labels and evaluation, never as a model input.
    """
    cfg = load_config()
    w = pd.Timedelta(days=cfg["extreme_search_days"])
    out = []
    for e in cfg["extremes"]:
        d = pd.Timestamp(e["date"])
        win = close.loc[d - w: d + w]
        if win.empty:
            continue
        idx = win.idxmax() if e["kind"] == "top" else win.idxmin()
        out.append(Extreme(e["kind"], idx, float(close.loc[idx])))
    return out


def epochs(end: pd.Timestamp, start: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Halving epochs E0..En as (name, start, end_exclusive)."""
    bounds = [start] + [h for h in halvings() if start < h < end] + [end + pd.Timedelta(days=1)]
    return [(f"E{i}", bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def epoch_of(dates: pd.DatetimeIndex) -> np.ndarray:
    hs = np.array(halvings(), dtype="datetime64[ns]")
    return np.searchsorted(hs, dates.values, side="right")
