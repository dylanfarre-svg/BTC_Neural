"""Purged, embargoed walk-forward blocks.

Halving epochs (the cycle model's folds) are the wrong unit here: four folds, when the real hazard
is that consecutive days share most of their forward path. Instead: expanding-window training and
`block_days` test blocks. Training rows within `purge_days` of the test block are dropped, because
their labels run into the test period. The embargo removes rows just *after* a test block; with
forward-only training it never binds, and is kept so the same code serves k-fold experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .horizons import Horizon


@dataclass(frozen=True)
class Block:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp          # inclusive
    train_start: pd.Timestamp
    train_end: pd.Timestamp    # exclusive: start - purge


def blocks(hz: Horizon, score_from: pd.Timestamp, last: pd.Timestamp,
           train_start: pd.Timestamp | None = None, min_train_days: int | None = None) -> list[Block]:
    train_start = train_start or hz.train_start
    min_train = hz.min_train_days if min_train_days is None else min_train_days
    out, s = [], pd.Timestamp(score_from)
    while s <= last:
        e = min(s + pd.Timedelta(days=hz.block_days - 1), last)
        te = s - pd.Timedelta(days=hz.purge_days)
        if (te - train_start).days >= min_train:
            out.append(Block(f"B{s:%Y-%m-%d}", s, e, train_start, te))
        s = e + pd.Timedelta(days=1)
    return out


def train_mask(dates: pd.DatetimeIndex, b: Block, hz: Horizon, other_tests: list[Block] = ()) -> pd.Series:
    """Rows usable for training block `b`: inside [train_start, train_end), minus the purge and
    embargo around any other test block (only relevant to k-fold use)."""
    m = (dates >= b.train_start) & (dates < b.train_end)
    for t in other_tests:
        lo = t.start - pd.Timedelta(days=hz.purge_days)
        hi = t.end + pd.Timedelta(days=hz.embargo_days)
        m &= ~((dates >= lo) & (dates <= hi))
    return pd.Series(m, index=dates)
