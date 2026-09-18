"""Signal-quality metrics: do strong signals land near the cycle extremes?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import Extreme
from ..labels import cycle_label


def _side_metrics(side: str, close: pd.Series, strength: pd.Series, ext: Extreme, pos: pd.Series) -> dict:
    ev = load_config()["evaluation"]
    sign = -1 if side == "sell" else 1
    s10 = strength[strength * sign >= 10].index
    strong = strength[strength * sign >= ev["strong_level"]].index
    lp = np.log(close)
    dist = (np.log(ext.price) - lp) * (1 if side == "sell" else -1)  # >= 0 in log units
    m = {f"{side}_n10": float(len(s10)), f"{side}_n_strong": float(len(strong))}
    if len(s10):
        m[f"{side}10_log_err"] = float(dist.loc[s10].median())
        m[f"{side}10_best_log_err"] = float(dist.loc[s10].min())
        m[f"{side}10_days_err"] = float(np.median(np.abs((s10 - ext.date).days)))
        m[f"{side}10_hit"] = float((dist.loc[s10].abs() <= ev["hit_log_distance"]).mean())
    if len(strong):
        m[f"{side}_strong_hit"] = float((dist.loc[strong].abs() <= ev["hit_log_distance"]).mean())
        wrong = pos.loc[strong] > 0 if side == "sell" else pos.loc[strong] < 0
        m[f"{side}_strong_wrong"] = float(wrong.mean())
        m[f"{side}_strong_log_err"] = float(dist.loc[strong].median())
    err = m.get(f"{side}10_log_err", 1.0)
    wrong = m.get(f"{side}_strong_wrong", 1.0)
    budget = abs(np.log((len(s10) + 1) / 11))
    m[f"{side}_score"] = float(err + wrong + 0.1 * budget)
    return m


def evaluate_period(close: pd.Series, score: pd.Series, strength: pd.Series, extremes: list[Extreme],
                    start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Metrics for one test period. Lower *_score is better."""
    c = close.loc[start:end]
    s = strength.reindex(c.index).fillna(0)
    lo, hi = np.log(c.min()), np.log(c.max())
    # Hindsight cycle position in [-1 (top), +1 (low)]: position within the surrounding
    # low->top or top->low leg; falls back to the period's range where a leg is incomplete.
    leg = cycle_label(close, extremes, close.index[-1] + pd.Timedelta(days=10_000)).reindex(c.index)
    pos = leg.fillna(1 - 2 * (np.log(c) - lo) / (hi - lo))
    m: dict[str, float] = {"days": float(len(c))}
    tops = [e for e in extremes if start <= e.date <= end and e.kind == "top"]
    lows = [e for e in extremes if start <= e.date <= end and e.kind == "low"]
    sides = []
    if tops:
        m.update(_side_metrics("sell", c, s, tops[-1], pos))
        sides.append(m["sell_score"])
    if lows:
        m.update(_side_metrics("buy", c, s, lows[-1], pos))
        sides.append(m["buy_score"])
    # rank correlation between score and the hindsight cycle position
    sc = score.reindex(c.index)
    ok = sc.notna()
    if ok.sum() > 30:
        m["spearman_vs_position"] = float(sc[ok].rank().corr(pos[ok].rank()))
    m["fold_score"] = float(np.mean(sides)) if sides else np.nan
    return m
