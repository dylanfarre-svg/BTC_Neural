"""Map continuous model scores to the signed -10..+10 strength scale.

Thresholds are quantiles chosen so that, on the calibration scores, each strength level
fires on the configured number of days per 4-year cycle (e.g. strength 10 = 10 days).
Thresholds are frozen and then applied to unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config

CYCLE_DAYS = 1460.0


@dataclass
class Calibration:
    buy: dict[int, float] = field(default_factory=dict)    # level -> min score for >= level
    sell: dict[int, float] = field(default_factory=dict)   # level -> max score for >= level

    def to_dict(self) -> dict:
        return {"buy": {str(k): v for k, v in self.buy.items()}, "sell": {str(k): v for k, v in self.sell.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls({int(k): v for k, v in d["buy"].items()}, {int(k): v for k, v in d["sell"].items()})


def fit_calibration(scores: pd.Series, budget: dict[int, int] | None = None) -> Calibration:
    budget = budget or {int(k): v for k, v in load_config()["signal_budget"].items()}
    s = scores.dropna().to_numpy()
    n = len(s)
    cal = Calibration()
    for level, days in budget.items():
        frac = min(days / CYCLE_DAYS, 0.49)
        k = max(int(round(frac * n)), 1)
        cal.buy[level] = float(np.sort(s)[-k])      # top-k scores are buys
        cal.sell[level] = float(np.sort(s)[k - 1])  # bottom-k scores are sells
    return cal


def apply_calibration(scores: pd.Series, cal: Calibration) -> pd.Series:
    s = scores.to_numpy()
    out = np.zeros(len(s), dtype=int)
    for level in sorted(cal.buy):
        out = np.where((s >= cal.buy[level]) & (s > 0), np.maximum(out, level), out)
    for level in sorted(cal.sell):
        out = np.where((s <= cal.sell[level]) & (s < 0), np.minimum(out, -level), out)
    out = np.where(np.isnan(s), 0, out)
    return pd.Series(out, index=scores.index)


def apply_trailing(scores: pd.Series, budget: dict[int, int] | None = None, window: int = 1095,
                   minp: int = 365) -> pd.Series:
    """Causal adaptive calibration: rank each day's score against the model's own scores over
    the trailing `window` days (including today). Strength k fires when the score is in the
    top/bottom (budget_k / 1460) fraction of that window.

    This adapts to the shrinking amplitude of each cycle, which frozen thresholds cannot.
    `scores` must include enough history before the period of interest.
    """
    budget = budget or {int(k): v for k, v in load_config()["signal_budget"].items()}
    out = pd.Series(0, index=scores.index, dtype=int)
    roll = scores.rolling(window, min_periods=minp)
    for level in sorted(budget):
        frac = min(budget[level] / CYCLE_DAYS, 0.49)
        hi = roll.quantile(1 - frac)
        lo = roll.quantile(frac)
        out = out.where(~(scores >= hi), np.maximum(out, level))
        out = out.where(~(scores <= lo), np.minimum(out, -level))
    return out.where(scores.notna(), 0)


def combine(frozen: pd.Series, trailing: pd.Series) -> pd.Series:
    """Hybrid: the stronger of the two readings when they agree in sign, else the frozen one."""
    same = np.sign(frozen) * np.sign(trailing) >= 0
    stronger = np.where(np.abs(trailing) > np.abs(frozen), trailing, frozen)
    return pd.Series(np.where(same, stronger, frozen), index=frozen.index).astype(int)


def decluster(strength: pd.Series, score: pd.Series, mode: str = "none", gap_days: int = 60,
              improve: float = 0.05, cap: int = 10, window: int = 365, flat_days: int = 30,
              flat_eps: float = 0.0) -> pd.Series:
    """Thin out top-level signals, causally (each day decided from that day and earlier only).

    The models fire strength 10 for weeks during a parabolic run, far below the eventual top,
    because each new day is the most extreme reading so far. These rules demote such days to 9.

    - cooldown: after a 10 fires, no further 10 for `gap_days` unless the score is `improve`
      more extreme than the score at the last one that fired.
    - cap:      at most `cap` tens per trailing `window` days.
    - plateau:  a 10 only once the score has stopped intensifying - it is no more than `flat_eps`
      more extreme than it was `flat_days` ago (a top forms when conviction plateaus, whereas
      mid-parabola it is still deepening).
    """
    if mode in ("none", None):
        return strength
    s = strength.copy()
    sc = score.reindex(s.index)
    fired: list[tuple[int, float]] = []          # (position, score) of tens that were kept
    prev = sc.shift(flat_days)
    for i, (d, v) in enumerate(s.items()):
        if abs(int(v)) < 10:
            continue
        sign = 1 if v > 0 else -1
        cur = sc.iloc[i]
        keep = True
        if mode == "cooldown":
            same = [(j, q) for j, q in fired if np.sign(q) == sign]
            if same:
                j, q = same[-1]
                if i - j < gap_days and (cur - q) * sign < improve:
                    keep = False
        elif mode == "cap":
            recent = [j for j, q in fired if np.sign(q) == sign and i - j < window]
            keep = len(recent) < cap
        elif mode == "plateau":
            p = prev.iloc[i]
            keep = bool(np.isnan(p)) or (cur - p) * sign <= flat_eps
        if keep:
            fired.append((i, float(cur)))
        else:
            s.iloc[i] = sign * 9
    return s


def to_strength(scores_full: pd.Series, cal: Calibration, mode: str | None = None,
                declustering: dict | None = None) -> pd.Series:
    """Strength for every row of `scores_full` (which should include history for trailing mode)."""
    cfg = load_config()
    mode = mode or cfg["strength_mode"]
    dec = declustering if declustering is not None else cfg.get("declustering", {"mode": "none"})
    frozen = apply_calibration(scores_full, cal)
    if mode == "frozen":
        return decluster(frozen, scores_full, **dec)
    trailing = apply_trailing(scores_full, window=cfg["trailing_window"])
    if mode == "trailing":
        return decluster(trailing, scores_full, **dec)
    if mode == "both":
        same = np.sign(frozen) * np.sign(trailing) > 0
        weaker = np.where(np.abs(trailing) < np.abs(frozen), trailing, frozen)
        return decluster(pd.Series(np.where(same, weaker, 0), index=frozen.index).astype(int), scores_full, **dec)
    return decluster(combine(frozen, trailing), scores_full, **dec)
