"""Next-extreme forecaster: when, and at what price, the next cycle low (or top) is likely.

This is a separate statistical cycle-analog model, not the neural net: with only 4-5 past
tops/lows, a trained network cannot learn timing or price levels. Several estimators, each
based on how past cycles behaved relative to an anchor, are combined into a Gaussian for the
date and a Gaussian for log price:

Timing (days)                        Price (log USD)
- previous extreme + typical gap     - size of the move from the previous extreme (with decay trend)
- next/last halving +/- typical gap  - MVRV at past extremes x current realized price
                                     - price/200-week MA at past extremes x projected 200-week MA

Each estimator's spread is the sample spread across the analog cycles, inflated for the small
sample; the combination adds the disagreement between estimators (mixture variance).
Hindcasts show how this method would have done on past extremes using only earlier cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import halvings, load_config
from .data import Extreme, cycle_extremes


@dataclass
class Estimate:
    name: str
    mu: float      # days since epoch (time) or ln price (price)
    sd: float
    n: int
    note: str = ""


@dataclass
class Forecast:
    kind: str                          # "low" | "top"
    asof: pd.Timestamp
    after: Extreme                     # the extreme this forecast follows
    date_mu: pd.Timestamp
    date_sd: float                     # days
    logp_mu: float
    logp_sd: float
    time_estimates: list[Estimate] = field(default_factory=list)
    price_estimates: list[Estimate] = field(default_factory=list)

    @property
    def price_mu(self) -> float:
        return float(np.exp(self.logp_mu))

    def price_interval(self, z: float = 1.0) -> tuple[float, float]:
        return float(np.exp(self.logp_mu - z * self.logp_sd)), float(np.exp(self.logp_mu + z * self.logp_sd))

    def date_interval(self, z: float = 1.0) -> tuple[pd.Timestamp, pd.Timestamp]:
        d = pd.Timedelta(days=z * self.date_sd)
        return self.date_mu - d, self.date_mu + d

    @property
    def p_already(self) -> float:
        """Probability mass of the date Gaussian that lies before the as-of date."""
        return float(norm.cdf((self.asof - self.date_mu).days / self.date_sd))


def _cfg() -> dict:
    return {"analog_cycles": 3, "min_date_sd_days": 30, "min_logp_sd": 0.15, "halving_sd_days": 20,
            **load_config().get("forecast", {})}


def _day(t: pd.Timestamp) -> float:
    return (t - pd.Timestamp("1970-01-01")).days


def _from_day(x: float) -> pd.Timestamp:
    return pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(x))


def _mean_est(name: str, values: list[float], anchor: float, sign: float = 1.0, extra_var: float = 0.0,
              note: str = "") -> Estimate | None:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return None
    sd = v.std(ddof=1) * np.sqrt(1 + 1 / len(v))
    return Estimate(name, anchor + sign * v.mean(), float(np.sqrt(sd ** 2 + extra_var)), len(v), note)


def _trend_est(name: str, values: list[float], anchor_fn, note: str = "") -> Estimate | None:
    """Log-linear decay trend across cycles, extrapolated one cycle ahead."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3 or np.any(v <= 0):
        return None
    x = np.arange(len(v))
    lv = np.log(v)
    b, a = np.polyfit(x, lv, 1)
    resid = lv - (a + b * x)
    s = np.sqrt((resid ** 2).sum() / max(len(v) - 2, 1))
    xn = len(v)
    se = s * np.sqrt(1 + 1 / len(v) + (xn - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    pred = float(np.exp(a + b * xn))
    mu = anchor_fn(pred)
    # translate the log-scale uncertainty of the predicted quantity into log price
    lo, hi = anchor_fn(pred * np.exp(-se)), anchor_fn(pred * np.exp(se))
    return Estimate(name, mu, float(abs(hi - lo) / 2), len(v), note)


def _combine(ests: list[Estimate], floor: float) -> tuple[float, float]:
    mus = np.array([e.mu for e in ests])
    var = np.array([e.sd ** 2 for e in ests])
    mu = mus.mean()
    sd = np.sqrt(var.mean() + (mus.var() if len(mus) > 1 else 0.0))
    return float(mu), float(max(sd, floor))


def forecast_next(daily: pd.DataFrame, feats: pd.DataFrame, extremes: list[Extreme] | None = None,
                  asof: pd.Timestamp | None = None) -> Forecast | None:
    """Forecast the extreme following the last extreme known at `asof` (hindsight-confirmed list)."""
    c = _cfg()
    close = daily["close"]
    asof = asof or close.index[-1]
    ex = [e for e in (extremes or cycle_extremes(close)) if e.date <= asof]
    if len(ex) < 3:
        return None
    last = ex[-1]
    kind = "low" if last.kind == "top" else "top"
    # transitions of the same type as the one being forecast (top->low or low->top), most recent N
    pairs = [(a, b) for a, b in zip(ex[:-1], ex[1:]) if a.kind == last.kind][-c["analog_cycles"]:]
    hs = pd.DatetimeIndex(halvings() + [pd.Timestamp(load_config()["next_halving_estimate"])])

    # ---- timing ----
    t_ests = []
    gaps = [(b.date - a.date).days for a, b in pairs]
    e = _mean_est(f"{last.kind} + typical gap", gaps, _day(last.date), note=f"gaps {gaps} days")
    if e:
        t_ests.append(e)
    if kind == "low":
        rel = [(hs[hs > b.date].min() - b.date).days for _, b in pairs]
        nh = hs[hs > last.date].min()
        e = _mean_est("next halving − typical lead", rel, _day(nh), sign=-1, extra_var=c["halving_sd_days"] ** 2,
                      note=f"lows came {rel} days before a halving")
    else:
        # the halving between the last low and the coming top (scheduled, so known in advance)
        rel = [(b.date - hs[(hs > a.date) & (hs <= b.date)].max()).days for a, b in pairs
               if ((hs > a.date) & (hs <= b.date)).any()]
        nh = hs[hs > last.date].min()
        e = _mean_est("halving + typical lag", rel, _day(nh), extra_var=c["halving_sd_days"] ** 2 if nh > asof else 0.0,
                      note=f"tops came {rel} days after a halving")
    if e:
        t_ests.append(e)
    if not t_ests:
        return None
    t_mu, t_sd = _combine(t_ests, c["min_date_sd_days"])
    date_mu = _from_day(t_mu)
    horizon_years = max((date_mu - asof).days, 0) / 365

    # ---- price ----
    p_ests = []
    lnmove = [abs(np.log(b.price / a.price)) for a, b in zip(ex[:-1], ex[1:]) if a.kind == last.kind][-4:]
    sgn = -1 if kind == "low" else 1
    e = _trend_est("move from last extreme (decaying)", lnmove, lambda m: np.log(last.price) + sgn * m,
                   note="size of past moves: " + ", ".join(f"{np.exp(m):.1f}x" for m in lnmove))
    if e:
        p_ests.append(e)
    now = feats.loc[:asof].iloc[-1]
    price_now = close.loc[:asof].iloc[-1]
    rp = price_now / np.exp(now["mvrv"])            # realized price
    ma200w = price_now / np.exp(now["p_200w"])      # 200-week MA
    ma_series = (close / np.exp(feats["p_200w"])).loc[:asof]
    ma_growth = np.log(ma_series.iloc[-1] / ma_series.iloc[-366]) if len(ma_series) > 366 else 0.0
    ma_at = ma200w * np.exp(ma_growth * horizon_years)
    mvrv_at = [np.exp(feats["mvrv"].get(b.date, np.nan)) for _, b in pairs]
    p200_at = [np.exp(feats["p_200w"].get(b.date, np.nan)) for _, b in pairs]
    mvrv_at = [float(v) for v in mvrv_at if np.isfinite(v)]
    p200_at = [float(v) for v in p200_at if np.isfinite(v)]
    if kind == "low":
        e = _mean_est("realized price × MVRV at past lows", list(np.log(mvrv_at)), np.log(rp),
                      note=f"MVRV at lows {[round(v, 2) for v in mvrv_at]}, realized price ${rp:,.0f}")
        if e:
            p_ests.append(e)
        e = _mean_est("200-week MA × ratio at past lows", list(np.log(p200_at)), np.log(ma_at),
                      note=f"price/200wMA at lows {[round(v, 2) for v in p200_at]}, projected 200wMA ${ma_at:,.0f}")
        if e:
            p_ests.append(e)
    else:
        all_tops = [(a, b) for a, b in zip(ex[:-1], ex[1:]) if a.kind == last.kind][-4:]
        mv = [np.exp(feats["mvrv"].get(b.date, np.nan)) for _, b in all_tops]
        pw = [np.exp(feats["p_200w"].get(b.date, np.nan)) for _, b in all_tops]
        mv = [float(v) for v in mv if np.isfinite(v)]
        pw = [float(v) for v in pw if np.isfinite(v)]
        e = _trend_est("realized price × MVRV at tops (decaying)", mv, lambda m: np.log(rp) + np.log(m),
                       note=f"MVRV at tops {[round(v, 2) for v in mv]}")
        if e:
            p_ests.append(e)
        e = _trend_est("200-week MA × ratio at tops (decaying)", pw, lambda m: np.log(ma_at) + np.log(m),
                       note=f"price/200wMA at tops {[round(v, 2) for v in pw]}")
        if e:
            p_ests.append(e)
    if not p_ests:
        return None
    p_mu, p_sd = _combine(p_ests, c["min_logp_sd"])
    return Forecast(kind, asof, last, date_mu, t_sd, p_mu, p_sd, t_ests, p_ests)


def hindcasts(daily: pd.DataFrame, feats: pd.DataFrame, confirm_days: int = 180) -> pd.DataFrame:
    """Forecast each past extreme using only earlier cycles, as of `confirm_days` after the previous one."""
    close = daily["close"]
    ex = cycle_extremes(close)
    rows = []
    for i in range(3, len(ex)):
        target, prev = ex[i], ex[i - 1]
        asof = prev.date + pd.Timedelta(days=confirm_days)
        if asof >= target.date:
            continue
        f = forecast_next(daily, feats, ex[:i], asof=asof)
        if f is None or len(f.time_estimates) == 0:
            continue
        rows.append({
            "extreme": f"{target.kind} {target.date:%Y-%m}", "forecast made": asof.strftime("%Y-%m-%d"),
            "forecast date": f.date_mu.strftime("%Y-%m-%d"), "actual date": target.date.strftime("%Y-%m-%d"),
            "date error (days)": (target.date - f.date_mu).days, "date z": (target.date - f.date_mu).days / f.date_sd,
            "forecast price": f.price_mu, "actual price": target.price,
            "price error": float(np.exp(np.log(target.price) - f.logp_mu) - 1),
            "price z": (np.log(target.price) - f.logp_mu) / f.logp_sd,
        })
    return pd.DataFrame(rows)


def model_forecast(score: pd.Series, daily: pd.DataFrame, extremes: list[Extreme] | None = None,
                   asof: pd.Timestamp | None = None) -> Forecast | None:
    """A forecast derived from one model's own score history (nearest-analog matching).

    Today's reading is matched against the same model's readings during the equivalent phase of
    past cycles — the decline from the previous top when forecasting a low, the rise from the
    previous low when forecasting a top: "the last time this model read 0.21 on the way down, the
    low was N days away and X% below." The spread across analog cycles becomes the uncertainty.
    """
    c = _cfg()
    close = daily["close"]
    asof = asof or min(score.dropna().index[-1], close.index[-1])
    ex = [e for e in (extremes or cycle_extremes(close)) if e.date <= asof]
    if len(ex) < 3 or score.dropna().empty:
        return None
    last = ex[-1]
    kind = "low" if last.kind == "top" else "top"
    s = score.dropna().loc[:asof]
    if s.empty:
        return None
    s_now = float(s.iloc[-1])
    # analog phases: (previous extreme -> target extreme) pairs of the same shape as now
    phases = [(a, b) for a, b in zip(ex[:-1], ex[1:]) if a.kind == last.kind][-c["analog_cycles"]:]
    days, gaps, notes = [], [], []
    for a, b in phases:
        win = s.loc[a.date:b.date]
        if len(win) < 30:
            continue
        diff = (win - s_now).abs()
        tol = max(0.02, 0.08 * float(win.max() - win.min()))
        near = win.index[diff <= max(tol, float(diff.min()))]
        d = pd.Timestamp(np.asarray(near, dtype="datetime64[ns]")[len(near) // 2]) if len(near) else diff.idxmin()
        days.append((b.date - d).days)
        gaps.append(float(np.log(b.price / close.loc[d])))
        notes.append(f"{b.date:%Y-%m} {b.kind}: matched {d:%Y-%m-%d} (score {win.loc[d]:+.2f}), "
                     f"{(b.date - d).days} days and {np.exp(gaps[-1]) - 1:+.0%} away")
    if len(days) < 2:
        return None
    t_est = _mean_est("score analogs: days to extreme", days, _day(asof), note="; ".join(notes))
    p_est = _mean_est("score analogs: price move to extreme", gaps, float(np.log(close.loc[:asof].iloc[-1])),
                      note=f"today's score {s_now:+.2f}; moves " + ", ".join(f"{np.exp(g) - 1:+.0%}" for g in gaps))
    if t_est is None or p_est is None:
        return None
    t_mu, t_sd = _combine([t_est], c["min_date_sd_days"])
    p_mu, p_sd = _combine([p_est], c["min_logp_sd"])
    return Forecast(kind, asof, last, _from_day(t_mu), t_sd, p_mu, p_sd, [t_est], [p_est])
