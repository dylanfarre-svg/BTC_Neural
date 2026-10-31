"""Evaluation for the trading signals: trade statistics, benchmarks, significance.

Everything is computed per window (full 2015+, modern 2019+, holdout) and never pooled across
them. The three buy-and-hold comparisons (PLAN_SWING.md Q8) are total return, Sharpe, and return
per unit of time in market. Buy-and-hold is charged capital gains tax on liquidation at the end of
the window - the same liquidation basis the strategy's tax uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..data import cycle_extremes
from .horizons import Horizon
from .simulator import Market, SimResult, after_tax_equity

ANN = 365.0


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def sortino(r: pd.Series) -> float:
    dn = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / dn * np.sqrt(ANN)) if dn > 0 else 0.0


def max_drawdown(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def in_market_days(res: SimResult) -> float:
    if res.trades.empty:
        return 0.0
    return float((res.trades.exit_day - res.trades.entry_date).dt.days.clip(lower=0).sum())


def rptim(final: float, days_in_market: float) -> float:
    """Return per unit of time in market: annualised log return over the days actually invested."""
    return float(np.log(max(final, 1e-9)) / (days_in_market / ANN)) if days_in_market > 0 else 0.0


def buy_and_hold(m: Market, start, end, capital_rate: float) -> dict:
    i0, i1 = m.pos(start), m.pos(end)
    i1 = min(i1, len(m.days) - 1) - (1 if m.days[min(i1, len(m.days) - 1)] > pd.Timestamp(end) else 0)
    px = pd.Series(m.px, index=m.days).iloc[max(i0 - 1, 0):i1 + 1]
    r = px.pct_change().dropna()
    g = float(px.iloc[-1] / px.iloc[0])
    after = 1 + (g - 1) * (1 - capital_rate) if g > 1 else g
    days = float(len(r))
    return {"total": g, "total_after_tax": after, "sharpe": sharpe(r), "sortino": sortino(r),
            "max_dd": max_drawdown(px / px.iloc[0]), "rptim": rptim(after, days), "rptim_pre_tax": rptim(g, days)}


def regimes(m: Market, feats_rv30: pd.Series, start, end) -> pd.Series:
    """Four evaluation buckets (hindsight allowed - evaluation only): cycle leg (bull = low->top,
    bear = top->low) x volatility (30-day realised vol above/below its median in the window)."""
    close = pd.Series(m.px, index=m.days)
    leg = pd.Series("bull", index=m.days)
    ex = cycle_extremes(close.dropna())
    for a, b in zip(ex, ex[1:] + [None]):
        stop = b.date if b is not None else m.days[-1] + pd.Timedelta(days=1)
        leg.loc[(m.days > a.date) & (m.days <= stop)] = "bull" if a.kind == "low" else "bear"
    rv = feats_rv30.reindex(m.days)
    w = (m.days >= pd.Timestamp(start)) & (m.days <= pd.Timestamp(end))
    hi = rv > rv[w].median()
    return (leg + np.where(hi, "_hivol", "_lovol")).where(w)


def trade_stats(t: pd.DataFrame, years: float) -> dict:
    if t.empty:
        return {"n_trades": 0, "trades_per_year": 0.0}
    wins, losses = t.net_ret[t.net_ret > 0].sum(), -t.net_ret[t.net_ret < 0].sum()
    out = {
        "n_trades": int(len(t)), "trades_per_year": len(t) / years,
        "n_long": int((t.side == "long").sum()), "n_short": int((t.side == "short").sum()),
        "hit_rate": float((t.net_ret > 0).mean()), "target_rate": float((t.exit_reason == "target").mean()),
        "stop_rate": float((t.exit_reason == "stop").mean()), "deadline_rate": float((t.exit_reason == "deadline").mean()),
        "mean_net": float(t.net_ret.mean()), "median_net": float(t.net_ret.median()),
        "mean_gross": float(t.gross_ret.mean()), "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "median_hold": float(t.hold_days.median()), "mean_hold": float(t.hold_days.mean()),
        "hold_in_band": float(((t.hold_days >= 7) & (t.hold_days <= 28)).mean()),
        "median_mae": float(t.mae.median()), "median_mfe": float(t.mfe.median()),
        "funding_total": float(t.funding.sum()), "mean_target": float(t.target.mean()),
    }
    for side in ("long", "short"):
        s = t[t.side == side]
        if len(s):
            out[f"{side}_mean_net"] = float(s.net_ret.mean())
            out[f"{side}_hit_rate"] = float((s.net_ret > 0).mean())
    return out


def bootstrap_mean_lb(x: np.ndarray, n: int = 5000, seed: int = 0, q: float = 0.05) -> float:
    if len(x) < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, q))


def window_metrics(m: Market, res: SimResult, hz: Horizon, rv30: pd.Series) -> dict:
    days = len(res.daily)
    years = days / ANN
    eq = (1 + res.daily).cumprod()
    dim = in_market_days(res)
    cap, rev = after_tax_equity(res, hz.capital_rate), after_tax_equity(res, hz.revenue_rate)
    out = {"start": str(res.start.date()), "end": str(res.end.date()), "years": years,
           **trade_stats(res.trades, years),
           "total_pre_tax": float(eq.iloc[-1]), "total_capital": float(cap.iloc[-1]), "total_revenue": float(rev.iloc[-1]),
           "cagr_pre_tax": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
           "sharpe": sharpe(res.daily), "sortino": sortino(res.daily), "max_dd": max_drawdown(eq),
           "time_in_market": dim / days if days else 0.0,
           "rptim_pre_tax": rptim(float(eq.iloc[-1]), dim), "rptim_capital": rptim(float(cap.iloc[-1]), dim),
           "rptim_revenue": rptim(float(rev.iloc[-1]), dim)}
    out["mean_net_lb95"] = bootstrap_mean_lb(res.trades.net_ret.to_numpy()) if len(res.trades) else np.nan
    bh = buy_and_hold(m, res.start, res.end, hz.capital_rate)
    out.update({f"bh_{k}": v for k, v in bh.items()})
    for tr in ("capital", "revenue"):
        out[f"beats_bh_total_{tr}"] = out[f"total_{tr}"] > bh["total_after_tax"]
        out[f"beats_bh_rptim_{tr}"] = out[f"rptim_{tr}"] > bh["rptim"]
    out["beats_bh_sharpe"] = out["sharpe"] > bh["sharpe"]
    if len(res.trades):
        reg = regimes(m, rv30, res.start, res.end)
        t = res.trades.assign(regime=reg.reindex(res.trades.entry_date).to_numpy())
        by = t.groupby("regime").net_ret.agg(["mean", "count"])
        out["regimes"] = {k: {"mean_net": float(v["mean"]), "n": int(v["count"])} for k, v in by.iterrows()}
        out["regimes_positive"] = int((by["mean"] > 0).sum())
    return out


def random_entry_bootstrap(m: Market, res: SimResult, sides: tuple[str, ...], n_draws: int = 1000,
                           seed: int = 0) -> dict:
    """Random entries with the strategy's trade count and long/short mix, same exits and cooldown.

    BTC drifts up, so a random long already makes money; this is the distribution a real edge has
    to escape. Compared on compounded net return and on mean net return per trade.
    """
    t = res.trades
    if t.empty:
        return {"p_total": np.nan, "p_mean": np.nan}
    i0, i1 = m.pos(res.start), m.pos(res.end)
    n_target = len(t)
    p_short = float((t.side == "short").mean()) if "short" in sides else 0.0
    short_ok = m.days >= m.hz.short_start
    rng = np.random.default_rng(seed)
    cd = max(1, m.hz.cooldown_days)

    def draw(p_enter: float) -> tuple[float, float, int]:
        i, logs, nets = i0, 0.0, []
        u = rng.random(i1 - i0 + 2)
        while i <= i1:
            if u[i - i0] < p_enter:
                side = "short" if (rng.random() < p_short and short_ok[i]) else "long"
                if side not in sides:
                    side = sides[0]
                k = m.exit_day[side][i]
                net = m.net[side][i]
                if k < 0 or k > i1 or not np.isfinite(net):
                    break
                nets.append(net)
                logs += np.log1p(net)
                i = k + cd
            else:
                i += 1
        return logs, (np.mean(nets) if nets else 0.0), len(nets)

    # calibrate the entry probability so the expected trade count matches the strategy's
    p = n_target / max(1, (i1 - i0) * 0.5)
    for _ in range(6):
        cnt = np.mean([draw(p)[2] for _ in range(60)])
        p = float(np.clip(p * n_target / max(cnt, 1), 1e-4, 1.0))
    sims = np.array([draw(p)[:2] for _ in range(n_draws)])
    strat_log = float(np.log1p(t.net_ret).sum())
    strat_mean = float(t.net_ret.mean())
    return {"p_total": float((sims[:, 0] >= strat_log).mean()), "p_mean": float((sims[:, 1] >= strat_mean).mean()),
            "rand_total_median": float(np.exp(np.median(sims[:, 0]))), "rand_total_p95": float(np.exp(np.quantile(sims[:, 0], 0.95))),
            "rand_mean_median": float(np.median(sims[:, 1])), "rand_mean_p95": float(np.quantile(sims[:, 1], 0.95))}


def deflated_sharpe(r: pd.Series, n_trials: int, sr_trials_var: float) -> float:
    """Probability the true Sharpe exceeds the best of `n_trials` unskilled ones
    (Bailey & Lopez de Prado 2014). Works on per-period Sharpe; inputs are daily returns."""
    r = r.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    emc = 0.5772156649
    v = max(sr_trials_var, 1e-12)
    n = max(n_trials, 2)
    sr0 = np.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / n) + emc * stats.norm.ppf(1 - 1 / (n * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))
