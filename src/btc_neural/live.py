"""Daily job: ingest -> health check -> (re)train if due -> predict -> log -> notify."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import db, pipeline
from .config import ROOT, load_config
from .notify import Alert, dispatch

KEY_SOURCES = {"price": "close", "on-chain (MVRV)": "mvrv", "hash rate": "hash_ribbon"}


def fmt_strength(s: int) -> str:
    return "NEUTRAL 0" if s == 0 else (f"BUY {s}" if s > 0 else f"SELL {-s}")


def health(daily: pd.DataFrame, feats: pd.DataFrame, ingest: dict[str, str]) -> list[str]:
    problems = [f"{k}: {v}" for k, v in ingest.items() if v.startswith("error")]
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    stale = load_config()["live"]["stale_after_days"]
    for label, col in KEY_SOURCES.items():
        s = daily[col] if col in daily else feats[col]
        last = s.last_valid_index()
        if last is None or (today - last).days > stale:
            problems.append(f"{label} data is stale (last {last.date() if last is not None else 'never'})")
    return problems


BUY_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
SELL_RAMP = ["#fbd5d4", "#f8c2c1", "#f5aeac", "#f19a98", "#ec8583", "#e76f6d", "#e34948", "#c93b3a", "#a82f2e", "#862423"]


def make_chart(daily: pd.DataFrame, feats: pd.DataFrame, hist: pd.DataFrame, out: Path, years: int = 6,
               show_forecast: bool = True, model_scores: pd.Series | None = None) -> Path:
    """PNG version of the dashboard Home chart: price + signals + next-extreme forecast, strength panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .data import cycle_extremes
    from .forecast import forecast_next, model_forecast

    today = daily.index[-1]
    start = today - pd.Timedelta(days=365 * years)
    c = daily["close"].loc[start:]
    s = hist["strength"].reindex(c.index).fillna(0).astype(int)
    fc = forecast_next(daily, feats) if show_forecast else None
    mfc = model_forecast(model_scores, daily) if (show_forecast and model_scores is not None and len(model_scores)) else None
    x_end = today + pd.Timedelta(days=60)
    if fc is not None:
        x_end = max(x_end, fc.date_mu + pd.Timedelta(days=3.2 * fc.date_sd + 25 * years + 45))
    if mfc is not None:
        x_end = max(x_end, mfc.date_mu + pd.Timedelta(days=1.5 * mfc.date_sd + 25 * years))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.6), dpi=130, sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.06})
    ax.plot(c.index, c.values, color="#6f6e6a", lw=1.2, label="BTC/USD", zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k" if v >= 1000 else f"${v:,.0f}"))
    for cond, color, lab, marker, size in (
        (s >= 10, "#2a78d6", "Buy 10", "^", 34), ((s >= 7) & (s < 10), "#86b6ef", "Buy 7-9", "^", 22),
        (s <= -10, "#e34948", "Sell 10", "v", 34), ((s <= -7) & (s > -10), "#f0a09f", "Sell 7-9", "v", 22)):
        if cond.any():
            ax.scatter(c.index[cond], c[cond], color=color, s=size, marker=marker, label=lab, zorder=3,
                       edgecolors="white", linewidths=0.4)
    for e in cycle_extremes(daily["close"]):
        if e.date >= start:
            ax.annotate(f"cycle {e.kind}", (e.date, e.price), xytext=(0, 14 if e.kind == "top" else -16),
                        textcoords="offset points", ha="center", fontsize=8, color="#444")

    if fc is not None:
        rgb = "#2a78d6" if fc.kind == "low" else "#e34948"
        zs = np.linspace(-3, 3, 41)
        dz = zs[1] - zs[0]
        for z in zs:
            a = 0.30 * float(np.exp(-z * z / 2))
            ax.axvspan(fc.date_mu + pd.Timedelta(days=(z - dz / 2) * fc.date_sd),
                       fc.date_mu + pd.Timedelta(days=(z + dz / 2) * fc.date_sd), color=rgb, alpha=a, lw=0, zorder=0)
            ax.axhspan(np.exp(fc.logp_mu + (z - dz / 2) * fc.logp_sd), np.exp(fc.logp_mu + (z + dz / 2) * fc.logp_sd),
                       color=rgb, alpha=0.75 * a, lw=0, zorder=0)
        ax.axvline(fc.date_mu, color=rgb, lw=1.2, ls="--", zorder=1)
        ax.axhline(fc.price_mu, color=rgb, lw=1.2, ls="--", zorder=1)
        ax.scatter([fc.date_mu], [fc.price_mu], marker="D", s=60, color=rgb, edgecolors="white", zorder=4,
                   label=f"Forecast next {fc.kind}")
        p1, p2 = fc.price_interval(1)
        ax.annotate(f"next cycle {fc.kind}\n≈ {fc.date_mu:%d %b %Y} ±{fc.date_sd:.0f}d\n${fc.price_mu / 1000:,.0f}k "
                    f"(${p1 / 1000:,.0f}k–${p2 / 1000:,.0f}k)",
                    (fc.date_mu, fc.price_mu), xytext=(-12, 16), textcoords="offset points", va="bottom", ha="right",
                    fontsize=8, color=rgb,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=rgb, lw=0.6, alpha=0.9))
    if mfc is not None:
        p1, p2 = mfc.price_interval(1)
        ax.errorbar([mfc.date_mu], [mfc.price_mu], xerr=[pd.Timedelta(days=mfc.date_sd)],
                    yerr=[[mfc.price_mu - p1], [p2 - mfc.price_mu]], fmt="D", ms=6, color="#1baf7a",
                    ecolor="#1baf7a", elinewidth=1.2, capsize=3, zorder=4,
                    label=f"model forecast: {mfc.date_mu:%b %Y}, ${mfc.price_mu / 1000:,.0f}k")
    for a_ in (ax, ax2):
        a_.axvline(today, color="#999", lw=0.8, ls=":")
    ax.text(today, ax.get_ylim()[0], " today", fontsize=7, color="#777", va="bottom", ha="right")

    colors = [BUY_RAMP[v - 1] if v > 0 else SELL_RAMP[-v - 1] if v < 0 else "#cccccc" for v in s]
    ax2.bar(s.index, s.values, width=1.0, color=colors, linewidth=0)
    ax2.set_ylim(-10.5, 10.5)
    ax2.set_yticks([-10, -5, 0, 5, 10], ["Sell 10", "Sell 5", "0", "Buy 5", "Buy 10"], fontsize=8)
    ax2.axhline(0, color="#999", lw=0.6)
    ax2.set_xlim(start, x_end)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a_ in (ax, ax2):
        a_.grid(alpha=0.2)
        a_.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"BTC_Neural {today:%Y-%m-%d}: {fmt_strength(int(s.iloc[-1]))} at ${c.iloc[-1]:,.0f}",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def send_chart(role: str = "production") -> Path:
    """Build today's chart and push it (with a short summary) to ntfy."""
    from .backtest import walkforward
    from .forecast import forecast_next

    daily, feats = walkforward.dataset()
    h = pipeline.signal_history(role, feats)
    if h is None:
        raise RuntimeError("no production model; run `btcn train-production`")
    date = daily.index[-1].strftime("%Y-%m-%d")
    src = pipeline.all_roles().set_index("role").loc[role, "source_run_id"]
    chart = make_chart(daily, feats, h, ROOT / "data" / "charts" / f"home_{date}.png",
                       model_scores=pipeline.oos_scores(src))
    fc = forecast_next(daily, feats)
    msg = f"{date}  BTC ${daily['close'].iloc[-1]:,.0f}\nSignal: {fmt_strength(int(h['strength'].iloc[-1]))}"
    if fc is not None:
        p1, p2 = fc.price_interval(1)
        msg += (f"\nForecast next cycle {fc.kind}: {fc.date_mu:%Y-%m-%d} +/-{fc.date_sd:.0f}d, "
                f"${fc.price_mu:,.0f} (68%: ${p1:,.0f}-${p2:,.0f})")
    dispatch(date, [Alert("chart", "BTC_Neural chart", msg, 3, "chart_with_upwards_trend", chart)], force=True)
    return chart


def build_alerts(date: str, price: float, prod: dict, ref: dict | None, recent: pd.Series,
                 problems: list[str], chart: Path | None) -> list[Alert]:
    cfg = load_config()["notify"]
    s = prod["strength"]
    drivers = ", ".join(f"{k} ({'sell' if v < 0 else 'buy'})" for k, v in list(prod["drivers"].items())[:3])
    conf = prod["confidence"]
    conf_lbl = "high" if conf >= 0.66 else "medium" if conf >= 0.33 else "low"
    body = (f"{date}  BTC ${price:,.0f}\n"
            f"Signal: {fmt_strength(s)}  (score {prod['score']:+.2f}, confidence {conf_lbl})\n"
            f"Model: {prod['model']}\n"
            + (f"NN reference: {fmt_strength(ref['strength'])}\n" if ref else "")
            + f"Top drivers: {drivers}\nDashboard: {cfg['dashboard_url']}")
    alerts: list[Alert] = []
    side = "BUY" if s > 0 else "SELL"
    if abs(s) == 10:
        alerts.append(Alert("strength10", f"BTC {side} 10 - maximum strength", body, 5,
                            "rotating_light," + ("large_blue_circle" if s > 0 else "red_circle"), chart))
    elif abs(s) >= cfg["strong_level"]:
        alerts.append(Alert("strong", f"BTC {side} {abs(s)} - strong signal", body, 4, "warning", chart))
    if len(recent) >= 2:
        window = recent.iloc[-7:]
        prev = int(recent.iloc[-2])
        flipped = (prev <= -3 and s >= 3) or (prev >= 3 and s <= -3)
        if flipped or (window.max() - window.min()) >= cfg["regime_change_levels"]:
            alerts.append(Alert("regime", f"BTC signal shift: {fmt_strength(prev)} -> {fmt_strength(s)}", body, 3, "arrows_counterclockwise"))
    if ref and abs(ref["strength"] - s) >= cfg["nn_disagreement_levels"]:
        alerts.append(Alert("nn_disagree", f"NN reference disagrees: {fmt_strength(ref['strength'])} vs {fmt_strength(s)}",
                            body, 2, "test_tube"))
    if pd.Timestamp(date).weekday() == cfg["weekly_digest_weekday"]:
        wk = recent.iloc[-7:]
        digest = body + f"\n7-day range: {fmt_strength(int(wk.min()))} .. {fmt_strength(int(wk.max()))}\nData health: " + \
            ("OK" if not problems else "; ".join(problems))
        alerts.append(Alert("digest", f"BTC weekly digest: {fmt_strength(s)}", digest, 2, "calendar", chart))
    if problems:
        alerts.append(Alert("health", "BTC_Neural data problem", "\n".join(problems), 4, "x"))
    return alerts


def run_daily(notify: bool = True, force_notify: bool = False, verbose: bool = True) -> dict:
    from .backtest import walkforward
    from .ingest import ingest_all

    try:
        ingest = ingest_all(verbose=verbose)
        walkforward.dataset.cache_clear()
        daily, feats = walkforward.dataset()
        problems = health(daily, feats, ingest)

        retrain_days = load_config()["live"]["retrain_after_days"]
        for role in pipeline.ROLES:
            if pipeline.age_days(role) > retrain_days:
                try:
                    pipeline.train_production(role=role, verbose=verbose)
                except RuntimeError as e:
                    problems.append(f"{role}: {e}")
        # keep one trained model per model type, so every model's signal and forecast stay current
        cand = [r for r in pipeline.all_roles().role if r.startswith("candidate_")]
        if not cand or min(pipeline.age_days(r) for r in cand) > retrain_days:
            try:
                pipeline.train_all_candidates(verbose=verbose)
            except RuntimeError as e:
                problems.append(f"candidates: {e}")

        date = daily.index[-1].strftime("%Y-%m-%d")
        price = float(daily["close"].iloc[-1])
        out: dict[str, dict] = {}
        hists: dict[str, pd.DataFrame] = {}
        for role in pipeline.ROLES:
            h = pipeline.signal_history(role, feats)
            if h is None:
                continue
            hists[role] = h
            last = h.iloc[-1]
            drivers = pipeline.explain_today(role, feats)
            out[role] = {"strength": int(last.strength), "score": float(last.score),
                         "confidence": pipeline.confidence_from_spread(float(last.spread)),
                         "drivers": {k: float(v) for k, v in drivers.items()},
                         "model": f"{h.attrs['meta']['model_type']}/{h.attrs['meta']['feature_set']}", "run_id": h.attrs["run_id"]}
            with db.connect() as con:
                con.execute(
                    "INSERT OR IGNORE INTO live_signals (date, run_id, role, score, strength, confidence, price, drivers_json, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (date, h.attrs["run_id"], role, out[role]["score"], out[role]["strength"], out[role]["confidence"],
                     price, json.dumps(out[role]["drivers"]), db.now_utc()))
        if verbose:
            for role, v in out.items():
                print(f"  {date} {role}: {fmt_strength(v['strength'])} (score {v['score']:+.3f}) [{v['model']}]")
            for p in problems:
                print(f"  PROBLEM: {p}")
        sent: list[str] = []
        if notify and "production" in out:
            src = pipeline.all_roles().set_index("role").loc["production", "source_run_id"]
            chart = make_chart(daily, feats, hists["production"], ROOT / "data" / "charts" / f"signal_{date}.png",
                               model_scores=pipeline.oos_scores(src))
            recent = hists["production"]["strength"].iloc[-8:]
            alerts = build_alerts(date, price, out["production"], out.get("reference_nn"), recent, problems, chart)
            sent = dispatch(date, alerts, force=force_notify)
        elif notify and problems:
            sent = dispatch(date, [Alert("health", "BTC_Neural data problem", "\n".join(problems), 4, "x")], force=force_notify)
        return {"date": date, "price": price, "signals": out, "problems": problems, "sent": sent}
    except Exception as e:  # noqa: BLE001 - a crash must still produce an alert
        msg = f"Daily job failed: {e}\n{traceback.format_exc()[-800:]}"
        if notify:
            dispatch(pd.Timestamp.now().strftime("%Y-%m-%d"), [Alert("health", "BTC_Neural daily job FAILED", msg, 5, "x")], force=True)
        raise


def recent_live(limit: int = 60) -> pd.DataFrame:
    return db.query("SELECT * FROM live_signals ORDER BY date DESC, role LIMIT ?", (limit,))


__all__ = ["run_daily", "build_alerts", "health", "make_chart", "send_chart", "fmt_strength", "recent_live", "np"]
