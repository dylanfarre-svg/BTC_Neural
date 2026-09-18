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


def make_chart(daily: pd.DataFrame, hist: pd.DataFrame, out: Path, years: int = 4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = daily.index[-1] - pd.Timedelta(days=365 * years)
    c = daily["close"].loc[start:]
    s = hist["strength"].reindex(c.index).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    ax.plot(c.index, c.values, color="#444", lw=1.1)
    ax.set_yscale("log")
    for lvl_min, color, lab in ((10, "#104281", "Buy 10"), (7, "#6da7ec", "Buy 7-9")):
        m = (s >= lvl_min) & (s < (11 if lvl_min == 10 else 10))
        ax.scatter(c.index[m], c[m], color=color, s=14, label=lab, zorder=3)
    for lvl_min, color, lab in ((10, "#b3261e", "Sell 10"), (7, "#ec8a89", "Sell 7-9")):
        m = (-s >= lvl_min) & (-s < (11 if lvl_min == 10 else 10))
        ax.scatter(c.index[m], c[m], color=color, s=14, label=lab, zorder=3)
    ax.set_title(f"BTC/USD with model signals — today {fmt_strength(int(s.iloc[-1]))}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


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
            chart = make_chart(daily, hists["production"], ROOT / "data" / "charts" / f"signal_{date}.png")
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


__all__ = ["run_daily", "build_alerts", "health", "make_chart", "fmt_strength", "recent_live", "np"]
