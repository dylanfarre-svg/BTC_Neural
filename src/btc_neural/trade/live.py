"""Live paper trading for the swing signal: the walk-forward, continued one day at a time.

The live signal is deliberately the *same construction* that was backtested, not a differently
fitted "production" model: the current 90-day block's models (trained through block start minus
the purge, one per seed and side, exactly as in the walk-forward), and a trade threshold equal to
the trailing quantile of the configuration's own out-of-sample scores - the walk-forward history
first, then the live scores this job has logged. Anything else would make the paper log a test of
a different strategy from the one the backtest measured.

Research only: nothing here notifies. Every day appends one row to `trade_signals` (append-only);
positions live in `trade_positions`. `notify.dispatch` also drops any `trade*` alert outright.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from .. import db
from ..config import ROOT, path
from .backtest import load_predictions, research_data, score_from
from .features import trade_columns
from .folds import blocks, train_mask
from .labels import training_target, uniqueness_weights
from .models import make_trade_model
from .research import REPORT, SEEDS

NAME = "swing"


def chosen_config() -> dict | None:
    """The configuration promoted by the pre-registered selection (runs/swing_report.json)."""
    if not REPORT.exists():
        return None
    c = json.loads(REPORT.read_text())["chosen"]
    return {"model_type": c["model_type"], "feature_set": c["feature_set"], "quantile": float(c["quantile"])}


def _config_runs(cfg: dict) -> list[str]:
    df = db.query("SELECT run_id, config_json FROM trade_runs WHERE horizon=? AND role='experiment' "
                  "AND model_type=? AND feature_set=?", (NAME, cfg["model_type"], cfg["feature_set"]))
    keep = [r.run_id for r in df.itertuples() if json.loads(r.config_json).get("cfg", {}).get("stop_ratio") is None]
    return keep


def history_scores(cfg: dict) -> dict[str, pd.Series]:
    """Seed-ensemble walk-forward OOS scores, then the live scores logged since."""
    preds = [load_predictions(r) for r in _config_runs(cfg)]
    hist = {s: pd.concat([p[s] for p in preds], axis=1).mean(axis=1) for s in ("long", "short")} if preds else \
        {"long": pd.Series(dtype=float), "short": pd.Series(dtype=float)}
    live = db.query("SELECT ts, score_long, score_short FROM trade_signals WHERE horizon=? ORDER BY ts", (NAME,))
    if len(live):
        live["ts"] = pd.to_datetime(live["ts"])
        live = live.set_index("ts")
        for s in ("long", "short"):
            add = live[f"score_{s}"].dropna()
            hist[s] = pd.concat([hist[s][hist[s].index < add.index.min()] if len(add) else hist[s], add])
    return hist


def block_models(cfg: dict, today: pd.Timestamp) -> tuple[dict[str, list], str]:
    """The models of the block containing `today`, fitted once and kept under runs/."""
    hz, daily, feats, out, _ = research_data(NAME)
    bl = [b for b in blocks(hz, score_from(cfg["feature_set"], hz), today) if b.start <= today <= b.end]
    if not bl:
        raise RuntimeError("no walk-forward block covers today")
    b = bl[-1]
    d = path("runs_dir") / f"swing_live__{cfg['model_type']}__{cfg['feature_set']}__{b.name}"
    f = d / "models.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes()), b.name
    F = feats[trade_columns(cfg["feature_set"], list(feats.columns))]
    mask = train_mask(F.index, b, hz)
    seeds = SEEDS if cfg["model_type"] in ("lgbm", "ens", "nn") else [0]
    models = {}
    for side in hz.sides:
        y, w = training_target(out[side]).where(mask), uniqueness_weights(out[side])
        models[side] = [make_trade_model(cfg["model_type"], side, {"seed": s}).fit(F.loc[:b.train_end], y.loc[:b.train_end], w.loc[:b.train_end])
                        for s in seeds]
    d.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(models))
    run_id = d.name
    with db.connect() as con:
        db.upsert(con, "trade_runs", pd.DataFrame([{
            "run_id": run_id, "created_at": db.now_utc(), "horizon": NAME, "model_type": cfg["model_type"],
            "feature_set": cfg["feature_set"], "role": "production",
            "config_json": json.dumps({**cfg, "block": b.name, "train_end": str(b.train_end.date()), "seeds": seeds}),
            "weights_path": str(f), "notes": "live block models"}]))
    return models, b.name


def _open_position() -> pd.Series | None:
    df = db.query("SELECT * FROM trade_positions WHERE horizon=? ORDER BY opened_at DESC LIMIT 1", (NAME,))
    return df.iloc[0] if len(df) else None


def run_swing_daily(verbose: bool = True) -> dict:
    """Score today, advance the position state machine, append to the paper log. Never notifies."""
    from .ingest import ingest_derivatives

    cfg = chosen_config()
    if cfg is None:
        return {"status": "no selected configuration (run `btcn trade research`)"}
    ingest_derivatives(verbose=False)
    research_data.cache_clear()
    from ..backtest.walkforward import dataset
    dataset.cache_clear()
    hz, daily, feats, out, m = research_data(NAME)
    today = daily.index[-1]
    i = m.pos(today)
    if not np.isfinite(m.px[i]):
        return {"status": f"no 4h close for {today.date()} yet"}

    freeze_gate()
    models, block = block_models(cfg, today)
    F = feats[trade_columns(cfg["feature_set"], list(feats.columns))]
    score = {s: float(np.mean([mdl.predict(F.loc[:today]).iloc[-1] for mdl in models[s]])) for s in hz.sides}
    hist = history_scores(cfg)
    thr = {}
    for s in hz.sides:
        h = hist[s][(hist[s].index < today) & (hist[s].index >= today - pd.Timedelta(days=hz.threshold_trailing_days))]
        thr[s] = float(h.quantile(cfg["quantile"])) if len(h) >= 180 else np.nan
    cost_R = hz.round_trip_cost / float(out["long"]["tw"].iloc[i])
    calibrated = not cfg["model_type"].startswith("srules_")
    ok = {s: bool(np.isfinite(thr[s]) and score[s] > thr[s] and (not calibrated or score[s] > cost_R)
                  and (s != "short" or today >= hz.short_start)) for s in hz.sides}

    # --- position state machine: flat -> long|short -> closed (target|stop|deadline) -> cooling_off -> flat
    pos, action, state = _open_position(), "none", "flat"
    if pos is not None and pos.state in ("long", "short"):
        o = out[pos.side].loc[pd.Timestamp(pos.opened_at)]
        if pd.notna(o["reason"]):
            k = m.exit_day[pos.side][m.pos(pd.Timestamp(pos.opened_at))]
            net = float(m.net[pos.side][m.pos(pd.Timestamp(pos.opened_at))])
            db.execute("UPDATE trade_positions SET state='closed', closed_at=?, exit_price=?, exit_reason=?, net_ret=? "
                       "WHERE trade_id=?", (str(o["exit_time"]), float(o["exit_price"]), o["reason"], net, pos.trade_id))
            action = f"exit_{o['reason']}"
            state = "cooling_off" if (i - k) < hz.cooldown_days else "flat"
        else:
            state = pos.state
    elif pos is not None and pos.state == "closed":
        k = m.pos((pd.Timestamp(pos.closed_at) - pd.Timedelta(seconds=1)).normalize())
        state = "cooling_off" if (i - k) < hz.cooldown_days else "flat"

    if state == "flat" and any(ok.values()):
        side = max((s for s in ok if ok[s]), key=lambda s: score[s] - thr[s])
        e, tw = float(m.px[i]), float(out[side]["tw"].iloc[i])
        tgt = e * (1 + tw) if side == "long" else e * (1 - tw)
        stp = e * (1 - hz.stop_ratio * tw) if side == "long" else e * (1 + hz.stop_ratio * tw)
        tid = f"{NAME}-{today:%Y%m%d}-{side}"
        with db.connect() as con:
            db.upsert(con, "trade_positions", pd.DataFrame([{
                "trade_id": tid, "horizon": NAME, "run_id": f"{cfg['model_type']}/{cfg['feature_set']}/{block}",
                "side": side, "state": side, "opened_at": today.strftime("%Y-%m-%d"), "entry_price": e,
                "target_price": tgt, "stop_price": stp,
                "deadline": (today + pd.Timedelta(days=1 + hz.hold_max_days)).strftime("%Y-%m-%d %H:%M"),
                "closed_at": None, "exit_price": None, "exit_reason": None, "net_ret": None}]))
        action, state = f"enter_{side}", side

    drivers = {}
    try:
        mdl = models["long" if state != "short" else "short"][0]
        drivers = {k: float(v) for k, v in mdl.explain(F.loc[:today]).head(6).items()}
    except Exception:  # noqa: BLE001 - drivers are cosmetic
        pass
    with db.connect() as con:
        con.execute("INSERT OR IGNORE INTO trade_signals (ts, horizon, run_id, score_long, score_short, thr_long, thr_short, "
                    "action, state, price, drivers_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (today.strftime("%Y-%m-%d"), NAME, f"{cfg['model_type']}/{cfg['feature_set']}/{block}",
                     score["long"], score["short"], thr["long"], thr["short"], action, state, float(m.px[i]),
                     json.dumps(drivers), db.now_utc()))
    res = {"date": str(today.date()), "action": action, "state": state, "score": score, "threshold": thr,
           "config": cfg, "block": block}
    if verbose:
        print(f"  swing {today.date()}: {action} (state {state}) long {score['long']:+.3f}/{thr['long']:+.3f} "
              f"short {score['short']:+.3f}/{thr['short']:+.3f} [{cfg['model_type']}/{cfg['feature_set']} q{cfg['quantile']}]")
    return res


GATE = ROOT / "runs" / "swing_paper_gate.json"


def freeze_gate() -> dict:
    """Record the backtest's lower bound the first time paper trading runs, and never again
    (PLAN_SWING.md section 9c: the goalposts are fixed before the live evidence arrives)."""
    if GATE.exists():
        return json.loads(GATE.read_text())
    rep = json.loads(REPORT.read_text())
    g = {"frozen_on": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), "config": rep["chosen"],
         "backtest_mean_net_lb95": rep["main"]["windows"]["full"].get("mean_net_lb95"),
         "backtest_mean_net": rep["main"]["windows"]["full"].get("mean_net"),
         "min_trades": 10, "evaluate": "quarterly", "stop_after_trades": 25}
    GATE.write_text(json.dumps(g, indent=2, default=str))
    return g


def paper_gate() -> dict:
    """Live mean net return per closed trade against the frozen backtest lower bound."""
    if not GATE.exists():
        return {}
    g = json.loads(GATE.read_text())
    closed = db.query("SELECT net_ret FROM trade_positions WHERE horizon=? AND state='closed'", (NAME,))
    n = len(closed)
    mean = float(closed.net_ret.mean()) if n else float("nan")
    lb = g.get("backtest_mean_net_lb95")
    return {**g, "closed_trades": n, "live_mean_net": mean, "evaluable": n >= g["min_trades"],
            "passes": bool(n >= g["min_trades"] and lb is not None and mean >= lb),
            "stop": bool(n >= g["stop_after_trades"] and lb is not None and mean < lb)}


__all__ = ["run_swing_daily", "paper_gate", "chosen_config", "ROOT"]
