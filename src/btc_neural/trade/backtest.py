"""Walk-forward research for the trading signals.

1. `run_walkforward` fits one model per side per purged block and stores the out-of-sample
   scores (`trade_predictions`). Thresholds play no part, so one run serves every threshold.
2. `evaluate` turns stored scores into trades: trade when today's score beats a trailing quantile
   of the model's own *earlier out-of-sample* scores (causal, no fitted threshold), and - for
   learned models, whose score estimates expected R - also clears its round-trip cost.
3. Windows are evaluated separately: full (2015+), modern (2019+), holdout (pinned final year).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import db
from ..data import daily_prices
from .features import build_trade_features, cycle_oos_predictions, trade_columns, valid_from
from .folds import blocks, train_mask
from .horizons import Horizon, horizon
from .labels import barrier_outcomes, training_target, uniqueness_weights
from .metrics import window_metrics
from .models import make_trade_model
from .simulator import Market, build_market, simulate


@lru_cache(maxsize=4)
def research_data(name: str = "swing", stop_ratio: float | None = None):
    """(horizon, daily prices, trade features, barrier outcomes, market) - built once per process.
    `stop_ratio` overrides the configured one, for the Q7 ablation (it changes the labels)."""
    from dataclasses import replace

    from ..backtest.walkforward import dataset
    hz = horizon(name)
    if stop_ratio is not None:
        hz = replace(hz, stop_ratio=float(stop_ratio))
    daily = daily_prices()
    _, cycle_feats = dataset()
    feats = build_trade_features(daily, db.read_series(), db.series_lags(), db.read_candles("4h"),
                                 cycle_oos_predictions(cycle_feats))
    h4 = db.read_candles("4h")
    out = barrier_outcomes(daily, h4, hz)
    return hz, daily, feats, out, build_market(daily, h4, hz, out)


def score_from(feature_set: str, hz: Horizon) -> pd.Timestamp:
    """First scored date: the warmup start, or one year after a late feature group begins."""
    vf = valid_from(feature_set)
    return max(hz.warmup_start, vf + pd.Timedelta(days=365)) if vf > hz.train_start else hz.warmup_start


def eval_from(feature_set: str, hz: Horizon) -> pd.Timestamp:
    """First evaluated date: thresholds need >= 180 days of the model's own OOS scores first."""
    return max(hz.backtest_start, score_from(feature_set, hz) + pd.Timedelta(days=180))


def windows(feature_set: str, hz: Horizon, last: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    e0, ho = eval_from(feature_set, hz), hz.holdout_start
    pre = ho - pd.Timedelta(days=1)
    return {"full": (e0, pre), "modern": (max(e0, hz.modern_start), pre), "holdout": (ho, last)}


def run_walkforward(model_type: str, feature_set: str, cfg: dict | None = None, name: str = "swing",
                    store: bool = True, verbose: bool = True) -> str:
    cfg = dict(cfg or {})
    hz, daily, feats, out, _ = research_data(name, cfg.get("stop_ratio"))
    t0 = time.time()
    F = feats[trade_columns(feature_set, list(feats.columns))]
    ys = {s: training_target(out[s]) for s in hz.sides}
    ws = {s: uniqueness_weights(out[s]) for s in hz.sides}
    blks = blocks(hz, score_from(feature_set, hz), F.index[-1])
    rows = []
    for b in blks:
        mask = train_mask(F.index, b, hz)
        Fb = F.loc[:b.end]
        for side in hz.sides:
            m = make_trade_model(model_type, side, cfg).fit(Fb, ys[side].where(mask).loc[:b.end], ws[side].loc[:b.end])
            s = m.predict(Fb).loc[b.start:b.end]
            rows.append(pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "side": side, "block": b.name,
                                      "score": s.to_numpy(dtype=float)}))
    preds = pd.concat(rows, ignore_index=True)
    seed = cfg.get("seed", 7)
    tag = f"__sr{cfg['stop_ratio']}" if cfg.get("stop_ratio") is not None else ""
    run_id = f"{name}__{model_type}__{feature_set}__s{seed}{tag}__{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"
    if store:
        with db.connect() as con:
            db.upsert(con, "trade_runs", pd.DataFrame([{
                "run_id": run_id, "created_at": db.now_utc(), "horizon": name, "model_type": model_type,
                "feature_set": feature_set, "role": "experiment",
                "config_json": json.dumps({"cfg": cfg, "blocks": len(blks), "score_from": str(score_from(feature_set, hz).date())}),
                "weights_path": None, "notes": ""}]))
            db.upsert(con, "trade_predictions", preds.assign(run_id=run_id)[["run_id", "date", "side", "block", "score"]])
    if verbose:
        print(f"  [{model_type}/{feature_set}/s{seed}] {len(blks)} blocks in {time.time() - t0:.0f}s -> {run_id}", flush=True)
    return run_id


def load_predictions(run_id: str) -> dict[str, pd.Series]:
    df = db.query("SELECT date, side, score FROM trade_predictions WHERE run_id=?", (run_id,))
    df["date"] = pd.to_datetime(df["date"])
    return {s: g.set_index("date")["score"].sort_index() for s, g in df.groupby("side")}


def signals(m: Market, scores: dict[str, pd.Series], q: float, calibrated: bool
            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.Series]]:
    hz = m.hz
    sig, pref, thr = {}, {}, {}
    cost_R = hz.round_trip_cost / m.out["long"]["tw"]
    for side, s in scores.items():
        s = s.reindex(m.days)
        t = s.rolling(f"{hz.threshold_trailing_days}D", min_periods=180, closed="left").quantile(q)
        ok = (s > t)
        if calibrated:
            ok &= s > cost_R
        sig[side] = ok.fillna(False).to_numpy()
        pref[side] = (s - t).fillna(-np.inf).to_numpy()
        thr[side] = t
    return sig, pref, thr


def evaluate(run_id: str, q: float, sides: tuple[str, ...] = ("long", "short"), name: str = "swing",
             market: Market | None = None, store: bool = True, keep: bool = False,
             windows_of: str | None = None, scores: dict[str, pd.Series] | None = None) -> dict:
    """`windows_of` evaluates over another feature set's windows (same-window comparisons);
    `scores` substitutes externally combined scores (the seed ensemble) for the run's own."""
    run = db.query("SELECT model_type, feature_set, config_json FROM trade_runs WHERE run_id=?", (run_id,)).iloc[0]
    sr = json.loads(run.config_json).get("cfg", {}).get("stop_ratio")
    hz, daily, feats, out, m0 = research_data(name, sr)
    m = market or m0
    calibrated = not run.model_type.startswith("srules_")
    sig, pref, _ = signals(m, scores or load_predictions(run_id), q, calibrated)
    res_by, metrics = {}, {}
    for w, (a, b) in windows(windows_of or run.feature_set, m.hz, m.days[-1]).items():
        res = simulate(m, sig, pref, a, b, sides)
        res_by[w] = res
        metrics[w] = window_metrics(m, res, m.hz, feats["rv_30"])
    sel = min(metrics["full"]["sharpe"], metrics["modern"]["sharpe"])
    tpy = metrics["full"].get("trades_per_year", 0.0)
    eligible = m.hz.min_trades_per_year <= tpy <= m.hz.max_trades_per_year
    eval_id = (f"{run_id}__q{q}__{'+'.join(sides)}__f{m.hz.fee_rate}"
               + (f"__w{windows_of}" if windows_of else "") + ("__ens" if scores is not None else ""))
    result = {"eval_id": eval_id, "run_id": run_id, "model_type": run.model_type, "feature_set": run.feature_set,
              "quantile": q, "sides": "+".join(sides), "fee_rate": m.hz.fee_rate, "selection": sel,
              "windows_of": windows_of or run.feature_set, "stop_ratio": m.hz.stop_ratio,
              "funding": m.hz.funding_enabled, "ensemble": scores is not None,
              "eligible": bool(eligible), "windows": metrics}
    if store:
        trades = res_by["full"].trades.copy()
        with db.connect() as con:
            db.upsert(con, "trade_evals", pd.DataFrame([{"eval_id": eval_id, "run_id": run_id, "quantile": q,
                                                          "sides": "+".join(sides), "created_at": db.now_utc(),
                                                          "metrics_json": json.dumps(result, default=_jsonable)}]))
            if len(trades):
                t = trades.drop(columns=["exit_day"]).assign(eval_id=eval_id)
                for c in ("entry_date", "exit_time"):
                    t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d %H:%M")
                db.upsert(con, "trade_backtest", t)
    if keep:
        result["_results"] = res_by
    return result


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def leaderboard(name: str = "swing") -> pd.DataFrame:
    df = db.query("SELECT e.eval_id, e.metrics_json, r.created_at FROM trade_evals e JOIN trade_runs r "
                  "ON e.run_id = r.run_id WHERE r.horizon = ? AND r.role = 'experiment'", (name,))
    rows = []
    for r in df.itertuples():
        j = json.loads(r.metrics_json)
        f, mo, ho = j["windows"]["full"], j["windows"]["modern"], j["windows"]["holdout"]
        seed = j["run_id"].split("__")[3]
        rows.append({"model_type": j["model_type"], "feature_set": j["feature_set"], "seed": seed,
                     "quantile": j["quantile"], "sides": j["sides"], "fee_rate": j["fee_rate"],
                     "selection": j["selection"], "eligible": j["eligible"],
                     "full_sharpe": f["sharpe"], "modern_sharpe": mo["sharpe"], "bh_full_sharpe": f["bh_sharpe"],
                     "full_tpy": f.get("trades_per_year", 0), "full_mean_net": f.get("mean_net", np.nan),
                     "full_total": f["total_pre_tax"], "bh_full_total": f["bh_total"],
                     "holdout_sharpe": ho["sharpe"], "eval_from": f["start"], "run_id": j["run_id"],
                     "eval_id": j["eval_id"]})
    return pd.DataFrame(rows)
