"""The swing research programme: grid -> pre-registered selection -> section 9 verdict.

The selection rule is frozen in PLAN_SWING.md section 15 *before* the grid runs; this module
implements it and nothing else decides promotion.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import db
from ..config import ROOT
from .backtest import evaluate, load_predictions, research_data, run_walkforward, windows
from .features import TRADE_FEATURE_SETS
from .metrics import deflated_sharpe, random_entry_bootstrap
from .simulator import reprice

POP_A = ["fast_core", "fast_flow", "fast_macro", "fast_cycle", "fast_all"]   # valid from 2015
POP_B = ["fast_sent", "fast_funding", "fast_deriv"]                         # valid from 2019+
SEEDS = list(range(1, 11))
REPORT = ROOT / "runs" / "swing_report.json"


def grid_tasks() -> list[tuple[str, str, dict]]:
    tasks = [("srules_momentum", "fast_core", {"seed": 0}), ("srules_pullback", "fast_core", {"seed": 0})]
    for fs in POP_A + POP_B:
        tasks.append(("linear", fs, {"seed": 0}))
        for s in SEEDS:
            tasks.append(("lgbm", fs, {"seed": s}))
            tasks.append(("ens", fs, {"seed": s}))
    return tasks


def _work(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mt, fs, cfg = task
    return run_walkforward(mt, fs, cfg, verbose=False)


def run_grid(workers: int = 6, nn: bool = True) -> list[str]:
    tasks = grid_tasks()
    print(f"grid: {len(tasks)} walk-forwards on {workers} workers", flush=True)
    t0, done = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, t): t for t in tasks}
        for f in as_completed(futs):
            done.append(f.result())
            if len(done) % 20 == 0:
                print(f"  {len(done)}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    if nn:  # reference track; MPS in the main process only
        for fs in ("fast_core", "fast_all"):
            for s in (1, 2, 3):
                done.append(run_walkforward("nn", fs, {"seed": s}))
    return done


def experiment_runs() -> pd.DataFrame:
    df = db.query("SELECT run_id, model_type, feature_set, config_json FROM trade_runs "
                  "WHERE horizon='swing' AND role='experiment'")
    cfg = df.config_json.map(lambda j: json.loads(j).get("cfg", {}))
    df["seed"] = cfg.map(lambda c: c.get("seed", 0))
    df["stop_ratio"] = cfg.map(lambda c: c.get("stop_ratio"))
    return df[df.stop_ratio.isna()].drop(columns=["config_json", "stop_ratio"])


def _eval_work(args):
    run_id, q = args
    r = evaluate(run_id, q)
    w = r["windows"]
    return {"run_id": run_id, "quantile": q, "selection": r["selection"], "eligible": r["eligible"],
            "full_sharpe": w["full"]["sharpe"], "modern_sharpe": w["modern"]["sharpe"],
            "holdout_sharpe": w["holdout"]["sharpe"], "full_tpy": w["full"].get("trades_per_year", 0.0),
            "full_mean_net": w["full"].get("mean_net", np.nan), "full_total": w["full"]["total_pre_tax"],
            "bh_full_sharpe": w["full"]["bh_sharpe"], "eval_from": w["full"]["start"]}


def evaluate_grid(workers: int = 6) -> pd.DataFrame:
    runs = experiment_runs()
    qs = research_data()[0].quantiles
    tasks = [(r, q) for r in runs.run_id for q in qs]
    print(f"evaluating {len(tasks)} (run, quantile) variants", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_eval_work, t) for t in tasks]):
            rows.append(f.result())
    ev = pd.DataFrame(rows).merge(runs, on="run_id")
    ev.to_csv(ROOT / "runs" / "swing_grid.csv", index=False)
    return ev


def config_table(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, feature set, quantile): seed means, seed spread and seed count."""
    g = ev.groupby(["model_type", "feature_set", "quantile"])
    t = g.agg(selection=("selection", "mean"), selection_sd=("selection", "std"), seeds=("selection", "size"),
              full_sharpe=("full_sharpe", "mean"), modern_sharpe=("modern_sharpe", "mean"),
              holdout_sharpe=("holdout_sharpe", "mean"), tpy=("full_tpy", "mean"),
              mean_net=("full_mean_net", "mean"), bh_sharpe=("bh_full_sharpe", "first"),
              eval_from=("eval_from", "first")).reset_index()
    lo, hi = research_data()[0].min_trades_per_year, research_data()[0].max_trades_per_year
    t["eligible"] = (t.tpy >= lo) & (t.tpy <= hi)
    return t.sort_values("selection", ascending=False)


def select(ev: pd.DataFrame) -> pd.Series:
    """Pre-registered: population A, eligible on mean trades/yr, max seed-mean selection score."""
    t = config_table(ev)
    a = t[t.feature_set.isin(POP_A) & t.eligible]
    if a.empty:
        raise RuntimeError("no eligible configuration in population A")
    return a.iloc[0]


def ensemble_scores(run_ids: list[str]) -> dict[str, pd.Series]:
    """Seed ensemble: average the out-of-sample scores of every seed of a configuration."""
    preds = [load_predictions(r) for r in run_ids]
    return {s: pd.concat([p[s] for p in preds], axis=1).mean(axis=1) for s in preds[0]}


def _copy_market(m):
    """A market that can be repriced without touching the cached one (its dicts are per-instance)."""
    return replace(m, exit_day={}, net={}, funding_paid={})


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def verdict(ev: pd.DataFrame, chosen: pd.Series) -> dict:
    hz, daily, feats, out, m = research_data()
    runs = ev[(ev.model_type == chosen.model_type) & (ev.feature_set == chosen.feature_set)].run_id.unique().tolist()
    scores = ensemble_scores(runs)
    q = float(chosen.quantile)
    base = runs[0]
    rep: dict = {"chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in chosen.items()},
                 "seeds_in_ensemble": len(runs)}
    main = evaluate(base, q, scores=scores, keep=True)
    rep["main"] = _strip(main)
    res = main["_results"]

    # significance, per window
    sig = {}
    for w in ("full", "modern", "holdout"):
        sig[w] = random_entry_bootstrap(m, res[w], ("long", "short"), n_draws=1000, seed=11)
    rep["random_entry"] = sig

    # deflated Sharpe over every configuration tried in population A
    t = config_table(ev)
    a = t[t.feature_set.isin(POP_A)]
    sr_daily = a.full_sharpe / np.sqrt(365)
    rep["deflated_sharpe"] = {"n_trials": int(len(a)), "dsr_full": deflated_sharpe(res["full"].daily, len(a), float(sr_daily.var()))}

    # sides separately (the long side may work when the short side doesn't)
    rep["sides"] = {s: _strip(evaluate(base, q, sides=(s,), scores=scores)) for s in ("long", "short")}

    # fee sweep and funding off (repricing only - no refits)
    sweep = {}
    for fee in (0.0, 0.0005, 0.001, 0.0025, 0.005):
        mm = _copy_market(m)
        reprice(mm, hz.with_costs(fee_rate=fee))
        r = evaluate(base, q, scores=scores, market=mm, store=False)
        sweep[str(fee)] = {w: {k: r["windows"][w].get(k) for k in ("sharpe", "total_pre_tax", "total_revenue", "mean_net")}
                           for w in ("full", "modern", "holdout")}
    rep["fee_sweep"] = sweep
    mm = _copy_market(m)
    reprice(mm, hz.with_costs(funding_enabled=False))
    rep["funding_off"] = _strip(evaluate(base, q, scores=scores, market=mm, store=False))["windows"]

    rep["checks"] = section9(rep)
    return rep


def section9(rep: dict) -> dict:
    """Every ship criterion, per window and per tax treatment (PLAN_SWING.md section 9a)."""
    out = {}
    for w in ("full", "modern"):
        mw, rs = rep["main"]["windows"][w], rep["random_entry"][w]
        c = {
            "expectancy_lb95_positive": bool(mw.get("mean_net_lb95", np.nan) > 0),
            "beats_random_entry_p95": bool(rs.get("p_total", 1.0) < 0.05),
            "regimes_3_of_4": bool(mw.get("regimes_positive", 0) >= 3),
            "median_hold_7_28": bool(7 <= mw.get("median_hold", 0) <= 28),
            "beats_bh_sharpe": bool(mw["beats_bh_sharpe"]),
        }
        for tr in ("capital", "revenue"):
            c[f"beats_bh_total_{tr}"] = bool(mw[f"beats_bh_total_{tr}"])
            c[f"beats_bh_rptim_{tr}"] = bool(mw[f"beats_bh_rptim_{tr}"])
        out[w] = c
    for tr in ("capital", "revenue"):
        keys = ["expectancy_lb95_positive", "beats_random_entry_p95", "regimes_3_of_4", "median_hold_7_28",
                "beats_bh_sharpe", f"beats_bh_total_{tr}", f"beats_bh_rptim_{tr}"]
        out[f"ship_{tr}"] = all(out[w][k] for w in ("full", "modern") for k in keys)
    return out


def population_b(ev: pd.DataFrame, chosen: pd.Series) -> dict:
    """Late feature sets vs fast_core over the SAME window, same model and quantile, seed means."""
    out = {}
    for fs in POP_B:
        for mt in ("lgbm", "ens", "linear"):
            late = ev[(ev.model_type == mt) & (ev.feature_set == fs)].run_id.unique()
            core = ev[(ev.model_type == mt) & (ev.feature_set == "fast_core")].run_id.unique()
            if not len(late) or not len(core):
                continue
            q = float(chosen.quantile)
            sl = [evaluate(r, q, store=False)["windows"]["full"]["sharpe"] for r in late]
            sc = [evaluate(r, q, store=False, windows_of=fs)["windows"]["full"]["sharpe"] for r in core]
            d = np.mean(sl) - np.mean(sc)
            se = np.sqrt(np.var(sl, ddof=1) / len(sl) + np.var(sc, ddof=1) / len(sc)) if min(len(sl), len(sc)) > 1 else np.nan
            out[f"{mt}/{fs}"] = {"window_from": windows(fs, research_data()[0], research_data()[4].days[-1])["full"][0].strftime("%Y-%m-%d"),
                                 "late_sharpe": float(np.mean(sl)), "core_sharpe_same_window": float(np.mean(sc)),
                                 "diff": float(d), "se": float(se) if np.isfinite(se) else None,
                                 "adopt": bool(np.isfinite(se) and d > 2 * se)}
    return out


def stop_ratio_ablation(chosen: pd.Series, ratios=(1.0,)) -> dict:
    """Q7: refit the chosen configuration with a different stop ratio (new labels), seed means."""
    out = {}
    seeds = SEEDS if chosen.model_type in ("lgbm", "ens") else [0]
    for sr in ratios:
        runs = [run_walkforward(chosen.model_type, chosen.feature_set, {"seed": s, "stop_ratio": sr}, verbose=False)
                for s in seeds]
        scores = ensemble_scores(runs)
        r = evaluate(runs[0], float(chosen.quantile), scores=scores)
        out[str(sr)] = {w: {k: r["windows"][w].get(k) for k in ("sharpe", "total_pre_tax", "mean_net", "median_hold",
                                                                  "trades_per_year", "hit_rate")}
                        for w in ("full", "modern", "holdout")}
    return out


def run_all(workers: int = 6) -> dict:
    run_grid(workers)
    ev = evaluate_grid(workers)
    chosen = select(ev)
    print("selected:", dict(chosen), flush=True)
    rep = verdict(ev, chosen)
    rep["population_b"] = population_b(ev, chosen)
    rep["stop_ratio"] = stop_ratio_ablation(chosen)
    rep["top_configs"] = json.loads(config_table(ev).head(25).to_json(orient="records"))
    REPORT.write_text(json.dumps(rep, indent=2, default=str))
    print(f"report -> {REPORT}", flush=True)
    return rep


if __name__ == "__main__":
    run_all()
