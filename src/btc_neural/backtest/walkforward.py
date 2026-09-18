"""Walk-forward backtesting over halving epochs.

Fold Ek: train on everything before epoch k (labels computed as of that cutoff), calibrate
strength thresholds on the training period only, then predict epoch k unseen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import db
from ..calibrate import Calibration, fit_calibration, to_strength
from ..config import load_config
from ..data import cycle_extremes, epoch_of, epochs
from ..features import columns_for, load_all
from ..labels import make_label
from ..models import Model, make_model
from .metrics import evaluate_period
from .simulator import simulate


@lru_cache(maxsize=1)
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_all()


def usdzar() -> pd.Series | None:
    s = db.read_series(["DEXSFUS"])
    return s["DEXSFUS"].dropna() if not s.empty else None


def sample_weights(y: pd.Series) -> pd.Series:
    """Balance epochs (each cycle counts equally) and emphasise the extremes."""
    ep = pd.Series(epoch_of(y.index), index=y.index)
    lab = y.notna()
    counts = ep[lab].value_counts()
    w = 1.0 / ep.map(counts).astype(float)
    return (w * (1 + 2 * y.fillna(0) ** 2)).where(lab)


def calibration_rows(F: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Index:
    first = pd.Timestamp(load_config()["extremes"][0]["date"])
    return F.loc[first:cutoff].index


def fit_with_calibration(model_type: str, cfg: dict, F: pd.DataFrame, y: pd.Series,
                         cutoff: pd.Timestamp) -> tuple[Model, Calibration]:
    w = sample_weights(y)
    model = make_model(model_type, cfg).fit(F.loc[:cutoff], y, w)
    rows = calibration_rows(F, cutoff)
    method = load_config()["calibration"]
    ep = pd.Series(epoch_of(rows), index=rows)
    labeled_eps = sorted(set(epoch_of(y.dropna().index)))
    if method == "oof" and len(labeled_eps) >= 2 and model_type != "rules":
        inner_cfg = dict(cfg)
        if "nn" in model_type:
            inner_cfg["seeds"] = max(2, load_config()["models"]["nn"]["seeds"] // 2)
        parts = []
        for e in sorted(ep.unique()):
            y_in = y.where(pd.Series(epoch_of(y.index), index=y.index) != e)
            if y_in.notna().sum() < 100:
                continue
            m = make_model(model_type, inner_cfg).fit(F.loc[:cutoff], y_in, sample_weights(y_in))
            parts.append(m.predict(F.loc[:cutoff]).loc[ep.index[ep == e]])
        scores = pd.concat(parts) if parts else model.predict(F.loc[:cutoff]).loc[rows]
    else:
        scores = model.predict(F.loc[:cutoff]).loc[rows]
    return model, fit_calibration(scores)


@dataclass
class RunResult:
    run_id: str
    selection_score: float
    folds: pd.DataFrame        # metrics per fold (rows) x metric (cols)
    predictions: pd.DataFrame  # date, fold, score, strength


def run_walkforward(model_type: str, feature_set: str, cfg: dict | None = None, modes: list[str] | None = None,
                    store: bool = True, verbose: bool = True) -> dict[str, RunResult]:
    """Fit once per fold, then score every strength mode in `modes` (one stored run per mode)."""
    cfg = dict(cfg or {})
    modes = modes or [cfg.pop("strength_mode", load_config()["strength_mode"])]
    cfg.pop("strength_mode", None)
    t0 = time.time()
    daily, feats = dataset()
    close = daily["close"]
    ex = cycle_extremes(close)
    cols = columns_for(feature_set, list(feats.columns))
    F = feats[cols]
    eps = epochs(end=close.index[-1], start=close.index[0])
    fx = usdzar()
    rows: dict[str, list] = {m: [] for m in modes}
    preds: dict[str, list] = {m: [] for m in modes}
    for name, start, end_ex in eps[1:]:
        cutoff = start - pd.Timedelta(days=1)
        end = min(end_ex - pd.Timedelta(days=1), close.index[-1])
        y = make_label(close, ex, cutoff)
        if y.notna().sum() < load_config()["folds"]["min_train_days"] // 3:
            continue
        model, cal = fit_with_calibration(model_type, cfg, F, y, cutoff)
        full = model.predict(F.loc[:end])
        score = full.loc[start:end]
        dca = simulate(close, score * 0, fx, start, end, dca_only=True)
        for mode in modes:
            strength = to_strength(full, cal, mode).loc[start:end]
            m = evaluate_period(close, score, strength, ex, start, end)
            sim = simulate(close, strength, fx, start, end)
            m["sim_btc_vs_dca"] = sim.summary["final_btc"] / dca.summary["final_btc"]
            m["sim_value_vs_dca"] = sim.summary["final_value_usd"] / dca.summary["final_value_usd"]
            m["sim_tax_usd"] = sim.summary["tax_paid_usd"] + sim.summary["tax_accrued_usd"]
            m["sim_avg_sell_price"] = sim.summary["avg_sell_price"]
            m["sim_avg_buy_price"] = sim.summary["avg_buy_price"]
            m["sim_dca_avg_buy_price"] = dca.summary["avg_buy_price"]
            m["train_labeled_days"] = float(y.notna().sum())
            rows[mode].append({"fold": name, **m})
            preds[mode].append(pd.DataFrame({"date": score.index, "fold": name, "score": score.values,
                                             "strength": strength.values}))
            if verbose:
                print(f"  [{model_type}/{feature_set}/{mode}] {name}: fold_score={m['fold_score']:.3f} "
                      f"sell10={m.get('sell_n10', 0):.0f} buy10={m.get('buy_n10', 0):.0f} "
                      f"sell_err={m.get('sell10_log_err', np.nan):.2f} buy_err={m.get('buy10_log_err', np.nan):.2f} "
                      f"rho={m.get('spearman_vs_position', np.nan):.2f} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for mode in modes:
        folds = pd.DataFrame(rows[mode]).set_index("fold")
        predictions = pd.concat(preds[mode], ignore_index=True)
        sel = [f for f in load_config()["folds"]["selection_folds"] if f in folds.index]
        selection_score = float(folds.loc[sel, "fold_score"].max())
        run_id = f"{model_type}__{feature_set}__{mode}__{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}"
        if store:
            _store(run_id, model_type, feature_set, {**cfg, "strength_mode": mode}, selection_score, folds, predictions)
        if verbose:
            print(f"  => {run_id} selection_score={selection_score:.3f}", flush=True)
        out[mode] = RunResult(run_id, selection_score, folds, predictions)
    return out


def _store(run_id, model_type, feature_set, cfg, selection_score, folds, predictions):
    long = folds.reset_index().melt(id_vars="fold", var_name="metric", value_name="value").dropna()
    long.insert(0, "run_id", run_id)
    p = predictions.copy()
    p["date"] = p["date"].dt.strftime("%Y-%m-%d")
    p.insert(0, "run_id", run_id)
    db.save_run(run_id, model_type, feature_set, "experiment",
                {"cfg": cfg, "label": load_config()["label"], "calibration": load_config()["calibration"],
                 "strength_mode": cfg["strength_mode"]},
                selection_score, None)
    with db.connect() as con:
        db.upsert(con, "fold_results", long)
        db.upsert(con, "predictions", p)


def leaderboard() -> pd.DataFrame:
    runs = db.query("SELECT run_id, created_at, model_type, feature_set, json_extract(config_json, '$.strength_mode') AS mode, "
                    "selection_score FROM runs WHERE role='experiment'")
    fr = db.query("SELECT run_id, fold, metric, value FROM fold_results WHERE metric='fold_score'")
    if runs.empty:
        return runs
    wide = fr.pivot_table(index="run_id", columns="fold", values="value")
    out = runs.merge(wide, left_on="run_id", right_index=True, how="left")
    return out.sort_values("selection_score")
