"""Production model management: pick the walk-forward winner, retrain on all data, predict."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import db
from .backtest.walkforward import dataset, fit_with_calibration, leaderboard
from .calibrate import Calibration, to_strength
from .config import load_config, path
from .data import cycle_extremes
from .features import columns_for
from .labels import make_label
from .models import Model

ROLES = ("production", "reference_nn")


def candidate_role(model_type: str) -> str:
    return f"candidate_{model_type}"


def best_run(nn_only: bool = False, exclude: tuple[str, str] | None = None) -> pd.Series | None:
    """Best walk-forward run; `exclude` = (model_type, feature_set) to skip."""
    lb = leaderboard()
    if lb.empty:
        return None
    if nn_only:
        lb = lb[lb.model_type == "nn"]
    if exclude:
        lb = lb[~((lb.model_type == exclude[0]) & (lb.feature_set == exclude[1]))]
    lb = lb.dropna(subset=["selection_score"])
    return lb.iloc[0] if len(lb) else None


def _production_config() -> tuple[str, str] | None:
    df = db.query("SELECT model_type, feature_set FROM runs WHERE role='production' ORDER BY created_at DESC LIMIT 1")
    return (df.model_type.iloc[0], df.feature_set.iloc[0]) if len(df) else None


def train_production(source_run_id: str | None = None, role: str = "production", verbose: bool = True) -> str:
    """Retrain the chosen configuration on all data available today and save it."""
    if source_run_id is None:
        # If the production model is itself a neural net, the reference track uses the next-best
        # NN configuration so that it still adds a second opinion.
        prod = _production_config() if role == "reference_nn" else None
        src = best_run(nn_only=(role == "reference_nn"), exclude=prod if prod and prod[0] == "nn" else None)
        if src is None:
            raise RuntimeError("No walk-forward runs found. Run `btcn backtest` first.")
        source_run_id = src.run_id
    run = db.query("SELECT * FROM runs WHERE run_id=?", (source_run_id,)).iloc[0]
    run_cfg = json.loads(run.config_json)
    cfg = run_cfg.get("cfg", {})
    daily, feats = dataset()
    close = daily["close"]
    cutoff = close.index[-1]
    # Optional in-sample cutoff: nothing after this date is used for fitting or calibration, so the
    # period since then is a genuine holdout that the live signal log records day by day.
    tc = load_config().get("training_cutoff")
    if tc:
        cutoff = min(cutoff, pd.Timestamp(tc))
    F = feats[columns_for(run.feature_set, list(feats.columns))]
    y = make_label(close, cycle_extremes(close), cutoff)
    model, cal = fit_with_calibration(run.model_type, cfg, F, y, cutoff)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"{role}__{run.model_type}__{run.feature_set}__{stamp}"
    out_dir = path("runs_dir") / run_id
    model.save(out_dir / "model.pkl")
    meta = {
        "source_run_id": source_run_id, "model_type": run.model_type, "feature_set": run.feature_set,
        "cfg": cfg, "strength_mode": cfg.get("strength_mode", run_cfg.get("strength_mode", load_config()["strength_mode"])),
        "declustering": cfg.get("declustering", load_config().get("declustering", {"mode": "none"})),
        "calibration": cal.to_dict(), "trained_through": cutoff.strftime("%Y-%m-%d"),
        "labeled_days": int(y.notna().sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    db.save_run(run_id, run.model_type, run.feature_set, role, meta, float(run.selection_score),
                str(out_dir / "model.pkl"), notes=f"retrained from {source_run_id}")
    if verbose:
        print(f"  {role}: {run_id} (from {source_run_id}, selection_score={run.selection_score:.3f})")
    return run_id


def train_all_candidates(verbose: bool = True) -> list[str]:
    """Keep one trained model per model type: the best walk-forward configuration of each,
    retrained on all data. Preserved so their signals and forecasts stay available."""
    from .models import MODEL_TYPES
    out = []
    lb = leaderboard()
    type_roles = set()
    for mt in MODEL_TYPES:
        sub = lb[lb.model_type == mt].dropna(subset=["selection_score"])
        if sub.empty:
            continue
        type_roles.add(candidate_role(mt))
        out.append(train_production(sub.iloc[0].run_id, role=candidate_role(mt), verbose=verbose))
    # refresh any extra saved models (e.g. candidate_nn_eth) from the run they were built from
    extra = all_roles()
    for r in extra[extra.role.str.startswith("candidate_")].itertuples():
        if r.role not in type_roles and r.source_run_id:
            out.append(train_production(r.source_run_id, role=r.role, verbose=verbose))
    return out


def all_roles() -> pd.DataFrame:
    """Every saved model, newest per role: production, NN reference and one per model type."""
    df = db.query("SELECT role, run_id, model_type, feature_set, selection_score, created_at, weights_path, "
                  "json_extract(config_json,'$.source_run_id') AS source_run_id FROM runs "
                  "WHERE role != 'experiment' ORDER BY created_at DESC")
    return df.drop_duplicates("role")


def oos_scores(source_run_id: str) -> pd.Series:
    """Walk-forward (out-of-sample) score series stored for a run."""
    df = db.query("SELECT date, score FROM predictions WHERE run_id=? ORDER BY date", (source_run_id,))
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["score"].to_numpy(), index=pd.to_datetime(df["date"]))


def current(role: str) -> tuple[Model, dict, str] | None:
    df = db.query("SELECT run_id, weights_path, config_json FROM runs WHERE role=? ORDER BY created_at DESC LIMIT 1", (role,))
    if df.empty:
        return None
    r = df.iloc[0]
    return Model.load(r.weights_path), json.loads(r.config_json), r.run_id


def age_days(role: str) -> float:
    df = db.query("SELECT created_at FROM runs WHERE role=? ORDER BY created_at DESC LIMIT 1", (role,))
    if df.empty:
        return np.inf
    return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(df.created_at.iloc[0])).total_seconds() / 86400


def signal_history(role: str, feats: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Score, strength and spread for every day, using the current model of `role`."""
    cur = current(role)
    if cur is None:
        return None
    model, meta, run_id = cur
    if feats is None:
        _, feats = dataset()
    F = feats[model.columns]
    score, spread = model.predict_with_spread(F)
    strength = to_strength(score, Calibration.from_dict(meta["calibration"]), meta.get("strength_mode"),
                           meta.get("declustering"), meta.get("cfg", {}).get("trailing_window"),
                           meta.get("cfg", {}).get("signal_budget"))
    out = pd.DataFrame({"score": score, "strength": strength, "spread": spread})
    out.attrs.update(run_id=run_id, meta=meta)
    return out


def explain_today(role: str, feats: pd.DataFrame | None = None, top: int = 6) -> pd.Series:
    cur = current(role)
    if cur is None:
        return pd.Series(dtype=float)
    model, _, _ = cur
    if feats is None:
        _, feats = dataset()
    return model.explain(feats[model.columns]).head(top)


def confidence_from_spread(spread: float) -> float:
    return float(max(0.0, 1.0 - min(1.0, spread / 0.3)))
