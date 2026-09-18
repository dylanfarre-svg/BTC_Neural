"""All walk-forward runs, ranked by selection score (worst selection fold; lower is better)."""

import streamlit as st

from btc_neural import db
from btc_neural.backtest.walkforward import leaderboard

st.set_page_config(page_title="Leaderboard · BTC_Neural", page_icon="₿", layout="wide")
st.title("Model leaderboard")
st.caption("Selection score = the worst per-cycle score across the selection folds. Each fold score combines the "
           "median distance of strength-10 signals from the true extreme, the share of strong signals on the wrong "
           "side of the cycle, and a small penalty for missing the 10-day budget. Lower is better.")

lb = leaderboard()
if lb.empty:
    st.info("No runs yet.")
    st.stop()
c1, c2, c3 = st.columns(3)
models = c1.multiselect("Model", sorted(lb.model_type.unique()), default=sorted(lb.model_type.unique()))
sets = c2.multiselect("Feature set", sorted(lb.feature_set.unique()), default=sorted(lb.feature_set.unique()))
modes = c3.multiselect("Strength mode", sorted(lb["mode"].dropna().unique()), default=sorted(lb["mode"].dropna().unique()))
v = lb[lb.model_type.isin(models) & lb.feature_set.isin(sets) & lb["mode"].isin(modes)]

prod = db.query("SELECT role, json_extract(config_json,'$.source_run_id') AS src FROM runs WHERE role IN ('production','reference_nn') "
                "ORDER BY created_at DESC")
tags = {}
for r in prod.drop_duplicates("role").itertuples():
    tags[r.src] = r.role
v = v.assign(role=v.run_id.map(tags).fillna(""))
cols = ["role", "model_type", "feature_set", "mode", "selection_score"] + [c for c in ["E1", "E2", "E3", "E4"] if c in v]
cfg = {c: st.column_config.NumberColumn(c, format="%.3f") for c in cols[5:]}
cfg["selection_score"] = st.column_config.ProgressColumn(
    "selection score ↓", format="%.3f", min_value=0.0, max_value=float(max(2.5, v.selection_score.max())))
st.dataframe(v[cols + ["run_id"]], column_config=cfg, use_container_width=True, hide_index=True, height=620)
