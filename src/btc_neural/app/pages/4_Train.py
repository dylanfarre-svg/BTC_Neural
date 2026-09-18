"""Run walk-forward experiments and promote a run to production."""

import contextlib
import io

import streamlit as st

from btc_neural import pipeline
from btc_neural.app.common import clear_caches
from btc_neural.backtest.walkforward import leaderboard, run_walkforward
from btc_neural.features import FEATURE_SETS
from btc_neural.models import MODEL_TYPES

st.set_page_config(page_title="Train · BTC_Neural", page_icon="₿", layout="wide")
st.title("Train and promote")

st.subheader("1 · Run a walk-forward experiment")
with st.form("wf"):
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", MODEL_TYPES, index=MODEL_TYPES.index("lgbm"))
    fs = c2.selectbox("Feature set", list(FEATURE_SETS), index=0)
    modes = c3.multiselect("Strength modes", ["trailing", "hybrid", "frozen", "both"], default=["trailing", "hybrid"])
    st.caption("NN runs take about a minute (Apple GPU). Training uses only data before each test epoch.")
    go = st.form_submit_button("Run backtest")
if go:
    buf = io.StringIO()
    with st.spinner("Running walk-forward…"), contextlib.redirect_stdout(buf):
        res = run_walkforward(model, fs, modes=modes, store=True, verbose=True)
    st.code(buf.getvalue())
    for mode, r in res.items():
        st.success(f"{mode}: selection score {r.selection_score:.3f} ({r.run_id})")
    clear_caches()

st.subheader("2 · Promote to production")
lb = leaderboard()
if lb.empty:
    st.info("Run a backtest first.")
    st.stop()
best = pipeline.best_run()
best_nn = pipeline.best_run(nn_only=True)
st.write(f"Best overall: **{best.model_type} / {best.feature_set} / {best['mode']}** (score {best.selection_score:.3f})")
if best_nn is not None:
    st.write(f"Best neural net: **{best_nn.model_type} / {best_nn.feature_set} / {best_nn['mode']}** (score {best_nn.selection_score:.3f})")
labels = {r.run_id: f"{r.model_type} / {r.feature_set} / {r.mode} — {r.selection_score:.3f}" for r in lb.itertuples()}
choice = st.selectbox("Run to promote as production", list(labels), format_func=labels.get)
c1, c2 = st.columns(2)
if c1.button("Retrain on all data → production", type="primary"):
    buf = io.StringIO()
    with st.spinner("Training on all data…"), contextlib.redirect_stdout(buf):
        pipeline.train_production(choice, role="production")
    st.success(buf.getvalue())
    clear_caches()
if c2.button("Retrain best NN → reference", disabled=best_nn is None):
    buf = io.StringIO()
    with st.spinner("Training NN reference…"), contextlib.redirect_stdout(buf):
        pipeline.train_production(None, role="reference_nn")
    st.success(buf.getvalue())
    clear_caches()
