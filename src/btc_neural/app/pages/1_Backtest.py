"""Walk-forward backtest viewer: out-of-sample signals, per-cycle metrics, tax-aware simulation."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_neural import db
from btc_neural.app.common import BUY_10, PRICE, SELL_10, load_dataset, signal_figure
from btc_neural.backtest.simulator import simulate
from btc_neural.backtest.walkforward import leaderboard, usdzar
from btc_neural.config import load_config
from btc_neural.data import epochs

st.set_page_config(page_title="Backtest · BTC_Neural", page_icon="₿", layout="wide")
st.title("Walk-forward backtest")
st.caption("Each halving epoch is predicted by a model trained only on earlier epochs. "
           "Every signal shown here is out of sample.")

daily, _ = load_dataset()
close = daily["close"]
lb = leaderboard()
if lb.empty:
    st.info("No backtest runs yet. Run `uv run btcn backtest` or use the Train page.")
    st.stop()

prod_src = db.query("SELECT json_extract(config_json,'$.source_run_id') AS s FROM runs WHERE role='production' "
                    "ORDER BY created_at DESC LIMIT 1")
default_run = prod_src.s.iloc[0] if len(prod_src) and prod_src.s.iloc[0] in set(lb.run_id) else lb.run_id.iloc[0]
labels = {r.run_id: f"{r.model_type} / {r.feature_set} / {r.mode}  —  score {r.selection_score:.3f}" for r in lb.itertuples()}
run_id = st.selectbox("Run (best first; lower selection score is better)", list(labels), index=list(labels).index(default_run),
                      format_func=labels.get)

preds = db.query("SELECT date, fold, score, strength FROM predictions WHERE run_id=? ORDER BY date", (run_id,))
preds["date"] = pd.to_datetime(preds["date"])
preds = preds.set_index("date")
fr = db.query("SELECT fold, metric, value FROM fold_results WHERE run_id=?", (run_id,)).pivot(index="fold", columns="metric", values="value")

eps = epochs(end=close.index[-1], start=close.index[0])
shade = [(n, a, b) for n, a, b in eps[1:]]
show_halvings = st.toggle("Show halvings", value=False, key="bt_halvings",
                          help="Test epochs already start at a halving; the lines make that explicit.")
st.plotly_chart(signal_figure(close.loc[preds.index[0]:], preds["strength"], shade=shade, show_halvings=show_halvings,
                              title="Out-of-sample signals by test epoch (E1…E4)"), use_container_width=True)

st.subheader("Signal quality per test cycle")
cols = {
    "fold_score": "fold score ↓", "sell_n10": "Sell-10 days", "sell10_log_err": "Sell-10 dist. from top (log)",
    "sell10_days_err": "Sell-10 days from top", "sell_strong_wrong": "Sell ≥8 wrong side",
    "buy_n10": "Buy-10 days", "buy10_log_err": "Buy-10 dist. from low (log)", "buy10_days_err": "Buy-10 days from low",
    "buy_strong_wrong": "Buy ≥8 wrong side", "spearman_vs_position": "Rank corr. vs cycle position",
}
show = fr[[c for c in cols if c in fr]].rename(columns=cols)
st.dataframe(show.style.format("{:.2f}", na_rep="–"), use_container_width=True)
st.caption("Log distance 0.10 ≈ 10%, 0.29 ≈ 25% below a top / 33% above a low. Selection score = the worst of the "
           f"folds {', '.join(load_config()['folds']['selection_folds'])} (E1 has too little training data). "
           "E4 has no buy side yet: the current cycle's low is not confirmed.")

st.subheader("Strategy simulation (South African CGT)")
sc = load_config()["simulation"]
with st.form("sim"):
    c1, c2, c3, c4 = st.columns(4)
    fold = c1.selectbox("Period", ["all out-of-sample"] + list(fr.index), index=0)
    contrib = c2.number_input("Monthly contribution (USD)", 50, 100000, int(sc["monthly_contribution_usd"]), step=50)
    floor = c3.slider("Core floor (% of peak BTC never sold)", 0, 100, int(sc["core_floor_fraction"] * 100))
    lot = c4.selectbox("Lot method", ["fifo", "hifo", "average"], index=["fifo", "hifo", "average"].index(sc["tax"]["lot_method"]))
    c5, c6, c7, _ = st.columns(4)
    base = c5.slider("Share of contribution bought immediately", 0, 100, int(sc["base_dca_fraction"] * 100))
    marginal = c6.slider("Marginal income-tax rate %", 0, 45, int(sc["tax"]["marginal_rate"] * 100))
    fee = c7.number_input("Fee rate %", 0.0, 3.0, sc["fee_rate"] * 100, step=0.05)
    st.form_submit_button("Run simulation")

if fold == "all out-of-sample":
    a, b = preds.index[0], preds.index[-1]
    a = max(a, pd.Timestamp(load_config()["halvings"][1]))  # start with E2 (E1's model is weak)
else:
    p = preds[preds.fold == fold]
    a, b = p.index[0], p.index[-1]
params = {"monthly_contribution_usd": contrib, "core_floor_fraction": floor / 100, "base_dca_fraction": base / 100,
          "fee_rate": fee / 100, "tax": {**sc["tax"], "lot_method": lot, "marginal_rate": marginal / 100}}
fx = usdzar()
strat = simulate(close, preds["strength"], fx, a, b, params)
dca = simulate(close, preds["strength"], fx, a, b, params, dca_only=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("BTC held (strategy)", f"{strat.summary['final_btc']:.4f}",
          f"{(strat.summary['final_btc'] / dca.summary['final_btc'] - 1) * 100:+.1f}% vs DCA")
m2.metric("Value (strategy)", f"${strat.summary['final_value_usd']:,.0f}",
          f"{(strat.summary['final_value_usd'] / dca.summary['final_value_usd'] - 1) * 100:+.1f}% vs DCA")
m3.metric("Tax (paid + accrued)", f"${strat.summary['tax_paid_usd'] + strat.summary['tax_accrued_usd']:,.0f}")
m4.metric("Avg sell price", f"${strat.summary['avg_sell_price']:,.0f}" if strat.summary["n_sells"] else "no sells")

fig = go.Figure()
fig.add_trace(go.Scatter(x=strat.equity.index, y=strat.equity["value_usd"], name="Strategy value", line=dict(color=BUY_10, width=2)))
fig.add_trace(go.Scatter(x=dca.equity.index, y=dca.equity["value_usd"], name="Plain DCA value", line=dict(color=PRICE, width=2)))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="USD", hovermode="x unified",
                  legend=dict(orientation="h", y=1.08), title="Portfolio value (USD)")
st.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=strat.equity.index, y=strat.equity["btc"], name="Strategy BTC", line=dict(color=BUY_10, width=2)))
fig2.add_trace(go.Scatter(x=dca.equity.index, y=dca.equity["btc"], name="Plain DCA BTC", line=dict(color=PRICE, width=2)))
fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="BTC", hovermode="x unified",
                   legend=dict(orientation="h", y=1.1), title="BTC held")
st.plotly_chart(fig2, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Tax by SA tax year** (rand)")
    if len(strat.taxes):
        st.dataframe(strat.taxes.style.format({"net_gain_zar": "R{:,.0f}", "tax_zar": "R{:,.0f}", "tax_usd": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Sells**")
    if len(strat.trades):
        s = strat.trades[strat.trades.side == "sell"][["date", "btc", "price", "usd", "gain_zar"]]
        st.dataframe(s.style.format({"btc": "{:.4f}", "price": "${:,.0f}", "usd": "${:,.0f}", "gain_zar": "R{:,.0f}"}),
                     use_container_width=True, hide_index=True)
st.caption("Simplified SA capital gains tax model: gains measured in rand at FRED USD/ZAR, 40% inclusion × marginal rate, "
           "annual exclusion, losses carried forward. Not tax advice.")
with st.expander("Run configuration"):
    st.json(json.loads(db.query("SELECT config_json FROM runs WHERE run_id=?", (run_id,)).config_json.iloc[0]))
