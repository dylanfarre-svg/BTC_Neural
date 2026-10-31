"""The swing research: pre-registered selection, section 9 checks, ablations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_neural.app.common import GRID
from btc_neural.app_trade.helpers import BENCH, STRATEGY, pct, report, tick
from btc_neural.config import ROOT

st.set_page_config(page_title="Research · BTC_Neural swing", page_icon="↔", layout="wide")
st.title("Swing research")
st.caption("Selection rule frozen in PLAN_SWING.md §15 before the grid ran. The holdout year was never used to select.")

rep = report()
if rep is None:
    st.info("No research report yet. Run `uv run btcn trade research`.")
    st.stop()

ch = rep["chosen"]
spread = (f"seed sd {ch['selection_sd']:.3f}, {int(ch['seeds'])} seeds" if int(ch["seeds"]) > 1
          else "deterministic, one fit")
st.markdown(f"**Promoted:** `{ch['model_type']}` · `{ch['feature_set']}` · quantile {ch['quantile']} · "
            f"selection score {ch['selection']:+.3f} ({spread}) · {ch['tpy']:.1f} trades/year")

# --- section 9 --------------------------------------------------------------------------------------
st.subheader("§9 ship criteria")
c = rep["checks"]
names = {"expectancy_lb95_positive": "Mean net per trade, 95% lower bound > 0",
         "beats_random_entry_p95": "Beats 95% of random-entry runs",
         "regimes_3_of_4": "Positive in ≥ 3 of 4 regimes", "median_hold_7_28": "Median hold inside 7–28 days",
         "beats_bh_sharpe": "Beats buy-and-hold: Sharpe",
         "beats_bh_total_capital": "Beats buy-and-hold: total return (capital tax)",
         "beats_bh_total_revenue": "Beats buy-and-hold: total return (revenue tax)",
         "beats_bh_rptim_capital": "Beats buy-and-hold: return per year in market (capital)",
         "beats_bh_rptim_revenue": "Beats buy-and-hold: return per year in market (revenue)"}
st.dataframe(pd.DataFrame({"criterion": list(names.values()),
                           "full (2015+)": [tick(c["full"][k]) for k in names],
                           "modern (2019+)": [tick(c["modern"][k]) for k in names]}), hide_index=True)
st.markdown(f"**Ship under capital-gains treatment:** {tick(c['ship_capital'])} · "
            f"**under revenue treatment:** {tick(c['ship_revenue'])}")
d = rep.get("deflated_sharpe", {})
if d:
    st.caption(f"Deflated Sharpe ratio over {d['n_trials']} population-A configurations: "
               f"P(true Sharpe > best unskilled) = {d['dsr_full']:.2f}.")

# --- windows ----------------------------------------------------------------------------------------
st.subheader("By window")
rows = []
for w in ("full", "modern", "holdout"):
    m = rep["main"]["windows"][w]
    rows.append({"window": f"{w} ({m['start']} → {m['end']})", "trades": m.get("n_trades", 0),
                 "per year": round(m.get("trades_per_year", 0), 1), "hit rate": pct(m.get("hit_rate"), 0),
                 "mean net": pct(m.get("mean_net"), 2), "Sharpe": round(m["sharpe"], 2), "B&H Sharpe": round(m["bh_sharpe"], 2),
                 "total pre-tax": f"{m['total_pre_tax']:.2f}×", "B&H after tax": f"{m['bh_total_after_tax']:.2f}×",
                 "random p": f"{rep['random_entry'][w].get('p_total', float('nan')):.2f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True)

# --- sides ------------------------------------------------------------------------------------------
st.subheader("Long and short, separately")
rows = []
for side, r in rep["sides"].items():
    for w in ("full", "modern", "holdout"):
        m = r["windows"][w]
        rows.append({"side": side, "window": w, "trades": m.get("n_trades", 0), "hit rate": pct(m.get("hit_rate"), 0),
                     "mean net": pct(m.get("mean_net"), 2), "LB95": pct(m.get("mean_net_lb95"), 2),
                     "Sharpe": round(m["sharpe"], 2), "total pre-tax": f"{m['total_pre_tax']:.2f}×"})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("Shorts are simulated only from 2019 (liquid perp venues); before that the strategy is long/flat.")

# --- regimes ----------------------------------------------------------------------------------------
reg = rep["main"]["windows"]["full"].get("regimes", {})
if reg:
    st.subheader("Regimes (full window)")
    st.dataframe(pd.DataFrame([{"regime": k.replace("_", " · "), "trades": v["n"], "mean net": pct(v["mean_net"], 2)}
                               for k, v in reg.items()]), hide_index=True)
    st.caption("Hindsight cycle leg (bull = low→top, bear = top→low) × 30-day volatility above/below its median.")

# --- fee sweep --------------------------------------------------------------------------------------
st.subheader("Fee sensitivity (per side, plus 0.05% slippage)")
fs = rep["fee_sweep"]
fees = [float(k) for k in fs]
fig = go.Figure()
for w, dash in (("full", "solid"), ("modern", "dash"), ("holdout", "dot")):
    fig.add_trace(go.Scatter(x=[f * 100 for f in fees], y=[fs[k][w]["sharpe"] for k in fs], name=f"strategy, {w}",
                             mode="lines+markers", line=dict(color=STRATEGY, width=2, dash=dash), marker=dict(size=8),
                             hovertemplate="fee %{x:.2f}% → Sharpe %{y:.2f}<extra>" + w + "</extra>"))
fig.add_hline(y=rep["main"]["windows"]["full"]["bh_sharpe"], line=dict(color=BENCH, width=2),
              annotation_text="buy-and-hold Sharpe, full window", annotation_position="top left")
fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=40), legend=dict(orientation="h", y=-0.25),
                  xaxis=dict(title="fee per side (%)", gridcolor=GRID), yaxis=dict(title="Sharpe", gridcolor=GRID))
st.plotly_chart(fig)
st.dataframe(pd.DataFrame([{"fee per side": f"{float(k):.2%}", **{f"{w} Sharpe": round(v[w]["sharpe"], 2) for w in v},
                            **{f"{w} total": f"{v[w]['total_pre_tax']:.2f}×" for w in v}} for k, v in fs.items()]),
             hide_index=True)
fo = rep.get("funding_off", {})
if fo:
    st.caption(f"Funding switched off (shorts as if borrow were free): full-window Sharpe {fo['full']['sharpe']:.2f} "
               f"vs {rep['main']['windows']['full']['sharpe']:.2f} with recorded funding.")

# --- ablations --------------------------------------------------------------------------------------
st.subheader("Ablations")
sr = rep.get("stop_ratio", {})
if sr:
    base = rep["main"]["windows"]
    rows = [{"stop ratio": "0.6 (configured)", **{f"{w} Sharpe": round(base[w]["sharpe"], 2) for w in base},
             "median hold": round(base["full"].get("median_hold", 0), 1)}]
    for k, v in sr.items():
        rows.append({"stop ratio": k, **{f"{w} Sharpe": round(v[w]["sharpe"], 2) for w in v},
                     "median hold": round(v["full"].get("median_hold") or 0, 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
pb = rep.get("population_b", {})
if pb:
    st.markdown("**Late feature sets vs `fast_core` over the same window** (adopt only if > 2 standard errors better)")
    st.dataframe(pd.DataFrame([{"config": k, "window from": v["window_from"], "late Sharpe": round(v["late_sharpe"], 2),
                                "fast_core Sharpe": round(v["core_sharpe_same_window"], 2), "difference": round(v["diff"], 2),
                                "se": None if v["se"] is None else round(v["se"], 2), "adopt": tick(v["adopt"])}
                               for k, v in pb.items()]), hide_index=True)

# --- leaderboard ------------------------------------------------------------------------------------
st.subheader("Leaderboard (seed means)")
grid = ROOT / "runs" / "swing_grid.csv"
if grid.exists():
    from btc_neural.trade.research import POP_A, POP_B, config_table
    t = config_table(pd.read_csv(grid))
    cols = ["model_type", "feature_set", "quantile", "selection", "selection_sd", "seeds", "full_sharpe",
            "modern_sharpe", "holdout_sharpe", "tpy", "mean_net", "eligible", "eval_from"]
    for name, pop in (("Population A — valid from 2015 (selection pool)", POP_A),
                      ("Population B — valid from 2019+ (never ranked against A)", POP_B)):
        st.markdown(f"**{name}**")
        st.dataframe(t[t.feature_set.isin(pop)][cols].round(3), hide_index=True, height=320)
