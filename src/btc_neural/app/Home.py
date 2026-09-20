"""BTC_Neural dashboard: today's signal."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from btc_neural import db, pipeline
from btc_neural.app.common import (SOURCE_COLORS, add_forecast_marker, add_forecast_overlay, drivers_figure,
                                   fmt_strength, gauge_html, load_dataset, signal_figure)
from btc_neural.forecast import forecast_next, hindcasts, model_forecast

st.set_page_config(page_title="BTC_Neural", page_icon="₿", layout="wide")
st.title("BTC_Neural: cycle signal")
st.caption("Signed strength from −10 (max sell) to +10 (max buy). Research tool, not financial advice.")

daily, feats = load_dataset()
date, price = daily.index[-1], float(daily["close"].iloc[-1])


@st.cache_data(ttl=3600, show_spinner="Scoring history…")
def history(role: str, stamp: str):
    h = pipeline.signal_history(role, feats)
    if h is None:
        return None, None
    return h, h.attrs


@st.cache_data(ttl=3600)
def drivers(role: str, stamp: str):
    return pipeline.explain_today(role, feats, top=8)


stamp = date.strftime("%Y-%m-%d") + "|" + "|".join(
    db.query("SELECT COALESCE(MAX(created_at),'') AS c FROM runs WHERE role=?", (r,)).c.iloc[0] for r in pipeline.ROLES)
prod, prod_attrs = history("production", stamp)
ref, ref_attrs = history("reference_nn", stamp)

if prod is None:
    st.warning("No production model yet. Run the backtest grid and `uv run btcn train-production` "
               "(or use the **Train** page).")
    st.stop()

meta = prod_attrs["meta"]
c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1])
with c1:
    st.markdown(gauge_html(int(prod["strength"].iloc[-1]), f"Production · {meta['model_type']}/{meta['feature_set']}"),
                unsafe_allow_html=True)
with c2:
    if ref is not None:
        st.markdown(gauge_html(int(ref["strength"].iloc[-1]), "Neural-net reference"), unsafe_allow_html=True)
with c3:
    st.metric("BTC/USD", f"${price:,.0f}", f"{(price / daily['close'].iloc[-8] - 1) * 100:+.1f}% 7d")
    st.metric("Data through", date.strftime("%Y-%m-%d"))
with c4:
    conf = pipeline.confidence_from_spread(float(prod["spread"].iloc[-1]))
    st.metric("Confidence", "high" if conf >= 0.66 else "medium" if conf >= 0.33 else "low",
              help="Agreement between ensemble members / seeds. Single models report high.")
    st.metric("Raw score", f"{prod['score'].iloc[-1]:+.3f}", help="Model output; +1 ≈ cycle low, −1 ≈ cycle top")

STAT = "Cycle-analog (statistical)"


@st.cache_data(ttl=3600, show_spinner=False)
def forecasts(stamp: str) -> dict:
    """Next-extreme forecast per source: the statistical cycle-analog model and every saved model
    (each from its own out-of-sample score history)."""
    out = {STAT: forecast_next(daily, feats)}
    for r in pipeline.all_roles().itertuples():
        f = model_forecast(pipeline.oos_scores(r.source_run_id), daily)
        if f is not None:
            out[f"{r.role} · {r.model_type}/{r.feature_set}"] = f
    return {k: v for k, v in out.items() if v is not None}


fcs = forecasts(stamp)
c_years, c_fc = st.columns([3, 1])
years = c_years.slider("Years shown", 1, 16, 6)
show_fc = c_fc.toggle("Show next-extreme forecasts", value=True)
picked: list[str] = []
shaded = None
if show_fc and fcs:
    default = [k for k in fcs if k == STAT or k.startswith("production")]
    picked = st.multiselect("Forecast of the next cycle extreme — models to display", list(fcs), default=default,
                            help="Each saved model forecasts from its own out-of-sample history; the statistical "
                                 "model uses cycle gaps, halving timing, MVRV and the 200-week average.")
    shaded = st.selectbox("Shade the Gaussian for", picked, index=0) if picked else None

start = date - pd.Timedelta(days=365 * years)
fig = signal_figure(daily["close"].loc[start:], prod["strength"].loc[start:],
                    title="Production model signals (in-sample before training cutoff; see Backtest for out-of-sample)")
x_end = date
for i, name in enumerate(picked):
    fc_i = fcs[name]
    if name == shaded:
        x_end = max(x_end, add_forecast_overlay(fig, fc_i, date, x_start=start))
    else:
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        x_end = max(x_end, add_forecast_marker(fig, fc_i, name.split(" · ")[0], color))
if picked:
    fig.update_xaxes(range=[start, x_end + pd.Timedelta(days=45 + 25 * years)])  # room for the label
st.plotly_chart(fig, use_container_width=True)

if picked:
    kinds = {f.kind for f in fcs.values()}
    st.subheader(f"Next cycle {'/'.join(sorted(kinds))}: forecasts by model")
    rows = []
    for name in picked:
        f_ = fcs[name]
        d1, d2 = f_.date_interval(1)
        p1, p2 = f_.price_interval(1)
        rows.append({"source": name, "kind": f_.kind, "date": f_.date_mu.strftime("%Y-%m-%d"),
                     "date 68%": f"{d1:%d %b} – {d2:%d %b %Y}", "price": f_.price_mu,
                     "price 68%": f"${p1:,.0f} – ${p2:,.0f}", "vs today": f_.price_mu / price - 1,
                     "days ahead": (f_.date_mu - date).days, "P(already happened)": f_.p_already})
    st.dataframe(pd.DataFrame(rows).style.format({"price": "${:,.0f}", "vs today": "{:+.0%}",
                                                  "P(already happened)": "{:.0%}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Shading intensity is the Gaussian density: the vertical band is the date, the horizontal band the "
               "price (log scale); other models are drawn as a marker with 1σ error bars. These forecasts come from "
               "3 past cycles, so treat the ranges as optimistic. They assume the 4-year rhythm continues.")
    with st.expander("How each forecast is built, and how the statistical one did on past cycles"):
        for name in picked:
            f_ = fcs[name]
            st.markdown(f"**{name}** — follows the cycle {f_.after.kind} of {f_.after.date:%Y-%m-%d} "
                        f"(\\${f_.after.price:,.0f})")
            est = pd.DataFrame(
                [{"estimator": e.name, "type": "date",
                  "estimate": (pd.Timestamp("1970-01-01") + pd.Timedelta(days=e.mu)).strftime("%Y-%m-%d"),
                  "1σ": f"±{e.sd:.0f} days", "basis": e.note} for e in f_.time_estimates]
                + [{"estimator": e.name, "type": "price", "estimate": f"${np.exp(e.mu):,.0f}",
                    "1σ": f"×/÷ {np.exp(e.sd):.2f}", "basis": e.note} for e in f_.price_estimates])
            st.dataframe(est, use_container_width=True, hide_index=True)
        st.markdown("**Hindcasts of the statistical model**: each past extreme forecast using only earlier cycles, "
                    "made 180 days after the preceding extreme.")
        hc = hindcasts(daily, feats)
        st.dataframe(hc.style.format({"date z": "{:+.2f}", "price z": "{:+.2f}", "forecast price": "${:,.0f}",
                                      "actual price": "${:,.0f}", "price error": "{:+.0%}"}),
                     use_container_width=True, hide_index=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("What drives today's reading")
    d = drivers("production", stamp)
    if len(d):
        st.plotly_chart(drivers_figure(d), use_container_width=True)
        st.caption("Positive bars push toward BUY, negative toward SELL.")
with right:
    st.subheader("Last 14 days")
    t = pd.DataFrame({"price": daily["close"], "production": prod["strength"],
                      "nn_reference": ref["strength"] if ref is not None else None}).iloc[-14:][::-1]
    t.index = t.index.strftime("%Y-%m-%d")
    t["production"] = t["production"].map(fmt_strength)
    if ref is not None:
        t["nn_reference"] = t["nn_reference"].map(fmt_strength)
    st.dataframe(t.style.format({"price": "${:,.0f}"}), use_container_width=True)

with st.expander("Production model details"):
    st.json({k: v for k, v in meta.items() if k != "calibration"})
    st.caption(f"Run: {prod_attrs['run_id']}")
