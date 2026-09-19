"""BTC_Neural dashboard: today's signal."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from btc_neural import db, pipeline
from btc_neural.app.common import (add_forecast_overlay, drivers_figure, fmt_strength, gauge_html, load_dataset,
                                   signal_figure)
from btc_neural.forecast import forecast_next, hindcasts

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

c_years, c_fc = st.columns([3, 1])
years = c_years.slider("Years shown", 1, 16, 6)
show_fc = c_fc.toggle("Show next-extreme forecast", value=True)
start = date - pd.Timedelta(days=365 * years)
fig = signal_figure(daily["close"].loc[start:], prod["strength"].loc[start:],
                    title="Production model signals (in-sample before training cutoff; see Backtest for out-of-sample)")
fc = forecast_next(daily, feats) if show_fc else None
if fc is not None:
    x_end = add_forecast_overlay(fig, fc, date, x_start=start)
    fig.update_xaxes(range=[start, x_end + pd.Timedelta(days=45 + 25 * years)])  # room for the label
st.plotly_chart(fig, use_container_width=True)

if fc is not None:
    st.subheader(f"Next cycle {fc.kind}: statistical forecast")
    d1, d2 = fc.date_interval(1)
    p1, p2 = fc.price_interval(1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Most likely date", fc.date_mu.strftime("%Y-%m-%d"))
    m1.caption(f"±{fc.date_sd:.0f} days (1σ): {d1:%b %d} – {d2:%b %d, %Y}")
    m2.metric("Most likely price", f"${fc.price_mu:,.0f}")
    m2.caption(f"68% range: \\${p1:,.0f} – \\${p2:,.0f}")
    m3.metric("Distance from today", f"{(fc.price_mu / price - 1) * 100:+.0f}%")
    m3.caption(f"{(fc.date_mu - date).days} days ahead of the latest close (\\${price:,.0f})")
    m4.metric("P(already happened)", f"{fc.p_already:.0%}", help="Share of the date Gaussian before today")
    st.caption(f"Follows the cycle {fc.after.kind} of {fc.after.date:%Y-%m-%d} (\\${fc.after.price:,.0f}). Shading intensity is "
               "the Gaussian density: the vertical band is the date, the horizontal band the price (log scale). "
               "This is a separate cycle-analog model built from 3–4 past cycles, not the neural net. It assumes the "
               "4-year rhythm continues.")
    with st.expander("How the forecast is built, and how it did on past cycles"):
        est = pd.DataFrame(
            [{"estimator": e.name, "type": "date", "estimate": (pd.Timestamp("1970-01-01") + pd.Timedelta(days=e.mu)).strftime("%Y-%m-%d"),
              "1σ": f"±{e.sd:.0f} days", "basis": e.note} for e in fc.time_estimates]
            + [{"estimator": e.name, "type": "price", "estimate": f"${np.exp(e.mu):,.0f}",
                "1σ": f"×/÷ {np.exp(e.sd):.2f}", "basis": e.note} for e in fc.price_estimates])
        st.dataframe(est, use_container_width=True, hide_index=True)
        st.markdown("**Hindcasts**: each past extreme forecast using only earlier cycles, made 180 days after the "
                    "preceding extreme.")
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
