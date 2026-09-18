"""BTC_Neural dashboard: today's signal."""

import json

import pandas as pd
import streamlit as st

from btc_neural import db, pipeline
from btc_neural.app.common import drivers_figure, fmt_strength, gauge_html, load_dataset, signal_figure

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

years = st.slider("Years shown", 1, 16, 6)
start = date - pd.Timedelta(days=365 * years)
trained = pd.Timestamp(meta["trained_through"])
st.plotly_chart(signal_figure(daily["close"].loc[start:], prod["strength"].loc[start:],
                              title="Production model signals (in-sample before training cutoff; see Backtest for out-of-sample)"),
                use_container_width=True)

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
