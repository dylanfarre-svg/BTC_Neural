"""Feature explorer: any feature against log price, with cycle extremes marked."""

import plotly.graph_objects as go
import streamlit as st

from btc_neural.app.common import (BUY_10, PRICE, add_halving_lines, add_hover_spine, extremes_df, load_dataset,
                                   stacked_layout)
from btc_neural.features import FEATURE_GROUPS, FEATURE_SETS, columns_for

st.set_page_config(page_title="Features · BTC_Neural", page_icon="₿", layout="wide")
st.title("Feature explorer")
st.caption("All features are causal: each value uses only data available at the end of that day.")

daily, feats = load_dataset()
groups = sorted(set(FEATURE_GROUPS.get(c, "?") for c in feats.columns))
c1, c2 = st.columns([1, 2])
g = c1.selectbox("Group", ["cyclenorm (set)"] + groups)
opts = columns_for("cyclenorm", list(feats.columns)) if g == "cyclenorm (set)" else [c for c in feats.columns if FEATURE_GROUPS.get(c) == g]
feat = c2.selectbox("Feature", opts)
show_halvings = st.toggle("Show halvings", value=False, key="feat_halvings")

fig = go.Figure()
fig.add_trace(go.Scatter(x=daily.index, y=daily["close"], name="BTC/USD", line=dict(color=PRICE, width=1.5),
                         hovertemplate="$%{y:,.0f}"))
fig.add_trace(go.Scatter(x=feats.index, y=feats[feat], name=feat, yaxis="y2", line=dict(color=BUY_10, width=1.5),
                         hovertemplate="%{y:.3f}"))
for _, e in extremes_df().iterrows():
    fig.add_vline(x=e.date, line=dict(color="rgba(227,73,72,0.35)" if e.kind == "top" else "rgba(42,120,214,0.35)", dash="dot"))
add_hover_spine(fig, daily.index[0], max(daily.index[-1], feats.index[-1]),
                float((daily["close"].min() * daily["close"].max()) ** 0.5))
if show_halvings:
    add_halving_lines(fig)
stacked_layout(fig, "USD (log)", feat, 680, split=0.45)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("Dotted lines: cycle tops (red) and lows (blue), hindsight reference only.")

s = feats[feat].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest", f"{s.iloc[-1]:.3f}")
c2.metric("First available", s.index[0].strftime("%Y-%m-%d"))
c3.metric("Percentile of history", f"{(s <= s.iloc[-1]).mean() * 100:.0f}%")
c4.metric("In feature sets", ", ".join(k for k in FEATURE_SETS if feat in columns_for(k, list(feats.columns))))
