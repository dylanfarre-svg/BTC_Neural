"""Feature explorer: any feature against log price, with cycle extremes marked."""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from btc_neural.app.common import BUY_10, PRICE, extremes_df, load_dataset
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

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.45])
fig.add_trace(go.Scatter(x=daily.index, y=daily["close"], name="BTC/USD", line=dict(color=PRICE, width=1.5),
                         hovertemplate="%{x|%Y-%m-%d} $%{y:,.0f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=feats.index, y=feats[feat], name=feat, line=dict(color=BUY_10, width=1.5),
                         hovertemplate="%{x|%Y-%m-%d} %{y:.3f}<extra></extra>"), row=2, col=1)
for _, e in extremes_df().iterrows():
    fig.add_vline(x=e.date, line=dict(color="rgba(227,73,72,0.35)" if e.kind == "top" else "rgba(42,120,214,0.35)", dash="dot"))
fig.update_yaxes(type="log", title="USD (log)", row=1, col=1)
fig.update_yaxes(title=feat, row=2, col=1)
fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption("Dotted lines: cycle tops (red) and lows (blue), hindsight reference only.")

s = feats[feat].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest", f"{s.iloc[-1]:.3f}")
c2.metric("First available", s.index[0].strftime("%Y-%m-%d"))
c3.metric("Percentile of history", f"{(s <= s.iloc[-1]).mean() * 100:.0f}%")
c4.metric("In feature sets", ", ".join(k for k in FEATURE_SETS if feat in columns_for(k, list(feats.columns))))
