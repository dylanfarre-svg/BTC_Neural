"""Shared helpers for the Streamlit pages."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Diverging buy/sell encoding (blue = buy, red = sell), neutral grey price line.
BUY_10, BUY_STRONG = "#2a78d6", "#86b6ef"
SELL_10, SELL_STRONG = "#e34948", "#f0a09f"
PRICE = "#8a8986"
TOP_MARK, LOW_MARK = "#e34948", "#2a78d6"


def fmt_strength(s: int) -> str:
    s = int(s)
    return "Neutral 0" if s == 0 else (f"Buy {s}" if s > 0 else f"Sell {-s}")


@st.cache_resource(ttl=3600, show_spinner="Loading data and features…")
def load_dataset():
    from btc_neural.backtest.walkforward import dataset
    dataset.cache_clear()
    return dataset()


def clear_caches():
    st.cache_resource.clear()
    st.cache_data.clear()


@st.cache_data(ttl=3600)
def extremes_df() -> pd.DataFrame:
    from btc_neural.data import cycle_extremes
    daily, _ = load_dataset()
    ex = cycle_extremes(daily["close"])
    return pd.DataFrame([{"kind": e.kind, "date": e.date, "price": e.price} for e in ex])


def signal_figure(close: pd.Series, strength: pd.Series, title: str = "", show_extremes: bool = True,
                  min_level: int = 7, shade: list[tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
                  height: int = 520) -> go.Figure:
    s = strength.reindex(close.index).fillna(0).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="BTC/USD", mode="lines",
                             line=dict(color=PRICE, width=1.6),
                             hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"))
    groups = [
        ("Buy 10", s >= 10, BUY_10, 10, "triangle-up"),
        (f"Buy {min_level}-9", (s >= min_level) & (s < 10), BUY_STRONG, 7, "triangle-up"),
        ("Sell 10", s <= -10, SELL_10, 10, "triangle-down"),
        (f"Sell {min_level}-9", (s <= -min_level) & (s > -10), SELL_STRONG, 7, "triangle-down"),
    ]
    for name, m, color, size, sym in groups:
        if m.any():
            fig.add_trace(go.Scatter(
                x=close.index[m], y=close[m], mode="markers", name=name,
                marker=dict(color=color, size=size, symbol=sym, line=dict(width=1, color="rgba(255,255,255,0.8)")),
                customdata=s[m].values,
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<br>strength %{customdata}<extra>" + name + "</extra>"))
    if show_extremes:
        ex = extremes_df()
        ex = ex[(ex.date >= close.index[0]) & (ex.date <= close.index[-1])]
        for _, e in ex.iterrows():
            fig.add_annotation(x=e.date, y=np.log10(e.price), text=("cycle top" if e.kind == "top" else "cycle low"),
                               showarrow=True, arrowhead=0, ay=(-28 if e.kind == "top" else 28), font=dict(size=10),
                               arrowcolor=TOP_MARK if e.kind == "top" else LOW_MARK)
    for name, a, b in shade or []:
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(128,128,128,0.07)", line_width=0,
                      annotation_text=name, annotation_position="top left", annotation_font_size=10)
    fig.update_yaxes(type="log", title="USD (log)", gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.10)")
    fig.update_layout(title=dict(text=title, y=0.98, yanchor="top"), height=height,
                      margin=dict(l=10, r=10, t=80 if title else 40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest")
    return fig


def drivers_figure(drivers: pd.Series, height: int = 300) -> go.Figure:
    d = drivers.iloc[::-1]
    colors = [BUY_10 if v > 0 else SELL_10 for v in d.values]
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="contribution to score (+ buy / − sell)", zeroline=True,
                                 zerolinecolor="rgba(128,128,128,0.5)", gridcolor="rgba(128,128,128,0.15)"))
    return fig


def gauge_html(strength: int, label: str) -> str:
    s = int(strength)
    color = BUY_10 if s > 0 else SELL_10 if s < 0 else PRICE
    pct = (s + 10) / 20 * 100
    return f"""
<div style="padding:4px 0 10px 0">
  <div style="font-size:0.85rem;opacity:0.75">{label}</div>
  <div style="font-size:2.4rem;font-weight:700;color:{color};line-height:1.1">{fmt_strength(s)}</div>
  <div style="position:relative;height:10px;border-radius:5px;margin-top:8px;
      background:linear-gradient(90deg,{SELL_10} 0%,{SELL_STRONG} 30%,rgba(128,128,128,0.35) 50%,{BUY_STRONG} 70%,{BUY_10} 100%)">
    <div style="position:absolute;left:calc({pct}% - 2px);top:-4px;width:4px;height:18px;border-radius:2px;background:currentColor"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;opacity:0.6"><span>Sell 10</span><span>0</span><span>Buy 10</span></div>
</div>"""
