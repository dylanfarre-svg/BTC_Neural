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


# One-hue ramps per side of the diverging buy/sell scale (level 1 -> 10).
BUY_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
SELL_RAMP = ["#fbd5d4", "#f8c2c1", "#f5aeac", "#f19a98", "#ec8583", "#e76f6d", "#e34948", "#c93b3a", "#a82f2e", "#862423"]
GRID = "rgba(128,128,128,0.15)"


def strength_colors(s: pd.Series) -> list[str]:
    return [BUY_RAMP[v - 1] if v > 0 else SELL_RAMP[-v - 1] if v < 0 else "rgba(128,128,128,0.3)" for v in s.astype(int)]


def stacked_layout(fig: go.Figure, top_title: str, bottom_title: str, height: int, title: str = "",
                   top_log: bool = True, split: float = 0.32) -> go.Figure:
    """Two panels on ONE shared x axis, so a single hover line/tooltip spans both panels."""
    fig.update_layout(
        title=dict(text=title, y=0.98, yanchor="top"), height=height,
        margin=dict(l=10, r=10, t=80 if title else 40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified", hoversubplots="axis",
        xaxis=dict(anchor="y2", gridcolor=GRID, hoverformat="%Y-%m-%d", showspikes=True, spikemode="across", spikesnap="cursor",
                   spikethickness=1, spikedash="solid", spikecolor="rgba(128,128,128,0.7)"),
        yaxis=dict(domain=[split + 0.06, 1.0], type="log" if top_log else "linear", title=top_title, gridcolor=GRID),
        yaxis2=dict(domain=[0.0, split], title=bottom_title, gridcolor=GRID, zerolinecolor="rgba(128,128,128,0.5)"),
    )
    return fig


def signal_figure(close: pd.Series, strength: pd.Series, title: str = "", show_extremes: bool = True,
                  min_level: int = 7, shade: list[tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
                  height: int = 680) -> go.Figure:
    """Log price with strong-signal markers, plus a panel with the full daily strength (-10..+10)."""
    s = strength.reindex(close.index).fillna(0).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="BTC/USD", mode="lines",
                             line=dict(color=PRICE, width=1.6), hovertemplate="$%{y:,.0f}"))
    groups = [
        ("Buy 10", s >= 10, BUY_10, 10, "triangle-up"),
        (f"Buy {min_level}-9", (s >= min_level) & (s < 10), BUY_STRONG, 7, "triangle-up"),
        ("Sell 10", s <= -10, SELL_10, 10, "triangle-down"),
        (f"Sell {min_level}-9", (s <= -min_level) & (s > -10), SELL_STRONG, 7, "triangle-down"),
    ]
    for name, m, color, size, sym in groups:
        if m.any():
            fig.add_trace(go.Scatter(
                x=close.index[m], y=close[m], mode="markers", name=name, hoverinfo="skip",
                marker=dict(color=color, size=size, symbol=sym, line=dict(width=1, color="rgba(255,255,255,0.8)"))))
    fig.add_trace(go.Bar(x=s.index, y=s.values, yaxis="y2", name="Strength", showlegend=False,
                         marker=dict(color=strength_colors(s), line_width=0),
                         customdata=[fmt_strength(v) for v in s.values], hovertemplate="%{customdata}"))
    if show_extremes:
        ex = extremes_df()
        ex = ex[(ex.date >= close.index[0]) & (ex.date <= close.index[-1])]
        for _, e in ex.iterrows():
            fig.add_annotation(x=e.date, y=np.log10(e.price), yref="y", text=("cycle top" if e.kind == "top" else "cycle low"),
                               showarrow=True, arrowhead=0, ay=(-28 if e.kind == "top" else 28), font=dict(size=10),
                               arrowcolor=TOP_MARK if e.kind == "top" else LOW_MARK)
    for name, a, b in shade or []:
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(128,128,128,0.07)", line_width=0,
                      annotation_text=name, annotation_position="top left", annotation_font_size=10)
    stacked_layout(fig, "USD (log)", "strength", height, title)
    fig.update_layout(bargap=0, yaxis2=dict(range=[-10.5, 10.5], tickvals=[-10, -5, 0, 5, 10],
                                            ticktext=["Sell 10", "Sell 5", "0", "Buy 5", "Buy 10"]))
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
