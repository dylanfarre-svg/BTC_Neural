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


def trained_on(created_at: str | None) -> str:
    """Date a model or run was trained, as YYYY-MM-DD (empty when unknown)."""
    if not created_at:
        return ""
    try:
        return pd.Timestamp(created_at).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(created_at)[:10]


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
        margin=dict(l=10, r=10, t=50 if title else 30, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
        hovermode="x unified", hoversubplots="axis",
        xaxis=dict(anchor="y2", gridcolor=GRID, hoverformat="%Y-%m-%d", showspikes=True, spikemode="across", spikesnap="cursor",
                   spikethickness=1, spikedash="solid", spikecolor="rgba(128,128,128,0.7)"),
        yaxis=dict(domain=[split + 0.07, 1.0], type="log" if top_log else "linear", title=top_title, gridcolor=GRID),
        yaxis2=dict(domain=[0.0, split], title=bottom_title, gridcolor=GRID, zerolinecolor="rgba(128,128,128,0.5)"),
    )
    return fig


HOVER_SPINE = "__hover_spine__"


def add_hover_spine(fig: go.Figure, x0: pd.Timestamp, x1: pd.Timestamp, y: float, pad_frac: float = 0.06) -> None:
    """Invisible daily points spanning x0..x1 so the date tooltip and crosshair keep working where
    no series has data: the forecast horizon beyond today, and the blank margins Plotly leaves at
    each end of an auto-ranged axis. Updates the existing spine if one was already added."""
    pad = pd.Timedelta(days=max(14, (x1 - x0).days * pad_frac))
    xs = pd.date_range(x0 - pad, x1 + pad, freq="D")
    ys = [float(y)] * len(xs)
    # Pin the visible range inside the spine, so Plotly adds no unhoverable margin of its own.
    fig.update_xaxes(range=[x0 - pad / 2, x1 + pad / 2], hoverformat="%Y-%m-%d")
    for tr in fig.data:
        if tr.name == HOVER_SPINE:
            tr.x, tr.y = xs, ys
            return
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=HOVER_SPINE, showlegend=False,
                             marker=dict(size=1, opacity=0), hovertemplate="<extra></extra>"))


def add_halving_lines(fig: go.Figure, x0: pd.Timestamp | None = None, x1: pd.Timestamp | None = None,
                     row_ref: str = "paper") -> None:
    """Vertical lines at each halving (and the estimated next one). Halvings are scheduled, so
    they are legitimate context, not hindsight."""
    from btc_neural.config import halvings, load_config
    past = [(h, False) for h in halvings()]
    nxt = pd.Timestamp(load_config()["next_halving_estimate"])
    for h, is_est in past + [(nxt, True)]:
        if (x0 is not None and h < x0) or (x1 is not None and h > x1):
            continue
        fig.add_vline(x=h, line=dict(color="rgba(140,140,140,0.75)", width=1, dash="dot" if is_est else "dashdot"),
                      annotation_text=("next halving (est)" if is_est else f"halving {h:%Y}"),
                      annotation_position="top right", annotation_font_size=9,
                      annotation_font_color="rgba(150,150,150,0.95)")


def signal_figure(close: pd.Series, strength: pd.Series, title: str = "", show_extremes: bool = True,
                  min_level: int = 7, shade: list[tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
                  height: int = 680, show_halvings: bool = False) -> go.Figure:
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
    add_hover_spine(fig, close.index[0], close.index[-1], float(np.sqrt(close.min() * close.max())))
    if show_halvings:
        add_halving_lines(fig, close.index[0])
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


def add_forecast_overlay(fig: go.Figure, fc, today: pd.Timestamp, x_start: pd.Timestamp | None = None,
                         bands: int = 41, max_alpha: float = 0.30) -> pd.Timestamp:
    """Gaussian-shaded bands for the forecast date (vertical) and price (horizontal).

    Each band's opacity follows the normal density, so the shading fades out at ±3σ.
    Returns the right edge of the forecast horizon (for the x-axis range).
    """
    rgb = "42,120,214" if fc.kind == "low" else "227,73,72"
    color = BUY_10 if fc.kind == "low" else SELL_10
    x_end = max(fc.date_mu + pd.Timedelta(days=3.2 * fc.date_sd), today + pd.Timedelta(days=30))
    hx0 = x_start if x_start is not None else today   # horizontal (price) band spans the whole chart
    zs = np.linspace(-3, 3, bands)
    dz = zs[1] - zs[0]
    for z in zs:
        a = max_alpha * float(np.exp(-z * z / 2))
        fill = f"rgba({rgb},{a:.3f})"
        fig.add_shape(type="rect", xref="x", yref="y domain", layer="below", line_width=0, fillcolor=fill,
                      x0=fc.date_mu + pd.Timedelta(days=(z - dz / 2) * fc.date_sd),
                      x1=fc.date_mu + pd.Timedelta(days=(z + dz / 2) * fc.date_sd), y0=0, y1=1)
        fig.add_shape(type="rect", xref="x", yref="y", layer="below", line_width=0,
                      fillcolor=f"rgba({rgb},{0.75 * a:.3f})", x0=hx0, x1=x_end,
                      y0=float(np.exp(fc.logp_mu + (z - dz / 2) * fc.logp_sd)),
                      y1=float(np.exp(fc.logp_mu + (z + dz / 2) * fc.logp_sd)))
    fig.add_shape(type="line", xref="x", yref="y domain", x0=fc.date_mu, x1=fc.date_mu, y0=0, y1=1,
                  line=dict(color=color, width=1.5, dash="dash"))
    # shapes on a log axis take data values (unlike annotations, which take log10)
    fig.add_shape(type="line", xref="x", yref="y", x0=hx0, x1=x_end, y0=fc.price_mu, y1=fc.price_mu,
                  line=dict(color=color, width=1.5, dash="dash"))
    fig.add_shape(type="line", xref="x", yref="paper", x0=today, x1=today, y0=0, y1=1,
                  line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot"))
    fig.add_annotation(x=today, y=0, yref="y domain", text="today", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=10), xshift=-3)
    d1, d2 = fc.date_interval(1)
    p1, p2 = fc.price_interval(1)
    p95a, p95b = fc.price_interval(1.96)
    label = f"next cycle {fc.kind}"
    fig.add_trace(go.Scatter(
        x=[fc.date_mu], y=[fc.price_mu], mode="markers+text", name=f"Forecast: {label}",
        marker=dict(symbol="diamond", size=12, color=color, line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
        text=[f"{label}<br>≈ {fc.date_mu:%b %Y}<br>${fc.price_mu / 1000:,.0f}k"], textposition="middle right",
        textfont=dict(size=11),
        hovertemplate=(f"Forecast {label}<br>date {fc.date_mu:%Y-%m-%d} (68%: {d1:%b %d} – {d2:%b %d, %Y})"
                       f"<br>price ${fc.price_mu:,.0f} (68%: ${p1:,.0f}–${p2:,.0f}; 95%: ${p95a:,.0f}–${p95b:,.0f})"
                       f"<br>P(already happened) {fc.p_already:.0%}<extra></extra>")))
    return x_end


# Categorical hues for forecast sources (fixed order, never cycled).
SOURCE_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#2a78d6"]


def add_forecast_marker(fig: go.Figure, fc, label: str, color: str) -> pd.Timestamp:
    """One forecast as a marker with 1σ error bars in both date and price."""
    p1, p2 = fc.price_interval(1)
    p95a, p95b = fc.price_interval(1.96)
    d1, d2 = fc.date_interval(1)
    fig.add_trace(go.Scatter(
        x=[fc.date_mu], y=[fc.price_mu], mode="markers", name=label,
        marker=dict(symbol="diamond", size=11, color=color, line=dict(width=1.2, color="rgba(255,255,255,0.9)")),
        error_x=dict(type="data", array=[fc.date_sd * 86_400_000], color=color, thickness=1.4, width=5),
        error_y=dict(type="data", symmetric=False, array=[p2 - fc.price_mu], arrayminus=[fc.price_mu - p1],
                     color=color, thickness=1.4, width=5),
        hovertemplate=(f"{label}<br>next cycle {fc.kind}<br>date {fc.date_mu:%Y-%m-%d} "
                       f"(68%: {d1:%b %d} – {d2:%b %d, %Y})<br>price ${fc.price_mu:,.0f} "
                       f"(68%: ${p1:,.0f}–${p2:,.0f}; 95%: ${p95a:,.0f}–${p95b:,.0f})<extra></extra>")))
    return fc.date_mu + pd.Timedelta(days=1.5 * fc.date_sd)
