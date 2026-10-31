"""Swing-app helpers. Shared chart idioms live in app/common.py and are imported from there; only
swing-specific pieces (trade markers, the research report) belong here."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_neural.app.common import GRID, PRICE, add_hover_spine

# Long/short identity. The cycle app owns the blue/red buy/sell ramp, so the swing app uses the
# palette's aqua and orange (validated as a pair, light and dark). Aqua sits below 3:1 on the light
# surface, so markers always carry shape too (triangle up/down) and every chart has a table view.
LONG, SHORT = "#1baf7a", "#eb6834"
STRATEGY, BENCH = "#6250d6", PRICE
SIDE_STYLE = {"long": (LONG, "triangle-up"), "short": (SHORT, "triangle-down")}


@st.cache_data(ttl=600)
def report() -> dict | None:
    from btc_neural.trade.research import REPORT
    return json.loads(REPORT.read_text()) if REPORT.exists() else None


@st.cache_resource(ttl=3600, show_spinner="Rebuilding the swing backtest…")
def chosen_backtest():
    """The promoted configuration's seed ensemble, re-simulated per window (full/modern/holdout)."""
    from btc_neural.trade.backtest import evaluate, research_data
    from btc_neural.trade.live import _config_runs, chosen_config
    from btc_neural.trade.research import ensemble_scores
    cfg = chosen_config()
    if cfg is None:
        return None
    runs = _config_runs(cfg)
    r = evaluate(runs[0], cfg["quantile"], scores=ensemble_scores(runs), store=False, keep=True)
    hz, daily, feats, out, m = research_data()
    return {"cfg": cfg, "results": r["_results"], "windows": r["windows"], "px": pd.Series(m.px, index=m.days)}


def add_trades(fig: go.Figure, trades: pd.DataFrame, px: pd.Series, label_prefix: str = "") -> None:
    """Entry markers (filled triangles) and exit markers (hollow) on the price line."""
    for side, (color, sym) in SIDE_STYLE.items():
        t = trades[trades.side == side]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(t.entry_date), y=t.entry_price, mode="markers", name=f"{label_prefix}{side} entry",
            marker=dict(color=color, size=10, symbol=sym, line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([t.exit_reason.fillna("open"), (t.net_ret.fillna(0) * 100)], axis=1),
            hovertemplate=f"{side} entry $%{{y:,.0f}}<br>exit: %{{customdata[0]}}, net %{{customdata[1]:+.1f}}%<extra></extra>"))
        done = t[t.exit_price.notna()]
        if len(done):
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(done.exit_time), y=done.exit_price, mode="markers", name=f"{label_prefix}{side} exit",
                marker=dict(color="rgba(0,0,0,0)", size=9, symbol="circle", line=dict(width=2, color=color)),
                customdata=done.exit_reason, hovertemplate=f"{side} exit $%{{y:,.0f}} (%{{customdata}})<extra></extra>"))


def price_figure(px: pd.Series, height: int = 420, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=px.index, y=px.values, name="BTC/USD", mode="lines",
                               line=dict(color=PRICE, width=2), hovertemplate="$%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=height, title=title, margin=dict(l=10, r=10, t=40 if title else 10, b=40),
                      hovermode="closest", legend=dict(orientation="h", y=-0.12),
                      yaxis=dict(type="log", title="USD (log)", gridcolor=GRID), xaxis=dict(gridcolor=GRID))
    add_hover_spine(fig, px.index[0], px.index[-1], float(np.sqrt(px.min() * px.max())))
    return fig


def pct(x, digits: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{digits}f}%"


def tick(ok: bool) -> str:
    return "✅ pass" if ok else "❌ fail"
