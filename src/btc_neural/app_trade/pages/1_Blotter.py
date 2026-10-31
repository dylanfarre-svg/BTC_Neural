"""Every trade of the promoted configuration, with the equity curve against buy-and-hold."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_neural import db
from btc_neural.app.common import GRID
from btc_neural.app_trade.helpers import BENCH, STRATEGY, add_trades, chosen_backtest, pct, price_figure, report
from btc_neural.trade.horizons import horizon
from btc_neural.trade.simulator import after_tax_equity

st.set_page_config(page_title="Blotter · BTC_Neural swing", page_icon="↔", layout="wide")
st.title("Trade blotter")

bt, rep = chosen_backtest(), report()
if bt is None or rep is None:
    st.info("No research report yet. Run `uv run btcn trade research`.")
    st.stop()

labels = {"full": "Full (2015 → holdout)", "modern": "Modern (2019 → holdout)", "holdout": "Holdout (never used to select)"}
w = st.radio("Window", list(labels), format_func=labels.get, horizontal=True, key="sw_window")
res, mw = bt["results"][w], bt["windows"][w]
hz = horizon("swing")

c = st.columns(3) + st.columns(3)  # two rows of three: six tiles in one row truncate their values
c[0].metric("Trades", f"{mw.get('n_trades', 0)}", f"{mw.get('trades_per_year', 0):.1f} / year", delta_color="off")
c[1].metric("Long / short", f"{mw.get('n_long', 0)} / {mw.get('n_short', 0)}")
c[2].metric("Hit rate (net > 0)", pct(mw.get("hit_rate"), 0).lstrip("+"))
c[3].metric("Mean net per trade", pct(mw.get("mean_net"), 2), f"95% lower bound {pct(mw.get('mean_net_lb95'), 2)}",
            delta_color="off")
c[4].metric("Median hold", f"{mw.get('median_hold', 0):.1f} days", f"{mw.get('hold_in_band', 0):.0%} inside 7–28",
            delta_color="off")
c[5].metric("Time in market", f"{mw.get('time_in_market', 0):.0%}")

# --- equity: strategy (pre-tax / after each tax treatment) vs buy-and-hold, one log axis ----------
eq = (1 + res.daily).cumprod()
cap, rev = after_tax_equity(res, hz.capital_rate), after_tax_equity(res, hz.revenue_rate)
px = bt["px"].loc[res.start - pd.Timedelta(days=1):res.end].dropna()
bh = px / px.iloc[0]
fig = go.Figure()
for name, s, dash in (("Strategy, pre-tax", eq, "solid"), ("Strategy, capital-gains tax", cap, "dash"),
                      ("Strategy, revenue tax", rev, "dot")):
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines", line=dict(color=STRATEGY, width=2, dash=dash),
                             hovertemplate="%{y:.2f}×<extra>" + name + "</extra>"))
fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy and hold (untaxed path)", mode="lines",
                         line=dict(color=BENCH, width=2), hovertemplate="%{y:.2f}×<extra>buy and hold</extra>"))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=40), hovermode="x unified",
                  title="Growth of 1 (log scale)", legend=dict(orientation="h", y=-0.15),
                  yaxis=dict(type="log", title="multiple of starting value", gridcolor=GRID), xaxis=dict(gridcolor=GRID))
st.plotly_chart(fig)

tbl = pd.DataFrame({
    "": ["Total (after tax)", "Sharpe", "Return per year in market (after tax)", "Max drawdown"],
    "Strategy, capital": [f"{mw['total_capital']:.2f}×", f"{mw['sharpe']:.2f}", pct(mw["rptim_capital"]), pct(mw["max_dd"], 0)],
    "Strategy, revenue": [f"{mw['total_revenue']:.2f}×", f"{mw['sharpe']:.2f}", pct(mw["rptim_revenue"]), pct(mw["max_dd"], 0)],
    "Buy and hold": [f"{mw['bh_total_after_tax']:.2f}×", f"{mw['bh_sharpe']:.2f}", pct(mw["bh_rptim"]), pct(mw["bh_max_dd"], 0)],
})
st.dataframe(tbl, hide_index=True)
st.caption("Buy-and-hold is charged capital-gains tax on liquidation at the end of the window — the same "
           "liquidation basis as the strategy. Sharpe is pre-tax for both.")
rs = rep["random_entry"].get(w, {})
if rs:
    st.caption(f"Random entries with the same trade count, long/short mix and exits: median outcome "
               f"{rs['rand_total_median']:.2f}×, 95th percentile {rs['rand_total_p95']:.2f}×. "
               f"Share of random runs that did at least as well as the strategy: {rs['p_total']:.1%}.")

# --- trades -----------------------------------------------------------------------------------------
t = res.trades.copy()
tfig = price_figure(px, height=380, title="Entries and exits")
add_trades(tfig, t, px)
st.plotly_chart(tfig)
show = t[["side", "entry_date", "entry_price", "exit_time", "exit_price", "exit_reason", "target", "hold_days",
          "gross_ret", "funding", "net_ret", "mae", "mfe"]].copy()
show["entry_date"] = pd.to_datetime(show.entry_date).dt.strftime("%Y-%m-%d")
show["exit_time"] = pd.to_datetime(show.exit_time).dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(show.style.format({"entry_price": "${:,.0f}", "exit_price": "${:,.0f}", "target": "{:.1%}",
                                "hold_days": "{:.1f}", "gross_ret": "{:+.2%}", "funding": "{:+.2%}",
                                "net_ret": "{:+.2%}", "mae": "{:+.1%}", "mfe": "{:+.1%}"}),
             hide_index=True, height=420)

live = db.query("SELECT * FROM trade_positions WHERE horizon='swing' ORDER BY opened_at DESC")
st.subheader("Paper trades")
if live.empty:
    st.write("None yet.")
else:
    st.dataframe(live.drop(columns=["horizon"]), hide_index=True)
