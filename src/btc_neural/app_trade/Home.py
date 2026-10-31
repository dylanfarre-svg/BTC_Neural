"""BTC_Neural swing dashboard: research verdict, today's paper-trading state, recent trades."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from btc_neural import db
from btc_neural.app.common import fmt_strength
from btc_neural.app_trade.helpers import (LONG, SHORT, add_trades, chosen_backtest, pct, price_figure, report)
from btc_neural.config import load_config

st.set_page_config(page_title="BTC_Neural swing", page_icon="↔", layout="wide")
st.title("BTC_Neural: swing signal (research)")
fees = load_config()["trade"]["swing"]["costs"]
st.caption(f"Discrete 7–28 day trades, long and short. Paper trading only — never sent to ntfy. "
           f"Costs assumed at {fees['fee_rate']:.2%} fee + {fees['slippage']:.2%} slippage per side until the real "
           f"schedule is supplied. Research tool, not financial advice.")

def _seeds(n: int) -> str:
    return "deterministic model, one fit" if n == 1 else f"{n}-seed ensemble"


rep = report()
if rep is None:
    st.info("No research report yet. Run `uv run btcn trade research`.")
    st.stop()

c = rep["checks"]
chosen = rep["chosen"]
ship = c["ship_capital"] or c["ship_revenue"]
verdict = ("Meets every §9 criterion" if ship else "Does not meet the §9 ship criteria — don't deploy")
(st.success if ship else st.warning)(
    f"**Research verdict: {verdict}.** Promoted by the pre-registered rule: "
    f"`{chosen['model_type']}` on `{chosen['feature_set']}`, trading above the {chosen['quantile']:.0%} quantile "
    f"of its own trailing scores ({_seeds(rep['seeds_in_ensemble'])}). Details on the Research page.")

# --- today ---------------------------------------------------------------------------------------
sig = db.query("SELECT * FROM trade_signals WHERE horizon='swing' ORDER BY ts DESC LIMIT 30")
pos = db.query("SELECT * FROM trade_positions WHERE horizon='swing' ORDER BY opened_at DESC")
cyc = db.query("SELECT date, strength FROM live_signals WHERE role='production' ORDER BY date DESC LIMIT 1")

left, mid, right = st.columns([1.2, 1, 1])
with left:
    st.subheader("Paper position")
    if sig.empty:
        st.write("No paper-trading days logged yet — the daily job adds one row per day.")
    else:
        last = sig.iloc[0]
        state = last.state
        color = LONG if state == "long" else SHORT if state == "short" else "inherit"
        st.markdown(f"<div style='font-size:2.2rem;font-weight:700;color:{color}'>{state.replace('_', ' ').title()}</div>"
                    f"<div style='opacity:0.7'>as of the {last.ts} close · action today: {last.action}</div>",
                    unsafe_allow_html=True)
        open_ = pos[pos.state.isin(["long", "short"])]
        if len(open_):
            p = open_.iloc[0]
            st.markdown(f"**{p.side.title()}** from {p.opened_at} at **${p.entry_price:,.0f}**  \n"
                        f"Target ${p.target_price:,.0f} · Stop ${p.stop_price:,.0f} · Deadline {p.deadline[:10]}")
with mid:
    st.subheader("Today's scores")
    if not sig.empty:
        last = sig.iloc[0]
        for side in ("long", "short"):
            s, t = last[f"score_{side}"], last[f"thr_{side}"]
            fires = bool(np.isfinite(t) and s > t)
            st.metric(f"{side.title()} score vs threshold", f"{s:+.3f}",
                      f"{s - t:+.3f} {'above — fires' if fires else 'below'}" if np.isfinite(t) else "no threshold yet",
                      delta_color="normal" if fires else "off")
with right:
    st.subheader("Context")
    if len(cyc):
        st.metric("Cycle model (context only)", fmt_strength(int(cyc.strength.iloc[0])), f"as of {cyc.date.iloc[0]}",
                  delta_color="off")
    from btc_neural.trade.live import paper_gate
    g = paper_gate()
    if g:
        st.metric("Paper-trading gate", f"{g['closed_trades']} / {g['min_trades']} trades",
                  "evaluable" if g["evaluable"] else "not yet evaluable", delta_color="off")
        st.caption(f"Live mean net per trade must reach the backtest's frozen lower bound "
                   f"({pct(g.get('backtest_mean_net_lb95'), 2)}), checked quarterly; stop after "
                   f"{g['stop_after_trades']} trades if it hasn't.")

# --- chart ---------------------------------------------------------------------------------------
bt = chosen_backtest()
if bt is not None:
    since = bt["px"].index[-1] - pd.Timedelta(days=183)
    px = bt["px"].loc[since:].dropna()
    trades = pd.concat([r.trades for r in bt["results"].values()]).drop_duplicates(["entry_date", "side"])
    trades = trades[pd.to_datetime(trades.entry_date) >= since]
    fig = price_figure(px, title="Last 6 months: backtested entries (filled) and exits (hollow)")
    add_trades(fig, trades, px)
    if len(pos):
        live = pos.rename(columns={"opened_at": "entry_date", "closed_at": "exit_time", "exit_reason": "exit_reason"})
        add_trades(fig, live.assign(net_ret=live.net_ret), px, label_prefix="paper ")
    st.plotly_chart(fig)

if not sig.empty and sig.iloc[0].drivers_json:
    d = pd.Series(json.loads(sig.iloc[0].drivers_json))
    if len(d):
        with st.expander("What drives today's score (largest feature contributions)"):
            st.dataframe(d.rename("contribution").to_frame().style.format("{:+.4f}"))

with st.expander("Paper-trading log (append-only)"):
    st.dataframe(sig[["ts", "action", "state", "score_long", "thr_long", "score_short", "thr_short", "price", "run_id"]],
                 hide_index=True)
