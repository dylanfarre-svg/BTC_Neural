"""Data freshness, ingestion log, live signal log and notifications."""

import pandas as pd
import streamlit as st

from btc_neural import db
from btc_neural.app.common import clear_caches, fmt_strength
from btc_neural.config import env

st.set_page_config(page_title="Data & alerts · BTC_Neural", page_icon="₿", layout="wide")
st.title("Data and alerts")

c1, c2 = st.columns(2)
if c1.button("Refresh data now"):
    from btc_neural.ingest import ingest_all
    with st.spinner("Downloading…"):
        res = ingest_all(verbose=False)
    st.write(res)
    clear_caches()
if c2.button("Run the daily job now (ingest, predict, notify)"):
    from btc_neural.live import run_daily
    with st.spinner("Running daily job…"):
        out = run_daily(notify=True, verbose=False)
    st.write({k: v for k, v in out.items() if k != "signals"})
    clear_caches()

st.subheader("Data freshness")
fresh = db.query("SELECT name, source, MAX(date) AS last_date, COUNT(*) AS rows, MIN(date) AS first_date FROM series GROUP BY name, source")
cand = db.query("SELECT timeframe AS name, 'candles' AS source, MAX(ts) AS last_date, COUNT(*) AS rows, MIN(ts) AS first_date FROM candles GROUP BY timeframe")
f = pd.concat([cand, fresh], ignore_index=True)
f["days_old"] = (pd.Timestamp.now().normalize() - pd.to_datetime(f["last_date"], format="mixed").dt.normalize()).dt.days
st.dataframe(f, use_container_width=True, hide_index=True)

st.subheader("Live signal log (append-only)")
ls = db.query("SELECT date, role, strength, score, confidence, price, run_id, created_at FROM live_signals ORDER BY date DESC, role LIMIT 500")
if len(ls):
    ls["signal"] = ls["strength"].map(fmt_strength)
    st.dataframe(ls[["date", "role", "signal", "score", "confidence", "price", "run_id", "created_at"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("No live signals logged yet. The daily job appends one row per day per role.")

st.subheader("Notifications")
topic = env("NTFY_TOPIC")
if topic:
    st.write(f"ntfy topic: `{topic}` — subscribe in the ntfy app (server ntfy.sh). Keep it private.")
    if st.button("Send test notification"):
        from btc_neural.notify import Alert, dispatch
        sent = dispatch(pd.Timestamp.now().strftime("%Y-%m-%d"),
                        [Alert("test", "BTC_Neural test", "Test from the dashboard.", 3, "white_check_mark")], force=True)
        st.write("sent:", sent)
else:
    st.warning("No ntfy topic configured. Run `uv run btcn setup-ntfy`.")
nt = db.query("SELECT created_at, date, channel, kind, title, status, error FROM notifications ORDER BY id DESC LIMIT 200")
st.dataframe(nt, use_container_width=True, hide_index=True)

st.subheader("Ingestion log")
st.dataframe(db.query("SELECT source, started_at, finished_at, rows, status, substr(error,1,200) AS error FROM ingest_log ORDER BY id DESC LIMIT 50"),
             use_container_width=True, hide_index=True)
