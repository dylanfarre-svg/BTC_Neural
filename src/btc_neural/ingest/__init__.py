"""Data ingestion from free sources into SQLite.

Sources
- Bitstamp: 1d and 4h BTC/USD OHLCV (from 2011-08)
- Coin Metrics Community API: daily price (from 2010-07) and on-chain metrics
- FRED: macro series (M2, Fed balance sheet, dollar index, 10y yield, USD/ZAR)
- alternative.me: Crypto Fear & Greed index (2018+)
"""

from __future__ import annotations

import io
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import requests

from .. import db

UA = {"User-Agent": "curl/8.7.1"}  # FRED drops connections from unknown agents

COINMETRICS_METRICS = {
    # name: (description, tier)
    "PriceUSD": ("Coin Metrics reference price, USD", "A"),
    "CapMrktCurUSD": ("Market capitalisation, USD", "A"),
    "CapMVRVCur": ("MVRV ratio (market cap / realized cap)", "A"),
    "HashRate": ("Mean hash rate", "A"),
    "AdrActCnt": ("Active addresses", "A"),
    "SplyCur": ("Current supply", "A"),
    "TxCnt": ("Transaction count", "A"),
    "IssTotUSD": ("Miner issuance value, USD (for Puell multiple)", "A"),
}

FRED_SERIES = {
    # name: (description, publication lag in days, tier)
    "M2SL": ("US M2 money stock (monthly, dated month start)", 60, "M"),
    "WALCL": ("Fed total assets (weekly)", 2, "M"),
    "DTWEXBGS": ("Broad US dollar index", 7, "M"),
    "DGS10": ("US 10-year Treasury yield", 1, "M"),
    "DEXSFUS": ("South African rand per US dollar", 7, "M"),
}


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def _logged(source: str):
    def deco(fn):
        def wrapper(*a, **kw):
            started = db.now_utc()
            try:
                n = fn(*a, **kw)
                db.execute("INSERT INTO ingest_log (source, started_at, finished_at, rows, status, error) VALUES (?,?,?,?,?,?)",
                           (source, started, db.now_utc(), n, "ok", None))
                return n
            except Exception as e:  # noqa: BLE001 - logged and re-raised
                db.execute("INSERT INTO ingest_log (source, started_at, finished_at, rows, status, error) VALUES (?,?,?,?,?,?)",
                           (source, started, db.now_utc(), 0, "error", f"{e}\n{traceback.format_exc()[-1500:]}"))
                raise
        return wrapper
    return deco


def _last_ts(timeframe: str) -> pd.Timestamp | None:
    df = db.query("SELECT MAX(ts) AS t FROM candles WHERE timeframe=? AND source='bitstamp'", (timeframe,))
    t = df["t"].iloc[0]
    return pd.Timestamp(t) if t else None


# --- Bitstamp -----------------------------------------------------------------

def _bitstamp_ohlc(step: int, start: int) -> pd.DataFrame:
    frames = []
    now = int(datetime.now(timezone.utc).timestamp())
    while True:
        r = _get("https://www.bitstamp.net/api/v2/ohlc/btcusd/", {"step": step, "limit": 1000, "start": start})
        rows = r.json()["data"]["ohlc"]
        if not rows:
            break
        df = pd.DataFrame(rows).astype(float)
        frames.append(df)
        last = int(df["timestamp"].max())
        if last + step >= now or last < start:
            break
        start = last + step
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] + step <= now]  # drop the still-forming bar
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[["ts", "open", "high", "low", "close", "volume"]]


@_logged("bitstamp")
def ingest_bitstamp() -> int:
    n = 0
    for tf, step in (("1d", 86400), ("4h", 14400)):
        last = _last_ts(tf)
        start = int(last.timestamp()) - 3 * step if last is not None else 1313000000  # 2011-08
        df = _bitstamp_ohlc(step, start)
        if df.empty:
            continue
        df.insert(0, "timeframe", tf)
        df["source"] = "bitstamp"
        with db.connect() as con:
            n += db.upsert(con, "candles", df)
    return n


# --- Coin Metrics -------------------------------------------------------------

@_logged("coinmetrics")
def ingest_coinmetrics() -> int:
    last = db.query("SELECT MAX(date) AS d FROM series WHERE source='coinmetrics'")["d"].iloc[0]
    start = (pd.Timestamp(last) - pd.Timedelta(days=10)).strftime("%Y-%m-%d") if last else "2010-01-01"
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {"assets": "btc", "metrics": ",".join(COINMETRICS_METRICS), "frequency": "1d",
              "start_time": start, "page_size": 10000, "paging_from": "start"}
    rows: list[dict] = []
    r = _get(url, params).json()
    rows += r["data"]
    while r.get("next_page_url"):
        r = _get(r["next_page_url"]).json()
        rows += r["data"]
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    long = df.melt(id_vars="date", value_vars=[m for m in COINMETRICS_METRICS if m in df], var_name="name").dropna()
    long["value"] = long["value"].astype(float)
    long["source"] = "coinmetrics"
    long["fetched_at"] = db.now_utc()
    meta = pd.DataFrame([{"name": k, "source": "coinmetrics", "tier": t, "publication_lag_days": 0, "description": d}
                         for k, (d, t) in COINMETRICS_METRICS.items()])
    with db.connect() as con:
        db.upsert(con, "series_meta", meta)
        return db.upsert(con, "series", long[["name", "date", "value", "source", "fetched_at"]])


# --- FRED ---------------------------------------------------------------------

@_logged("fred")
def ingest_fred() -> int:
    n = 0
    for name, (desc, lag, tier) in FRED_SERIES.items():
        r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": name})
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df["name"], df["source"], df["fetched_at"] = name, "fred", db.now_utc()
        with db.connect() as con:
            db.upsert(con, "series_meta", pd.DataFrame([{"name": name, "source": "fred", "tier": tier,
                                                          "publication_lag_days": lag, "description": desc}]))
            n += db.upsert(con, "series", df[["name", "date", "value", "source", "fetched_at"]])
        time.sleep(0.5)
    return n


# --- Fear & Greed -------------------------------------------------------------

@_logged("alternative.me")
def ingest_fear_greed() -> int:
    data = _get("https://api.alternative.me/fng/", {"limit": 0, "format": "json"}).json()["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.strftime("%Y-%m-%d")
    df["value"] = df["value"].astype(float)
    df["name"], df["source"], df["fetched_at"] = "FearGreed", "alternative.me", db.now_utc()
    with db.connect() as con:
        db.upsert(con, "series_meta", pd.DataFrame([{"name": "FearGreed", "source": "alternative.me", "tier": "B",
                                                      "publication_lag_days": 0, "description": "Crypto Fear & Greed index"}]))
        return db.upsert(con, "series", df[["name", "date", "value", "source", "fetched_at"]])


# --- Derived candles ----------------------------------------------------------

def build_derived_candles() -> int:
    """Fill daily candles before Bitstamp history with Coin Metrics closes and build 1w bars."""
    d = db.read_candles("1d")
    cm = db.read_series(["PriceUSD"])
    n = 0
    with db.connect() as con:
        if not cm.empty:
            first_bs = d.index.min() if not d.empty else pd.Timestamp.max
            early = cm.loc[cm.index < first_bs, "PriceUSD"].dropna()
            if len(early):
                prev = early.shift(1).fillna(early)
                df = pd.DataFrame({
                    "timeframe": "1d", "ts": early.index.strftime("%Y-%m-%dT%H:%M:%S"),
                    "open": prev.values, "high": [max(a, b) for a, b in zip(prev, early)],
                    "low": [min(a, b) for a, b in zip(prev, early)], "close": early.values,
                    "volume": None, "source": "coinmetrics"})
                n += db.upsert(con, "candles", df)
        d = pd.read_sql_query("SELECT ts, open, high, low, close, volume FROM candles WHERE timeframe='1d' ORDER BY ts", con)
        d["ts"] = pd.to_datetime(d["ts"])
        d = d.set_index("ts")
        w = d.resample("W-MON", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
        w = w.reset_index()
        w["ts"] = w["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        w.insert(0, "timeframe", "1w")
        w["source"] = "derived"
        con.execute("DELETE FROM candles WHERE timeframe='1w'")
        n += db.upsert(con, "candles", w)
    return n


def ingest_all(verbose: bool = True) -> dict[str, str]:
    results = {}
    for name, fn in (("bitstamp", ingest_bitstamp), ("coinmetrics", ingest_coinmetrics),
                     ("fred", ingest_fred), ("fear_greed", ingest_fear_greed)):
        try:
            results[name] = f"ok ({fn()} rows)"
        except Exception as e:  # noqa: BLE001 - one failing source must not block the others
            results[name] = f"error: {e}"
        if verbose:
            print(f"  {name}: {results[name]}")
    results["derived"] = f"ok ({build_derived_candles()} rows)"
    return results
