"""Derivatives data for the trading signals: perp funding, open interest, implied volatility.

Free public endpoints, stored in the shared `series` table (cycle features read named series
only, so these change nothing there). Deliberately NOT part of `ingest.ingest_all`: a failure
here must never become a cycle-side health alert, which would reach ntfy.

Series (daily, UTC date):
- FUND_BINANCE  Binance BTCUSDT perp, sum of the day's 8h funding rates (2019-09-10+)
- FUND_BYBIT    Bybit BTCUSD inverse perp, same (2018-11+); fills the gap before Binance
- OI_BYBIT      Bybit BTCUSDT open interest in BTC, daily snapshot (2020-08+)
- DVOL          Deribit BTC implied-volatility index, daily close, % (2021-03-24+)
"""

from __future__ import annotations

import time

import pandas as pd

from .. import db
from ..ingest import _get, _logged

META = {
    "FUND_BINANCE": ("Binance BTCUSDT perp funding, daily sum of 8h rates", "binance", "2019-09-10"),
    "FUND_BYBIT": ("Bybit BTCUSD inverse perp funding, daily sum of 8h rates", "bybit", "2018-11-01"),
    "OI_BYBIT": ("Bybit BTCUSDT open interest, BTC, daily snapshot", "bybit", "2020-08-01"),
    "DVOL": ("Deribit BTC implied volatility index (DVOL), daily close", "deribit", "2021-03-24"),
}


def _ms(t: pd.Timestamp) -> int:
    return int(pd.Timestamp(t).timestamp() * 1000)


def _start(name: str) -> pd.Timestamp:
    last = db.query("SELECT MAX(date) AS d FROM series WHERE name=?", (name,))["d"].iloc[0]
    return pd.Timestamp(last) - pd.Timedelta(days=5) if last else pd.Timestamp(META[name][2])


def _store(name: str, daily: pd.Series) -> int:
    daily = daily.dropna()
    if daily.empty:
        return 0
    df = pd.DataFrame({"name": name, "date": daily.index.strftime("%Y-%m-%d"), "value": daily.to_numpy(dtype=float),
                       "source": META[name][1], "fetched_at": db.now_utc()})
    meta = pd.DataFrame([{"name": name, "source": META[name][1], "tier": "D", "publication_lag_days": 0,
                          "description": META[name][0]}])
    with db.connect() as con:
        db.upsert(con, "series_meta", meta)
        return db.upsert(con, "series", df)


def _daily_funding(rows: list[tuple[int, float]]) -> pd.Series:
    s = pd.Series({pd.to_datetime(t, unit="ms"): r for t, r in rows}).sort_index()
    s = s[~s.index.duplicated()]
    return s.groupby(s.index.normalize()).sum()


@_logged("binance_funding")
def ingest_binance_funding() -> int:
    start, now, rows = _ms(_start("FUND_BINANCE")), _ms(pd.Timestamp.now(tz="UTC").tz_localize(None)), []
    while start < now:
        batch = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                     {"symbol": "BTCUSDT", "startTime": start, "limit": 1000}).json()
        if not batch:
            break
        rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in batch]
        start = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    # drop today's partial day: it would understate the daily sum
    s = _daily_funding(rows)
    return _store("FUND_BINANCE", s[s.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()])


@_logged("bybit_funding")
def ingest_bybit_funding() -> int:
    start = _start("FUND_BYBIT")
    end, rows = _ms(pd.Timestamp.now(tz="UTC").tz_localize(None)), []
    while True:  # pages backwards from `end`
        j = _get("https://api.bybit.com/v5/market/funding/history",
                 {"category": "inverse", "symbol": "BTCUSD", "limit": 200, "startTime": _ms(start), "endTime": end}).json()
        batch = j.get("result", {}).get("list", [])
        if not batch:
            break
        rows += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in batch]
        oldest = min(int(x["fundingRateTimestamp"]) for x in batch)
        if len(batch) < 200 or oldest <= _ms(start):
            break
        end = oldest - 1
        time.sleep(0.2)
    s = _daily_funding(rows)
    return _store("FUND_BYBIT", s[s.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()])


@_logged("bybit_oi")
def ingest_bybit_oi() -> int:
    t, now, out = _start("OI_BYBIT"), pd.Timestamp.now(tz="UTC").tz_localize(None), {}
    while t < now:
        t2 = min(t + pd.Timedelta(days=199), now)
        j = _get("https://api.bybit.com/v5/market/open-interest",
                 {"category": "linear", "symbol": "BTCUSDT", "intervalTime": "1d", "limit": 200,
                  "startTime": _ms(t), "endTime": _ms(t2)}).json()
        for x in j.get("result", {}).get("list", []):
            out[pd.to_datetime(int(x["timestamp"]), unit="ms").normalize()] = float(x["openInterest"])
        t = t2 + pd.Timedelta(days=1)
        time.sleep(0.2)
    return _store("OI_BYBIT", pd.Series(out).sort_index())


@_logged("deribit_dvol")
def ingest_dvol() -> int:
    t, now, out = _start("DVOL"), pd.Timestamp.now(tz="UTC").tz_localize(None), {}
    while t < now:
        t2 = min(t + pd.Timedelta(days=500), now)
        j = _get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                 {"currency": "BTC", "start_timestamp": _ms(t), "end_timestamp": _ms(t2), "resolution": "1D"}).json()
        for ts_, _o, _h, _l, c in j.get("result", {}).get("data", []):
            out[pd.to_datetime(int(ts_), unit="ms").normalize()] = float(c)
        t = t2 + pd.Timedelta(days=1)
        time.sleep(0.2)
    s = pd.Series(out).sort_index()
    return _store("DVOL", s[s.index < now.normalize()])


def ingest_derivatives(verbose: bool = True) -> dict[str, str]:
    results = {}
    for name, fn in (("binance_funding", ingest_binance_funding), ("bybit_funding", ingest_bybit_funding),
                     ("bybit_oi", ingest_bybit_oi), ("deribit_dvol", ingest_dvol)):
        try:
            results[name] = f"ok ({fn()} rows)"
        except Exception as e:  # noqa: BLE001 - one failing source must not block the others
            results[name] = f"error: {e}"
        if verbose:
            print(f"  {name}: {results[name]}")
    return results
