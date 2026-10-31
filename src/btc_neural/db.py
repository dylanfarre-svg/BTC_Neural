"""SQLite storage. One file holds all tabular data; model weights live under runs/."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from .config import path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    timeframe TEXT NOT NULL,          -- 4h | 1d | 1w
    ts TEXT NOT NULL,                 -- bar open time, UTC ISO
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    source TEXT,
    PRIMARY KEY (timeframe, ts)
);
CREATE TABLE IF NOT EXISTS series (
    name TEXT NOT NULL,
    date TEXT NOT NULL,               -- observation date (UTC day)
    value REAL,
    source TEXT,
    fetched_at TEXT,
    PRIMARY KEY (name, date)
);
CREATE TABLE IF NOT EXISTS series_meta (
    name TEXT PRIMARY KEY,
    source TEXT,
    tier TEXT,
    publication_lag_days INTEGER,
    description TEXT
);
CREATE TABLE IF NOT EXISTS ingest_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT, started_at TEXT, finished_at TEXT,
    rows INTEGER, status TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT,
    model_type TEXT,
    feature_set TEXT,
    role TEXT,                        -- experiment | production | reference_nn
    config_json TEXT,
    selection_score REAL,
    weights_path TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS fold_results (
    run_id TEXT, fold TEXT, metric TEXT, value REAL,
    PRIMARY KEY (run_id, fold, metric)
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT, fold TEXT, date TEXT, score REAL, strength INTEGER,
    PRIMARY KEY (run_id, date)
);
CREATE TABLE IF NOT EXISTS live_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, run_id TEXT, role TEXT,
    score REAL, strength INTEGER, confidence REAL, price REAL,
    drivers_json TEXT,
    created_at TEXT,
    UNIQUE (date, role)
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT, date TEXT, channel TEXT, kind TEXT,
    title TEXT, message TEXT, status TEXT, error TEXT
);
-- Append-only guard: live signals are never rewritten or deleted.
CREATE TRIGGER IF NOT EXISTS live_signals_no_update BEFORE UPDATE ON live_signals
BEGIN SELECT RAISE(ABORT, 'live_signals is append-only'); END;
CREATE TRIGGER IF NOT EXISTS live_signals_no_delete BEFORE DELETE ON live_signals
BEGIN SELECT RAISE(ABORT, 'live_signals is append-only'); END;

-- Trading signals (btc_neural.trade). New tables only: the cycle tables above are never altered,
-- because CREATE TABLE IF NOT EXISTS cannot add a column to a table that already exists.
CREATE TABLE IF NOT EXISTS trade_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, horizon TEXT, model_type TEXT,
    feature_set TEXT, role TEXT,      -- experiment | production
    config_json TEXT, weights_path TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS trade_predictions (   -- walk-forward out-of-sample scores
    run_id TEXT, date TEXT, side TEXT, block TEXT, score REAL,
    PRIMARY KEY (run_id, date, side)
);
CREATE TABLE IF NOT EXISTS trade_evals (         -- one per (run, threshold, sides) variant
    eval_id TEXT PRIMARY KEY, run_id TEXT, quantile REAL, sides TEXT,
    created_at TEXT, metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS trade_backtest (      -- one row per simulated trade
    eval_id TEXT, trade_id INTEGER, side TEXT, entry_date TEXT, entry_price REAL,
    exit_time TEXT, exit_price REAL, exit_reason TEXT, target REAL, gross_ret REAL,
    cost REAL, funding REAL, net_ret REAL, hold_days REAL, mae REAL, mfe REAL,
    PRIMARY KEY (eval_id, trade_id)
);
CREATE TABLE IF NOT EXISTS trade_positions (     -- live paper-trading state machine
    trade_id TEXT PRIMARY KEY, horizon TEXT, run_id TEXT, side TEXT, state TEXT,
    opened_at TEXT, entry_price REAL, target_price REAL, stop_price REAL, deadline TEXT,
    closed_at TEXT, exit_price REAL, exit_reason TEXT, net_ret REAL
);
CREATE TABLE IF NOT EXISTS trade_signals (       -- append-only paper log, one row per horizon per bar
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, horizon TEXT, run_id TEXT,
    score_long REAL, score_short REAL, thr_long REAL, thr_short REAL,
    action TEXT, state TEXT, price REAL, drivers_json TEXT, created_at TEXT,
    UNIQUE (ts, horizon)
);
CREATE TRIGGER IF NOT EXISTS trade_signals_no_update BEFORE UPDATE ON trade_signals
BEGIN SELECT RAISE(ABORT, 'trade_signals is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trade_signals_no_delete BEFORE DELETE ON trade_signals
BEGIN SELECT RAISE(ABORT, 'trade_signals is append-only'); END;
"""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    db_path = path("db_path")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = list(df.columns)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in df.itertuples(index=False, name=None)]
    con.executemany(sql, rows)
    return len(rows)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)


def execute(sql: str, params: tuple = ()) -> None:
    with connect() as con:
        con.execute(sql, params)


# --- Typed readers -----------------------------------------------------------

def read_candles(timeframe: str) -> pd.DataFrame:
    df = query("SELECT ts, open, high, low, close, volume FROM candles WHERE timeframe=? ORDER BY ts", (timeframe,))
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def read_series(names: list[str] | None = None) -> pd.DataFrame:
    """Wide frame: index = observation date, columns = series names."""
    if names:
        q = f"SELECT name, date, value FROM series WHERE name IN ({','.join('?' * len(names))})"
        df = query(q, tuple(names))
    else:
        df = query("SELECT name, date, value FROM series")
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="name", values="value").sort_index()


def series_lags() -> dict[str, int]:
    df = query("SELECT name, publication_lag_days FROM series_meta")
    return dict(zip(df["name"], df["publication_lag_days"].fillna(0).astype(int)))


def log_notification(date: str, channel: str, kind: str, title: str, message: str, status: str, error: str | None = None) -> None:
    execute(
        "INSERT INTO notifications (created_at, date, channel, kind, title, message, status, error) VALUES (?,?,?,?,?,?,?,?)",
        (now_utc(), date, channel, kind, title, message, status, error),
    )


def save_run(run_id: str, model_type: str, feature_set: str, role: str, config: dict,
             selection_score: float | None, weights_path: str | None, notes: str = "") -> None:
    with connect() as con:
        upsert(con, "runs", pd.DataFrame([{
            "run_id": run_id, "created_at": now_utc(), "model_type": model_type,
            "feature_set": feature_set, "role": role, "config_json": json.dumps(config, default=str),
            "selection_score": selection_score, "weights_path": weights_path, "notes": notes,
        }]))
