"""Notifications: ntfy (phone push) + macOS notification centre.

ntfy topics on the public server are readable by anyone who knows the name, so the topic is
a long random string kept in .env (NTFY_TOPIC). Messages contain only market signals.
"""

from __future__ import annotations

import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

from . import db
from .config import ROOT, env, load_config


@dataclass
class Alert:
    kind: str
    title: str
    message: str
    priority: int = 3            # ntfy 1..5
    tags: str = ""
    chart: Path | None = None


def ensure_topic() -> str:
    """Create a random ntfy topic in .env if none exists; return it."""
    topic = env("NTFY_TOPIC")
    if topic:
        return topic
    topic = "btcn-" + secrets.token_urlsafe(18).replace("_", "x").replace("-", "y")
    envf = ROOT / ".env"
    existing = envf.read_text() if envf.exists() else ""
    envf.write_text(existing + ("" if existing.endswith("\n") or not existing else "\n") + f"NTFY_TOPIC={topic}\n")
    import os
    os.environ["NTFY_TOPIC"] = topic
    return topic


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode()


def send_ntfy(alert: Alert) -> None:
    topic = env("NTFY_TOPIC")
    if not topic:
        raise RuntimeError("NTFY_TOPIC not set; run `btcn setup-ntfy`")
    server = env("NTFY_SERVER") or load_config()["notify"]["ntfy_server"]
    url = f"{server.rstrip('/')}/{topic}"
    headers = {"Title": _ascii(alert.title), "Priority": str(alert.priority)}
    if alert.tags:
        headers["Tags"] = alert.tags
    token = env("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, data=alert.message.encode("utf-8"), headers=headers, timeout=30)
    r.raise_for_status()
    if alert.chart and alert.chart.exists():
        h2 = {"Filename": alert.chart.name, "Title": _ascii(alert.title + " - chart"), "Priority": "1"}
        if token:
            h2["Authorization"] = f"Bearer {token}"
        requests.put(url, data=alert.chart.read_bytes(), headers=h2, timeout=60).raise_for_status()


def send_macos(alert: Alert) -> None:
    msg = alert.message.split("\n")[0].replace('"', "'")
    title = alert.title.replace('"', "'")
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}"'],
                   check=False, capture_output=True, timeout=15)


def already_sent(date: str, kind: str) -> bool:
    df = db.query("SELECT 1 FROM notifications WHERE date=? AND kind=? AND channel='ntfy' AND status='ok' LIMIT 1", (date, kind))
    return not df.empty


def dispatch(date: str, alerts: list[Alert], force: bool = False) -> list[str]:
    sent = []
    cfg = load_config()["notify"]
    for a in alerts:
        if not force and already_sent(date, a.kind):
            continue
        try:
            send_ntfy(a)
            db.log_notification(date, "ntfy", a.kind, a.title, a.message, "ok")
            sent.append(a.kind)
        except Exception as e:  # noqa: BLE001 - notification failures are logged, never fatal
            db.log_notification(date, "ntfy", a.kind, a.title, a.message, "error", str(e))
        if cfg.get("macos", True):
            send_macos(a)
            db.log_notification(date, "macos", a.kind, a.title, a.message, "ok")
    return sent
