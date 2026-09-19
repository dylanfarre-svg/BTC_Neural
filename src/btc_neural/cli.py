"""Command line: `uv run btcn <command>`."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings

import pandas as pd

from .config import ROOT

DEFAULT_GRID_MODELS = ["rules", "linear", "lgbm", "ens_fast", "nn"]
DEFAULT_GRID_SETS = ["core", "cyclenorm", "core_macro", "core_calendar", "core_intraday", "price_clock", "no_clock"]
DEFAULT_MODES = ["trailing", "hybrid", "frozen"]


def cmd_ingest(a):
    from .ingest import ingest_all
    ingest_all()


def cmd_features(a):
    from .features import load_all
    _, f = load_all()
    pd.set_option("display.max_rows", 200)
    print(f.iloc[-1].rename("latest").to_frame().join(f.describe().T[["mean", "std"]]).round(3))


def cmd_backtest(a):
    from .backtest.walkforward import run_walkforward
    models = a.models.split(",") if a.models else DEFAULT_GRID_MODELS
    sets = a.feature_sets.split(",") if a.feature_sets else DEFAULT_GRID_SETS
    modes = a.modes.split(",") if a.modes else DEFAULT_MODES
    nn_sets = a.nn_feature_sets.split(",") if a.nn_feature_sets else ["core", "cyclenorm", "no_clock", "core_macro"]
    for m in models:
        for fs in sets:
            if m in ("nn", "ens_all") and fs not in nn_sets:
                continue
            if m == "rules" and fs != sets[0]:
                continue  # rules ignore the feature set
            run_walkforward(m, fs, modes=modes, store=True, verbose=True)
            sys.stdout.flush()
    cmd_leaderboard(a)


def cmd_leaderboard(a):
    from .backtest.walkforward import leaderboard
    lb = leaderboard()
    pd.set_option("display.width", 200)
    print(lb.drop(columns=["created_at"]).head(getattr(a, "top", 25)).round(3).to_string(index=False))


def cmd_train_production(a):
    from .pipeline import train_production
    train_production(a.run_id, role="production")
    if not a.skip_nn:
        train_production(a.nn_run_id, role="reference_nn")


def cmd_daily(a):
    from .live import run_daily
    run_daily(notify=not a.no_notify, force_notify=a.force_notify)


def cmd_setup_ntfy(a):
    from .notify import ensure_topic
    topic = ensure_topic()
    print(f"ntfy topic: {topic}\nSubscribe in the ntfy app (server https://ntfy.sh) to this topic.")


def cmd_notify_test(a):
    from .notify import Alert, dispatch, ensure_topic
    ensure_topic()
    sent = dispatch(pd.Timestamp.now().strftime("%Y-%m-%d"),
                    [Alert("test", "BTC_Neural test notification", "If you can read this, notifications work.", 3, "white_check_mark")],
                    force=True)
    print("sent:", sent)


def cmd_send_chart(a):
    from .live import send_chart
    print("sent", send_chart())


def cmd_app(a):
    app = ROOT / "src" / "btc_neural" / "app" / "Home.py"
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"])


PLIST_LABEL = "com.btcneural.daily"


def _plist_path():
    from pathlib import Path
    return Path.home() / "Library" / "LaunchAgents" / f"{PLIST_LABEL}.plist"


def cmd_install_schedule(a):
    """Install a launchd agent running the daily job at 00:30 and 06:00 UTC (converted to local time)."""
    import time
    offset_h = -time.timezone / 3600 if not time.localtime().tm_isdst else -time.altzone / 3600
    times = [((0.5 + offset_h) % 24), ((6 + offset_h) % 24)]
    entries = "".join(
        f"<dict><key>Hour</key><integer>{int(t)}</integer><key>Minute</key><integer>{int(round((t % 1) * 60))}</integer></dict>"
        for t in times)
    uv = subprocess.run(["which", "uv"], capture_output=True, text=True).stdout.strip() or "/opt/homebrew/bin/uv"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{PLIST_LABEL}</string>
  <key>ProgramArguments</key><array>
    <string>{uv}</string><string>run</string><string>--project</string><string>{ROOT}</string><string>btcn</string><string>daily</string>
  </array>
  <key>WorkingDirectory</key><string>{ROOT}</string>
  <key>StartCalendarInterval</key><array>{entries}</array>
  <key>StandardOutPath</key><string>{logs}/daily.log</string>
  <key>StandardErrorPath</key><string>{logs}/daily.err.log</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
</dict></plist>
"""
    p = _plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
    p.write_text(plist)
    r = subprocess.run(["launchctl", "load", str(p)], capture_output=True, text=True)
    print(f"Installed {p}\nRuns daily at local {', '.join(f'{int(t):02d}:{int(round((t % 1) * 60)):02d}' for t in times)} "
          f"(00:30 and 06:00 UTC). {r.stderr.strip()}")


def cmd_uninstall_schedule(a):
    p = _plist_path()
    subprocess.run(["launchctl", "unload", str(p)], capture_output=True)
    if p.exists():
        p.unlink()
    print(f"Removed {p}")


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(prog="btcn", description="BTC_Neural: cycle buy/sell signals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download/refresh all data").set_defaults(fn=cmd_ingest)
    sub.add_parser("features", help="show latest feature values").set_defaults(fn=cmd_features)
    b = sub.add_parser("backtest", help="walk-forward grid over models x feature sets x strength modes")
    b.add_argument("--models")
    b.add_argument("--feature-sets")
    b.add_argument("--modes")
    b.add_argument("--nn-feature-sets")
    b.set_defaults(fn=cmd_backtest)
    lb = sub.add_parser("leaderboard", help="walk-forward results, best first")
    lb.add_argument("--top", type=int, default=25)
    lb.set_defaults(fn=cmd_leaderboard)
    tp = sub.add_parser("train-production", help="retrain the best run (and best NN) on all data")
    tp.add_argument("--run-id")
    tp.add_argument("--nn-run-id")
    tp.add_argument("--skip-nn", action="store_true")
    tp.set_defaults(fn=cmd_train_production)
    d = sub.add_parser("daily", help="ingest, predict, log and notify")
    d.add_argument("--no-notify", action="store_true")
    d.add_argument("--force-notify", action="store_true")
    d.set_defaults(fn=cmd_daily)
    sub.add_parser("setup-ntfy", help="create a private random ntfy topic in .env").set_defaults(fn=cmd_setup_ntfy)
    sub.add_parser("notify-test", help="send a test notification").set_defaults(fn=cmd_notify_test)
    sub.add_parser("send-chart", help="push today's Home chart and forecast to ntfy").set_defaults(fn=cmd_send_chart)
    sub.add_parser("app", help="launch the Streamlit dashboard").set_defaults(fn=cmd_app)
    sub.add_parser("install-schedule", help="install the daily launchd job").set_defaults(fn=cmd_install_schedule)
    sub.add_parser("uninstall-schedule", help="remove the daily launchd job").set_defaults(fn=cmd_uninstall_schedule)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
