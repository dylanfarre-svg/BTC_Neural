"""Trading signals: discrete trade proposals on short horizons (see PLAN_SWING.md).

Separate from the cycle signal by design. Everything here takes a `Horizon`, so the planned
4h-4d signal is a second configuration rather than a second package. Swing signals are logged
to the append-only `trade_signals` table and are never pushed to ntfy.
"""
