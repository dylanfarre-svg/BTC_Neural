"""Horizon specs: everything that differs between the swing and intraday signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import load_config


@dataclass(frozen=True)
class Horizon:
    name: str
    bars: str                      # decision frequency: 1d | 4h
    resolve_on: str                # path resolution for barriers
    hold_min_days: float
    hold_max_days: float
    sides: tuple[str, ...]
    sigma_mult: float
    floor: float
    stop_ratio: float
    vol_window_days: int
    cooldown_days: int = 5
    target_trades_per_year: float = 20
    fee_rate: float = 0.001
    slippage: float = 0.0005
    funding_enabled: bool = True
    funding_sides: tuple[str, ...] = ("short",)
    funding_fallback_8h: float = 0.0001
    capital_rate: float = 0.18
    revenue_rate: float = 0.45
    tax_year_start_month: int = 3
    block_days: int = 90
    purge_days: int = 28
    embargo_days: int = 10
    min_train_days: int = 730
    train_start: pd.Timestamp = pd.Timestamp("2012-01-01")
    warmup_start: pd.Timestamp = pd.Timestamp("2014-01-01")
    backtest_start: pd.Timestamp = pd.Timestamp("2015-01-01")
    short_start: pd.Timestamp = pd.Timestamp("2019-01-01")
    modern_start: pd.Timestamp = pd.Timestamp("2019-01-01")
    holdout_days: int = 365
    holdout_start: pd.Timestamp = pd.Timestamp("2025-10-30")
    threshold_trailing_days: int = 730
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    min_trades_per_year: float = 12
    max_trades_per_year: float = 30
    extra: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_rate + self.slippage)

    def with_costs(self, fee_rate: float | None = None, slippage: float | None = None,
                   funding_enabled: bool | None = None) -> "Horizon":
        from dataclasses import replace
        return replace(self, fee_rate=self.fee_rate if fee_rate is None else fee_rate,
                       slippage=self.slippage if slippage is None else slippage,
                       funding_enabled=self.funding_enabled if funding_enabled is None else funding_enabled)


def horizon(name: str = "swing") -> Horizon:
    c = load_config()["trade"][name]
    b, f = c["barrier"], c["folds"]
    costs = c.get("costs", {})
    fund = costs.get("funding", {})
    tax = c.get("tax", {})
    cap = tax.get("capital", {"inclusion_rate": 0.40, "marginal_rate": 0.45})
    th = c.get("threshold", {})
    if costs.get("fee_rate", 0.001) is None:
        raise ValueError("trade.%s.costs.fee_rate is not set - see PLAN_SWING.md Q4" % name)
    ts = lambda k, d: pd.Timestamp(c.get(k, d))  # noqa: E731
    return Horizon(
        name=name, bars=c["bars"], resolve_on=c["resolve_on"],
        hold_min_days=float(c["hold_days"]["min"]), hold_max_days=float(c["hold_days"]["max"]),
        sides=tuple(c.get("sides", ["long"])),
        sigma_mult=float(b["sigma_mult"]), floor=float(b["floor"]), stop_ratio=float(b["stop_ratio"]),
        vol_window_days=int(b.get("vol_window_days", 30)),
        cooldown_days=int(c.get("cooldown_days", 5)),
        target_trades_per_year=float(c.get("target_trades_per_year", 20)),
        fee_rate=float(costs.get("fee_rate", 0.001)), slippage=float(costs.get("slippage", 0.0005)),
        funding_enabled=bool(fund.get("enabled", True)), funding_sides=tuple(fund.get("sides", ["short"])),
        funding_fallback_8h=float(fund.get("fallback_8h", 0.0001)),
        capital_rate=float(cap["inclusion_rate"]) * float(cap["marginal_rate"]),
        revenue_rate=float(tax.get("revenue", {}).get("marginal_rate", 0.45)),
        tax_year_start_month=int(tax.get("tax_year_start_month", 3)),
        block_days=int(f["block_days"]), purge_days=int(f["purge_days"]), embargo_days=int(f["embargo_days"]),
        min_train_days=int(f.get("min_train_days", 730)),
        train_start=pd.Timestamp(f.get("train_start", "2012-01-01")),
        warmup_start=pd.Timestamp(f.get("warmup_start", "2014-01-01")),
        backtest_start=ts("backtest_start", "2015-01-01"), short_start=ts("short_start", "2019-01-01"),
        modern_start=ts("modern_start", "2019-01-01"), holdout_days=int(c.get("holdout_days", 365)),
        holdout_start=ts("holdout_start", "2025-10-30"),
        threshold_trailing_days=int(th.get("trailing_days", 730)),
        quantiles=tuple(float(q) for q in th.get("quantiles", [0.90, 0.95, 0.975])),
        min_trades_per_year=float(c.get("min_trades_per_year", 12)),
        max_trades_per_year=float(c.get("max_trades_per_year", 30)),
    )
