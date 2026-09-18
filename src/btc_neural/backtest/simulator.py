"""Portfolio simulation with South African capital gains tax (secondary metric).

Policy (configurable in config/default.yaml -> simulation):
- Each month a fixed USD contribution arrives. A share is bought immediately (DCA core);
  the rest accumulates as a cash reserve.
- Buy signals deploy a fraction of the reserve per day, rising with strength.
- Sell signals sell a fraction of the BTC held above the core floor
  (core_floor_fraction x peak BTC held), rising with strength.
- Tax: gains are computed in rand (USD/ZAR from FRED). Per SA tax year (1 Mar - end Feb):
  net gain - annual exclusion, times inclusion rate, times marginal rate. Net capital
  losses carry forward. Tax is paid from cash at year end. Lot method: FIFO / HIFO / average.

This is a simplified model for comparing strategies, not tax advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import load_config


@dataclass
class Lot:
    date: pd.Timestamp
    btc: float
    cost_usd: float
    cost_zar: float


@dataclass
class SimResult:
    summary: dict
    equity: pd.DataFrame  # date, btc, cash, value_usd
    trades: pd.DataFrame
    taxes: pd.DataFrame = field(default_factory=pd.DataFrame)


def _tax_year(d: pd.Timestamp, start_month: int) -> int:
    return d.year + 1 if d.month >= start_month else d.year


def simulate(close: pd.Series, strength: pd.Series, usdzar: pd.Series | None, start: pd.Timestamp,
             end: pd.Timestamp, params: dict | None = None, dca_only: bool = False) -> SimResult:
    p = {**load_config()["simulation"], **(params or {})}
    tax = p["tax"]
    buy_frac = {int(k): v for k, v in p["buy_fraction"].items()}
    sell_frac = {int(k): v for k, v in p["sell_fraction"].items()}
    fee = p["fee_rate"]
    c = close.loc[start:end]
    fx = (usdzar.reindex(usdzar.index.union(c.index)).ffill().bfill().reindex(c.index)
          if usdzar is not None and len(usdzar) else pd.Series(1.0, index=c.index))
    s = strength.reindex(c.index).fillna(0).astype(int)

    lots: list[Lot] = []
    cash = 0.0
    peak_btc = 0.0
    contributed = 0.0
    year_gain = 0.0
    loss_cf = 0.0
    trades, equity, taxes = [], [], []
    ty = _tax_year(c.index[0], tax["tax_year_start_month"])
    last_month = None

    def holdings() -> float:
        return sum(lt.btc for lt in lots)

    def buy(d, usd, reason):
        nonlocal cash, peak_btc
        if usd <= 1e-9:
            return
        price = c.loc[d]
        btc = usd * (1 - fee) / price
        lots.append(Lot(d, btc, usd, usd * fx.loc[d]))
        cash -= usd
        peak_btc = max(peak_btc, holdings())
        trades.append({"date": d, "side": "buy", "btc": btc, "usd": usd, "price": price, "reason": reason})

    def sell(d, btc):
        nonlocal cash, year_gain, lots
        price = c.loc[d]
        proceeds = btc * price * (1 - fee)
        method = tax["lot_method"]
        if method == "average":
            tot = holdings()
            cost_zar = sum(lt.cost_zar for lt in lots) * btc / tot
            cost_usd = sum(lt.cost_usd for lt in lots) * btc / tot
            for lt in lots:
                f = 1 - btc / tot
                lt.btc, lt.cost_usd, lt.cost_zar = lt.btc * f, lt.cost_usd * f, lt.cost_zar * f
        else:
            order = sorted(lots, key=(lambda lt: -lt.cost_usd / lt.btc) if method == "hifo" else (lambda lt: lt.date))
            remaining, cost_zar, cost_usd = btc, 0.0, 0.0
            for lt in order:
                if remaining <= 1e-12:
                    break
                take = min(lt.btc, remaining)
                f = take / lt.btc
                cost_zar += lt.cost_zar * f
                cost_usd += lt.cost_usd * f
                lt.btc, lt.cost_usd, lt.cost_zar = lt.btc - take, lt.cost_usd * (1 - f), lt.cost_zar * (1 - f)
                remaining -= take
        lots = [lt for lt in lots if lt.btc > 1e-12]
        gain_zar = proceeds * fx.loc[d] - cost_zar
        year_gain += gain_zar
        cash += proceeds
        trades.append({"date": d, "side": "sell", "btc": btc, "usd": proceeds, "price": price,
                       "gain_zar": gain_zar, "cost_usd": cost_usd})

    def settle_tax(d, final=False):
        nonlocal cash, year_gain, loss_cf
        net = year_gain - loss_cf
        if net < 0:
            loss_cf, taxable = -net, 0.0
        else:
            loss_cf = 0.0
            taxable = max(0.0, net - tax["annual_exclusion_zar"])
        t_zar = taxable * tax["inclusion_rate"] * tax["marginal_rate"]
        t_usd = t_zar / fx.loc[d]
        cash -= t_usd
        taxes.append({"tax_year": ty, "net_gain_zar": net, "tax_zar": t_zar, "tax_usd": t_usd, "accrued_only": final})
        year_gain = 0.0

    for d in c.index:
        new_ty = _tax_year(d, tax["tax_year_start_month"])
        if new_ty != ty:
            settle_tax(d)
            ty = new_ty
        if last_month != (d.year, d.month):
            last_month = (d.year, d.month)
            amt = p["monthly_contribution_usd"]
            contributed += amt
            cash += amt
            buy(d, amt if dca_only else amt * p["base_dca_fraction"], "dca")
        if not dca_only:
            k = int(s.loc[d])
            if k > 0 and k in buy_frac and cash > 0:
                buy(d, cash * buy_frac[k], f"signal+{k}")
            elif k < 0 and -k in sell_frac:
                sellable = holdings() - p["core_floor_fraction"] * peak_btc
                if sellable > 1e-9:
                    sell(d, sellable * sell_frac[-k])
        h = holdings()
        equity.append({"date": d, "btc": h, "cash": cash, "value_usd": h * c.loc[d] + cash})

    settle_tax(c.index[-1], final=True)
    tr = pd.DataFrame(trades)
    tx = pd.DataFrame(taxes)
    eq = pd.DataFrame(equity).set_index("date")
    buys = tr[tr.side == "buy"] if len(tr) else tr
    sells = tr[tr.side == "sell"] if len(tr) else tr
    final_price = c.iloc[-1]
    summary = {
        "contributed_usd": contributed,
        "final_btc": holdings(),
        "final_cash_usd": cash,
        "final_value_usd": holdings() * final_price + cash,
        "tax_paid_usd": float(tx.loc[~tx.accrued_only, "tax_usd"].sum()) if len(tx) else 0.0,
        "tax_accrued_usd": float(tx.loc[tx.accrued_only, "tax_usd"].sum()) if len(tx) else 0.0,
        "avg_buy_price": float(buys.usd.sum() / buys.btc.sum()) if len(buys) else np.nan,
        "avg_sell_price": float((sells.usd.sum()) / sells.btc.sum()) if len(sells) else np.nan,
        "n_sells": int(len(sells)),
        "btc_sold": float(sells.btc.sum()) if len(sells) else 0.0,
    }
    return SimResult(summary, eq, tr, tx)
