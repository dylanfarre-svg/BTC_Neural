# BTC_Neural

Daily Bitcoin **cycle** signal on a signed scale from **−10 (max sell) to +10 (max buy)**, built for a
multi-year accumulate-and-trim strategy. It targets about 10 strength-10 sell days near each cycle top and 10
strength-10 buy days near each cycle low. It is research software, not financial advice.

See [PLAN.md](PLAN.md) for the design rationale.

## Quick start

```bash
uv sync                       # install
uv run btcn ingest            # download data into data/btc_neural.db (SQLite)
uv run btcn backtest          # walk-forward grid: models x feature sets x strength modes
uv run btcn leaderboard       # results, best first
uv run btcn train-production  # retrain the winner (and the best NN reference) on all data
uv run btcn daily             # ingest -> predict -> log -> notify
uv run btcn app               # dashboard at http://localhost:8501
```

Notifications: `uv run btcn setup-ntfy` writes a private random topic to `.env`. Subscribe to it in the
ntfy app (server `ntfy.sh`). `uv run btcn notify-test` sends a test message.
Scheduling: `uv run btcn install-schedule` installs a launchd agent that runs `btcn daily` at 00:30 and
06:00 UTC. If the Mac was asleep, the job runs on wake. `uninstall-schedule` removes it.

## How it works

| Stage | What happens | Code |
|---|---|---|
| Data | Bitstamp 1d/4h candles, Coin Metrics BTC price + on-chain (MVRV, hash rate, issuance, addresses) and ETH price/market cap/addresses (2015-08+), FRED macro, Fear & Greed | `ingest/` |
| Features | ~87 causal, scale-free features: MA ratios, Pi Cycle, log-regression residual, daily/weekly/monthly RSI (the current bar is partial), MVRV-Z, NUPL, Puell, hash ribbons, halving clock, causal "confirmed" cycle tops/lows, macro (with publication lags), 4-year percentiles, and ETH/BTC + ETH/USD relative strength | `features/` |
| Label | Hindsight position within the cycle leg, from +1 (cycle low) to −1 (cycle top), computed **as of each training cutoff** | `labels.py` |
| Models | Rule composite (no training), ridge, LightGBM with monotone constraints, a small temporal-CNN neural net (8 seeds, Apple GPU), ensembles | `models/` |
| Strength | The score maps to −10..+10 by budget (strength 10 = 10 days per 1460). Modes: `frozen` thresholds from training, `trailing` causal 3-year rank of the model's own scores, `hybrid`, `both` | `calibrate.py` |
| Backtest | Halving epochs E1..E4. Each epoch is predicted by a model trained only on earlier epochs | `backtest/walkforward.py` |
| Metrics | Distance of strength-10 signals from the true top/low (log), days from the extreme, wrong-side share, budget. **Selection = worst fold**, lower is better | `backtest/metrics.py` |
| Simulation | Monthly DCA + reserve, buys on buy signals, trims on sell signals above a core floor (25%), South African CGT in rand (40% inclusion × marginal rate, R40k exclusion, FIFO/HIFO/average lots, loss carry-forward) | `backtest/simulator.py` |
| Production | Best run retrained on all data, plus the best configuration of **every** model type, all saved under `runs/` | `pipeline.py` |
| Forecast | Next cycle extreme (date + price) as Gaussians: a statistical cycle-analog model, and one per saved model from its own out-of-sample score history | `forecast.py` |
| Live | Append-only `live_signals` table (enforced by SQLite triggers), alerts via ntfy + macOS | `live.py`, `notify.py` |

### Alerts
| Kind | When |
|---|---|
| Strength 10 | Any ±10 day (urgent priority, with chart) |
| Strong | \|strength\| ≥ 7 |
| Regime shift | Moves ≥ 3 levels in 7 days, or flips between the buy and sell zones |
| NN disagreement | NN reference differs from production by ≥ 5 levels |
| Weekly digest | Sundays |
| Data problem | Ingest failure or stale data (a silent failure could hide a strength-10 day) |

## Feature sets

`core` (price + on-chain + cycle clock), `cyclenorm` (amplitude-free only), `core_macro`, `core_calendar`,
`core_intraday`, `price_clock`, `no_clock`, and `core_eth` / `cyclenorm_eth` which add the Ether inputs
(ETH/BTC ratio vs its own year average, 90/365-day moves, drawdown from the ETH/BTC high, 4-year
percentile, ETH/USD trend and volatility relative to BTC, ETH share of BTC market cap, ETH address
growth). ETH history starts 2015-08, so those columns are empty for the earliest cycles.

## Saved models

Every walk-forward run keeps its out-of-sample predictions in the `predictions` table, and
`btcn train-production` saves one trained model per type (`candidate_<type>` roles), so each model's
signal and forecast stay available. The Home chart can display any of them.

## Configuration

`config/default.yaml` holds everything: cycle extremes (used only for labels and evaluation), signal budget,
strength mode, simulation and tax settings, model hyperparameters and alert thresholds. Put overrides in
`config/local.yaml`.

**When the current cycle's low is confirmed**, add it to `extremes` in the config. Then rerun
`btcn backtest` and `btcn train-production`. This adds the buy side of cycle E4 to the evaluation.

## Tests

```bash
uv run pytest
```

`tests/test_no_lookahead.py` rebuilds all features on data truncated at six dates and asserts that they
match the full-history values exactly. That makes the backtest honest.

## Known limitations
- Only about 4 cycles of history: every result is a handful of independent observations.
- Each cycle's amplitude is smaller than the last. The adaptive (`trailing`) strength mode exists because
  thresholds fixed on past cycles often never fire in the next one.
- Free on-chain data (Coin Metrics Community) lacks some metrics (SOPR, LTH supply). The code accepts extra
  series if you later add a paid source.
- The tax model is simplified and is not tax advice.
