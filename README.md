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
uv run btcn train-production  # retrain the winner, the NN reference and one model per type
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
| Data | Bitstamp 1d/4h candles, Coin Metrics BTC price + on-chain (MVRV, hash rate, issuance, addresses, exchange supply/flows, fees) and ETH price/market cap/addresses (2015-08+), FRED macro (M2, Fed balance sheet, fed funds rate, 10-year nominal/real yields, 2s10s curve, dollar index, USD/ZAR), Fear & Greed | `ingest/` |
| Features | ~87 causal, scale-free features: MA ratios, Pi Cycle, log-regression residual, daily/weekly/monthly RSI (the current bar is partial), MVRV-Z, NUPL, Puell, hash ribbons, halving clock, causal "confirmed" cycle tops/lows, macro (with publication lags), 4-year percentiles, and ETH/BTC + ETH/USD relative strength | `features/` |
| Label | Hindsight position within the cycle leg, from +1 (cycle low) to −1 (cycle top), computed **as of each training cutoff** | `labels.py` |
| Models | Rule composite (no training), ridge, LightGBM with monotone constraints, a small temporal-CNN neural net (8 seeds, Apple GPU; weekly sequence of 10 BTC series plus 3 ETH/BTC channels when the feature set includes Ether), ensembles | `models/` |
| Strength | The score maps to −10..+10 by budget (strength 10 = 10 days per 1460). Modes: `frozen` thresholds from training, `trailing` causal 3-year rank of the model's own scores, `hybrid`, `both`. A **de-clustering** rule then thins strength-10 runs: after one fires, the next needs 120 days or a 0.05 more extreme score | `calibrate.py` |
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

## In-sample cutoff (holdout)

`training_cutoff` in `config/default.yaml` (currently **2026-01-01**) is the last date the saved models may
learn from. Labels, model fitting and calibration thresholds all stop there, so everything after it is a
genuine holdout that the append-only `live_signals` table records day by day. Set it to `null` to train on
everything. It does not affect walk-forward folds, which already stop at the start of each test epoch.

Raise it once you want the models to learn from the recent past — for example after the current cycle's low
is confirmed and added to `extremes` — then rerun `btcn train-production`.

## Feature sets

A set is a list of feature groups, optionally plus single columns named as `+column`.

| Set | Contents |
|---|---|
| `core` | price/technical + on-chain + cycle clock — **the production set** |
| `cyclenorm` | amplitude-free only (4-year percentiles, oscillators, clocks) |
| `core_ma` | + 20W/50W ratios and MA structure (slopes, 20W-50W gap, weeks since cross) |
| `core_ma4050` | + price/40W and price/50W ratios only |
| `core_exhaust` | + parabola-exhaustion measures (acceleration, 30-day move in sigmas, days above the 20W MA, time monthly RSI spends above 70) |
| `core_flow` | + exchange supply/flows, fee pressure, address base |
| `core_tier1` | + flow and exhaustion together |
| `core_macro` | + 13 macro inputs (M2, Fed balance sheet, fed funds level/direction/percentile, 10-year nominal and real yields, 2s10s curve, dollar index), each lagged to its publication date |
| `core_eth` / `cyclenorm_eth` | + Ether: ETH/BTC vs its year average, 90/365-day moves, drawdown from the ETH/BTC high, 4-year percentile, ETH/USD trend, ETH volatility vs BTC, ETH share of BTC market cap, ETH address growth (2015-08+) |
| `core_calendar`, `core_intraday`, `price_clock`, `no_clock` | ablations |

## Experiment log (what has and hasn't helped)

Walk-forward mean over seeds, worst-cycle score (lower is better). Production `core` = **0.679 ± 0.006** (6 seeds).

| Added to `core` | Mean | Verdict |
|---|---|---|
| 20W/50W MA ratios + MA structure (`core_ma`) | 0.670 | tie |
| Parabola-exhaustion measures (`core_exhaust`) | 0.680 | tie |
| Exhaustion + 40W/50W (`core_exhaust_ma4050`) | 0.689 ± 0.014 (6 seeds) | tie |
| Ether, ETH/BTC also in the NN sequence (`core_eth`) | 0.691 | tie / slightly worse |
| 40W/50W alone (`core_ma4050`) | 0.813 | worse |
| Macro incl. fed funds (`core_macro`) | 0.877 | worse |
| Exchange flows + fees (`core_flow`) | 0.980 | worse |
| Exchange flows + exhaustion (`core_tier1`) | 0.933 | worse |
| *Label* changed to `window` (365d / 540d) | 2.240 | much worse — no strength-10 sells at all in E4 |
| **De-clustering, cooldown 120d / 0.05** | **0.511 ± 0.012** (6 seeds) | **adopted** |
| De-clustering, cooldown 60d / 0.05 | 0.526 ± 0.017 (6 seeds) | also good (rule is not knife-edge) |
| Trailing window 730 / 900 / 1000 / **1095** / 1200 / 1300 / 1460 days | 1.894 / 1.489 / 0.586 / **0.495** / 1.393 / 2.240 / 2.240 | current value already best |
| Signal budget x0.5 (5 days/cycle) / x1 / x2 (20) | 0.661 / **0.495** / 0.537 | current value already best |

### The trailing window is the fragile part

Strength comes from ranking today's score against the model's own scores over the trailing window, and only
~1000-1100 days works. Too long and the window still contains the **previous same-side extreme**, whose
score today can never beat, so nothing fires at all (1300+ scores 2.240 = no strength-10 signals). Too short
and the window holds only the recent regime, so signals fire early and often (730 scores 1.89). 1095 works
because it reaches past the last cycle low but stops short of the last top — which quietly assumes the
4-year rhythm holds. If cycles lengthen, this breaks. The better design (not built yet) is an adaptive
window measured from the **last causally confirmed opposite extreme**, which `features.confirmed_extremes`
already computes.

No *input* has ever beaten the original feature set, and neither did re-tuning the calibration; the one
change that did was the de-clustering rule,
which cut the 2017 fold's strength-10 sells from 68 (median 64% below the top) to 6 (median 47% below,
16 days away). Its parameters were chosen by scanning the same test folds, so the true out-of-sample gain
is likely smaller than 0.679 -> 0.511; the whole cooldown family (30-120 days, 0.02-0.10) beat the
baseline, which is what makes it credible. The append-only live signal log is the unbiased test.

Differences below ~0.05 are inside seed noise; three seeds is not enough to separate close candidates
(a 3-seed run made `core_exhaust_ma4050` look like a winner at 0.661; six seeds reversed it).

## Saved models

Every walk-forward run keeps its out-of-sample predictions in the `predictions` table, and
`btcn train-production` saves the winner (`production`), an NN reference, one model per model type
(`candidate_<type>`) and one per experiment worth keeping (`candidate_nn_eth`, `candidate_nn_macro`,
`candidate_nn_ma`, `candidate_nn_exhaust`, …). Each stores its own calibration, de-clustering rule and
cutoff in `meta.json`, so a live signal always matches the backtest that produced it. The Home chart can
display any of them, and each also produces its own next-extreme forecast.

## Dashboard

`uv run btcn app` → http://localhost:8501. Restart it after code changes; Streamlit only reloads page files,
not imported modules.

| Page | What it shows |
|---|---|
| Home | Today's signal and confidence, what drives it, price chart with signals and a strength panel, next-extreme forecasts (pick which models to display; one is shaded as a Gaussian, the rest are markers with 1σ error bars) |
| Backtest | Out-of-sample signals per test epoch, signal-quality metrics per cycle, tax-aware simulation vs plain DCA |
| Leaderboard | Every run ranked by worst-cycle score, heat-shaded |
| Features | Any feature against log price, with cycle extremes marked |
| Train | Run a walk-forward experiment and promote a run to production |
| Data & alerts | Data freshness, ingestion log, live signal log, notification log and a test button |

Every price chart has a **Show halvings** toggle and keeps the date tooltip alive past the end of the data.

## Configuration

`config/default.yaml` holds everything: cycle extremes (used only for labels and evaluation), `training_cutoff`, the signal
budget, strength mode, de-clustering rule, trailing-window length, simulation and South African tax
settings, model hyperparameters, forecast settings and alert thresholds. Put overrides in
`config/local.yaml`.

**When the current cycle's low is confirmed**, add it to `extremes` in the config. Then rerun
`btcn backtest` and `btcn train-production`. This adds the buy side of cycle E4 to the evaluation.

## Tests

```bash
uv run pytest
```

`tests/test_no_lookahead.py` rebuilds all features on data truncated at six dates and asserts that they
match the full-history values exactly. That makes the backtest honest.

## Method notes
- **Judge a change on 6 seeds, not 3.** The neural net's seed spread (~0.03) is as large as most effects
  being measured; a 3-seed run once made a tied configuration look like a winner.
- **Promote only on the worst fold and the seed mean together**, decided before looking.
- Anything tuned by scanning the test folds (the de-clustering parameters) is optimistic by construction.
  Treat the live log as the only unbiased evidence.

## Known limitations
- Only about 4 cycles of history: every result is a handful of independent observations.
- Each cycle's amplitude is smaller than the last. The adaptive (`trailing`) strength mode exists because
  thresholds fixed on past cycles often never fire in the next one.
- Free on-chain data (Coin Metrics Community) lacks some metrics (SOPR, LTH supply). The code accepts extra
  series if you later add a paid source.
- The tax model is simplified and is not tax advice.
