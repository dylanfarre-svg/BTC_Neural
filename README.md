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

## Feature reference (122 features)

Names are exactly as they appear in the app's **Features** page, which shows the same description.
Everything is causal (only data available that day) and scale-free. Log values: 0 = equal, +0.69 = double,
−0.69 = half. `_pct4y` = percentile within the last 4 years (0–1); `_pctx` = percentile of all history so far.

### Price / technical — group `price`, 34 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `mayer` | Mayer Multiple in logs: price vs its 200-day average. >0 above, historically ~+0.9 at tops. |
| `p_200w` | Price vs its 200-week average (log). The 200w average has acted as the bear-market floor: near 0 at cycle lows. |
| `p_2yma` | Price vs its 2-year average (log). |
| `pi_cycle` | Pi Cycle Top: log gap between the 111-day average and twice the 350-day average. Crossing 0 flagged the 2013/2017/2021 tops. |
| `macd_w` | Weekly-style MACD: 12-week minus 26-week EMA, divided by price. |
| `ret_30` | 30-day log return. |
| `ret_90` | 90-day log return. |
| `ret_180` | 180-day log return. |
| `ret_365` | 1-year log return. |
| `vol_30` | 30-day realised volatility, annualised. |
| `vol_90` | 90-day realised volatility, annualised. |
| `vol_365` | 1-year realised volatility, annualised. |
| `vol_ratio` | Log ratio of 30-day to 1-year volatility: >0 means the market is more volatile than usual. |
| `dd_ath` | Drawdown from the all-time high (log). 0 = at a new high, -1.4 = 75% below. |
| `days_since_ath` | Days since the last all-time high, in 4-year units (1.0 = 1460 days). |
| `rise_2y_low` | How far price has risen above its lowest point in the last 2 years (log). |
| `rsi_d` | Daily RSI(14). |
| `rsi_w` | Weekly RSI(14), with this week's bar still forming - what a live chart shows. |
| `rsi_m` | Monthly RSI(14), with this month's bar still forming. Historically >90 at cycle tops. |
| `rsi_m_div` | Monthly RSI minus its own highest value of the past year: 0 = at a new RSI high, negative = momentum diverging. |
| `bb_width` | Bollinger-band width over 140 days (log): high = wide/volatile, low = a squeeze. |
| `logreg_resid` | Distance above/below the long-term log-log growth trend, refit monthly on past data only. |
| `mayer_pctx` | Mayer Multiple as a percentile of its whole history to date (0-1). |
| `p_200w_pctx` | Price/200-week average as a percentile of its whole history to date. |
| `rsi_m_pctx` | Monthly RSI as a percentile of its whole history to date. |
| `pi_cycle_pctx` | Pi Cycle gap as a percentile of its whole history to date. |
| `mayer_pct4y` | Mayer Multiple as a percentile of the last 4 years: 1.0 = hottest in a cycle. Cycle-relative, so it still reaches extremes as amplitudes shrink. |
| `p_200w_pct4y` | Price/200-week average as a percentile of the last 4 years. |
| `rsi_m_pct4y` | Monthly RSI as a percentile of the last 4 years. |
| `pi_cycle_pct4y` | Pi Cycle gap as a percentile of the last 4 years. |
| `p_2yma_pct4y` | Price/2-year average as a percentile of the last 4 years. |
| `logreg_resid_pct4y` | Trend residual as a percentile of the last 4 years. |
| `ret_365_pct4y` | 1-year return as a percentile of the last 4 years. |
| `vol_90_pct4y` | 90-day volatility as a percentile of the last 4 years. |

### Moving averages — group `ma`, 11 features, from 2010-09-25

| Feature | Meaning |
|---|---|
| `p_20w` | Price vs its 20-week (140-day) average (log) - the classic bull-market support line. |
| `p_40w` | Price vs its 40-week (280-day) average (log). |
| `p_50w` | Price vs its 50-week (350-day) average (log). |
| `p_40w_pct4y` | Price/40-week average as a percentile of the last 4 years. |
| `p_50w_pct4y` | Price/50-week average as a percentile of the last 4 years. |
| `ma20w_slope` | Slope of the 20-week average: its 90-day log change. >0 = rising. |
| `ma50w_slope` | Slope of the 50-week average: its 180-day log change. |
| `ma20w_vs_50w` | Log gap between the 20-week and 50-week averages: >0 after a golden cross, <0 after a death cross. |
| `weeks_since_ma_cross` | Time since the 20-week average last crossed the 50-week, in 4-year units. |
| `p_20w_pct4y` | Price/20-week average as a percentile of the last 4 years. |
| `ma20w_vs_50w_pct4y` | That 20W/50W gap as a percentile of the last 4 years. |

### On-chain valuation & network — group `onchain`, 14 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `mvrv` | MVRV in logs: market value vs realised value (what all coins last moved at). High = large unrealised profits. |
| `mvrv_z` | MVRV Z-score: (market cap - realised cap) / expanding std of market cap. A classic top marker, though it peaks lower each cycle. |
| `nupl` | Net Unrealised Profit/Loss: share of market value that is paper profit. <0 = the average coin is under water. |
| `puell` | Puell Multiple (log): daily miner issuance value vs its 1-year average. High = miner revenue unusually rich. |
| `hash_ribbon` | Hash ribbon: log ratio of 30-day to 60-day hash rate. <0 = miner capitulation, historically near lows. |
| `hash_growth` | 1-year growth in hash rate (log). |
| `adr_growth` | 1-year growth in active addresses (log). |
| `tx_growth` | 1-year growth in transaction count (log). |
| `mvrv_pctx` | MVRV as a percentile of its whole history to date. |
| `mvrv_pct4y` | MVRV as a percentile of the last 4 years. |
| `puell_pctx` | Puell Multiple as a percentile of its whole history to date. |
| `puell_pct4y` | Puell Multiple as a percentile of the last 4 years. |
| `mvrv_z_pctx` | MVRV Z-score as a percentile of its whole history to date. |
| `mvrv_z_pct4y` | MVRV Z-score as a percentile of the last 4 years. |

### Cycle clock — group `clock`, 10 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `days_since_halving` | Time since the last halving, in 4-year units. Tops have come ~0.36-0.38 (525-550 days) after one. |
| `halving_phase_sin` | Sine of the halving phase - lets the model treat the 4-year clock as circular. |
| `halving_phase_cos` | Cosine of the halving phase. |
| `days_since_top1` | Time since the most recent *confirmed* cycle top, in 4-year units. Confirmation is causal: a top counts only once price has halved from it. |
| `days_since_top2` | Time since the confirmed top before that. |
| `days_since_low1` | Time since the most recent confirmed cycle low (confirmed once price has doubled off it). |
| `days_since_low2` | Time since the confirmed low before that. |
| `vs_last_top` | Price vs the last confirmed top (log): 0 = back at that level. |
| `vs_last_low` | Price vs the last confirmed low (log). |
| `top_minus_low_recency` | Which confirmed extreme is more recent, and by how much: >0 = the low came more recently (we are rising). |

### Parabola exhaustion — group `exhaust`, 7 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `accel_90_365` | Acceleration: daily pace of the last 90 days minus that of the last year. >0 = the advance is speeding up. |
| `accel_30_90` | Short-term acceleration: 30-day pace minus 90-day pace. |
| `ret_30_z` | The 30-day move in standard deviations of daily returns - how stretched the last month is. |
| `ret_30_pct4y` | 30-day return as a percentile of the last 4 years. |
| `dd_90d_high` | Drawdown from the highest price of the last 90 days (log). |
| `days_above_20w` | Unbroken run of days above the 20-week average, in 4-year units. |
| `rsi_m_hot_365` | Share of the last year that monthly RSI spent above 70 (0-1) - how long the market has been hot. |

### Exchange flows, fees, addresses — group `flow`, 12 features, from 2010-08-01

| Feature | Meaning |
|---|---|
| `ex_supply_share` | Share of all coins sitting on exchanges (log). A falling share = accumulation into self-custody. |
| `ex_supply_chg_90` | 90-day change in that exchange share. >0 = coins moving onto exchanges. |
| `ex_supply_chg_365` | 1-year change in the exchange share. |
| `ex_supply_pct4y` | Exchange share as a percentile of the last 4 years. |
| `ex_flow_ratio` | Log ratio of 30-day exchange deposits to withdrawals. >0 = more coins arriving than leaving (selling pressure). |
| `ex_inflow_vs_mcap` | 30-day exchange deposits relative to market cap (log). |
| `ex_inflow_pct4y` | That deposit intensity as a percentile of the last 4 years. |
| `ex_inflow_spike` | Log ratio of 7-day to 1-year average deposits - a sudden rush to exchanges. |
| `fee_vs_issuance` | Total transaction fees vs miner issuance (log, 30-day). High = blockspace demand is intense, as in Dec 2017. |
| `fee_per_tx` | Average fee per transaction in BTC (log, 30-day). |
| `fee_pct4y` | Fee-to-issuance ratio as a percentile of the last 4 years. |
| `addr_base_growth` | 1-year growth in the number of addresses holding a balance (log) - adoption. |

### Ether (ETH/BTC, ETH/USD) — group `eth`, 12 features, from 2015-08-08

| Feature | Meaning |
|---|---|
| `ethbtc` | ETH/BTC vs its own 1-year average (log). Rising = risk appetite rotating into altcoins. |
| `ethbtc_ret_90` | 90-day log change in ETH/BTC. |
| `ethbtc_ret_365` | 1-year log change in ETH/BTC. |
| `ethbtc_dd_ath` | ETH/BTC drawdown from its all-time high (log). |
| `ethbtc_pct4y` | ETH/BTC (vs its year average) as a percentile of the last 4 years. |
| `eth_vs_sma200` | ETH price vs its 200-day average (log). |
| `eth_ret_90` | 90-day log return of ETH/USD. |
| `eth_ret_365` | 1-year log return of ETH/USD. |
| `eth_vol_ratio` | Log ratio of ETH to BTC 90-day volatility: how much more violently ETH is moving. |
| `eth_btc_mcap` | ETH market cap vs BTC market cap (log). |
| `eth_btc_mcap_pct4y` | That market-cap ratio as a percentile of the last 4 years. |
| `eth_adr_growth` | 1-year growth in ETH active addresses (log). |

### Macro — group `macro`, 13 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `m2_yoy` | US M2 money supply, 1-year log growth. |
| `fed_bs_yoy` | Fed balance sheet, 1-year log growth. >0 = QE, <0 = QT. |
| `dxy_chg_90` | Broad dollar index, 90-day log change. A rising dollar is usually a headwind. |
| `dxy_chg_365` | Broad dollar index, 1-year log change. |
| `us10y` | US 10-year Treasury yield, percent. |
| `us10y_chg_180` | Change in the 10-year yield over 180 days, percentage points. |
| `fed_funds` | Effective fed funds rate, percent - the policy rate itself. |
| `fed_funds_chg_180` | Change in the policy rate over 180 days: >0 hiking, <0 cutting. |
| `fed_funds_chg_365` | Change in the policy rate over a year. |
| `fed_funds_pct4y` | Policy rate as a percentile of the last 4 years: 1.0 = tightest in a cycle. |
| `real_yield_10y` | US 10-year real (inflation-linked) yield, percent - the opportunity cost of holding a non-yielding asset. |
| `yield_curve` | 10-year minus 2-year Treasury spread. <0 = inverted. |
| `m2_yoy_chg_365` | Change in that M2 growth rate over a year - liquidity accelerating or tightening. |

### Intraday (4-hour bars) — group `intraday`, 4 features, from 2011-08-24

| Feature | Meaning |
|---|---|
| `intraday_vol_30` | 30-day average of volatility measured within each day from 4-hour bars, annualised. |
| `intraday_vs_close_vol` | Log ratio of intraday to close-to-close volatility. |
| `rsi_4h_max_7d` | Highest 4-hour RSI of the past week. |
| `rsi_4h_min_7d` | Lowest 4-hour RSI of the past week. |

### Calendar (ablation only) — group `calendar`, 4 features, from 2010-07-18

| Feature | Meaning |
|---|---|
| `month_sin` | Sine of the month-of-year - included only to test whether seasonality helps (it does not). |
| `month_cos` | Cosine of the month-of-year. |
| `week_sin` | Sine of the week-of-year. |
| `week_cos` | Cosine of the week-of-year. |

### Sentiment (display only) — group `sentiment`, 1 features, from 2018-02-10

| Feature | Meaning |
|---|---|
| `fear_greed_30` | Crypto Fear & Greed index, 30-day average (2018+). Not in any training set - history is too short. |
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
| **De-clustering, cooldown 120d / 0.05** | **0.568 ± 0.050** (10 seeds) | **adopted** |
| De-clustering, cooldown 60d / 0.05 | 0.526 ± 0.017 (6 seeds) | also good (rule is not knife-edge) |
| De-clustering, cooldown 30d / 0.05 | 0.656 ± 0.059 (10 seeds) | worse than 120d, still beats no rule |
| Adaptive window, anchored on the opposite extreme | 2.194 ± 0.026 (10 seeds) | fails: fires 144 times in 2017 |
| Adaptive window, anchored on the same-side extreme (margin 30 / 120d) | 2.240 (10 seeds, every seed) | fails: fires never in E4 |
| Trailing window 730 / 900 / 1000 / **1095** / 1200 / 1300 / 1460 days | 1.894 / 1.489 / 0.586 / **0.495** / 1.393 / 2.240 / 2.240 | current value already best |
| Signal budget x0.5 (5 days/cycle) / x1 / x2 (20) | 0.661 / **0.495** / 0.537 | current value already best |

### Why the adaptive window failed (and what the fixed one is really doing)

Making the window adapt to the cycle - the obvious fix for the fragility below - was tried twice and both
versions collapsed, in opposite directions:

- **Anchored on the last confirmed opposite extreme** ("this leg so far"): early in a bull leg the window is
  only a few hundred days of rising prices, so almost every day ranks as extreme. It fired 144 strength-10
  sells in 2017, at a median **95% below** the eventual top.
- **Anchored just before the last confirmed same-side extreme**: the window then still contains the previous
  top, whose score nothing can out-rank, so it fired **nothing at all** in the current cycle (2.240 = the
  no-signal penalty, on every seed and at both margins tried).

Those two failures bracket what the fixed 1095-day window is quietly doing: mid-cycle it still reaches back
over the previous top, which suppresses premature sells, and roughly three years later that old top drops
out just as the next one approaches, allowing a genuine extreme to rank. That only works because tops are
about four years apart, so the fragility is real - but no anchor derived from the confirmed extremes
reproduces it, because the useful window boundary sits *between* the two extremes rather than on either.

### The trailing window is the fragile part

Strength comes from ranking today's score against the model's own scores over the trailing window, and only
~1000-1100 days works (`strength_mode: adaptive` exists and is configurable, but scores far worse - see above). Too long and the window still contains the **previous same-side extreme**, whose
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
- **Judge a change on 10 seeds where it matters.** The baseline's 6-seed mean was 0.511; a 10-seed run
  found a bad seed at 1.01 and moved the mean to 0.568 ± 0.050. Tail seeds matter as much as the average.
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
