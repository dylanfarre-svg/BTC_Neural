# BTC_Neural: Project Plan

A model that outputs a daily **buy or sell strength from 0 to 10** for long-horizon (3+ year) Bitcoin accumulation and distribution. The targets are about **10 strength-10 buy signals near each cycle low** and **10 strength-10 sell signals near each cycle high**. The project includes walk-forward (cycle-by-cycle) backtesting and a GUI.

> This is a research tool, not financial advice. Its outputs are only as good as four historical cycles can make them. That is the central constraint, and much of this plan is built around it.

## 0. Decisions (confirmed 2026-09-18)

| Topic | Decision |
|---|---|
| Success criterion | **Signals land near the cycle extremes.** Signal-quality metrics decide which model wins. Portfolio simulation is secondary, for information only |
| Sells | **Real, taxable trims** of holdings. The simulator models tax lots, a holding-period rule and a core position floor |
| Signal budget | **10 days at strength 10** per cycle for sells, and the same for buys (daily evaluation) |
| Scale | One signed scale: **−10 (max sell) … +10 (max buy)** |
| Data | **Free sources only** for now. The code is written so a paid feed can be added later |
| Model choice | The **best walk-forward model produces the live signal**. The neural net is always trained and reported as a **reference track**, to improve over time |
| GUI | Local **Streamlit** app |
| Storage | **SQLite** (a single file, `data/btc_neural.db`) for all tabular data. Model weights stay as files |
| Live | **Daily update + notifications** (see §8a) |

---

## 1. The core problem, and how it shapes everything

Bitcoin has had about **4 complete cycles**:

| Cycle (low → low) | Low | Top | Next low | Halving inside |
|---|---|---|---|---|
| C1 | 2011‑11 (~$2) | 2013‑12 (~$1,150) | 2015‑01 (~$170) | 2012‑11‑28 |
| C2 | 2015‑01 | 2017‑12 (~$19.7k) | 2018‑12 (~$3.2k) | 2016‑07‑09 |
| C3 | 2018‑12 | 2021‑11 (~$69k) | 2022‑11 (~$15.5k) | 2020‑05‑11 |
| C4 | 2022‑11 | 2025‑10 (~$126k) | *not yet confirmed* | 2024‑04‑20 |

That gives us **about 8 turning points to learn from**. Even with ~5,500 daily rows, the number of *independent* examples of "this is a top" is tiny. The consequences:

1. **A large deep net will memorize the past, not learn a pattern.** It will learn "December 2017 = sell", and walk-forward testing will expose this. The plan still includes a neural net as you asked, but it is small and heavily regularized, and it has to beat simpler baselines to earn its place.
2. **Label and feature design matter far more than architecture.** Most of the effort goes into scale-free, causal features and a well-defined target.
3. **Amplitudes shrink every cycle.** From low to top, the moves were ~500x, ~115x, ~21x, then ~8x. On-chain extremes also peak lower each time (for example, MVRV-Z reached ~10+ in 2013/2017, ~7.5 in 2021 and much less in 2025). Features that use fixed thresholds decay. Features must be normalized so the model can extrapolate to weaker cycles.
4. **Report results per cycle, never as one pooled number.** One lucky cycle can hide two bad ones.

---

## 2. Target (label) design (the most important decision)

Hand-labeling "these 10 days are the top" gives very few positives and a jagged target. I recommend a **continuous, forward-looking score** that is computed automatically:

### Primary label: forward upside vs. downside balance
For each day *t*, with horizon *H* (default 730 days, tunable from 365 to 1095):

```
U = max(close[t .. t+H]) / close[t]       # best future upside
D = close[t] / min(close[t .. t+H])       # worst future downside
y = (ln U − ln D) / (ln U + ln D)         # in [-1, +1]
```

- `y ≈ +1`: price almost only goes up from here, so this is a **buy** zone (a cycle low).
- `y ≈ −1`: price almost only goes down from here, so this is a **sell** zone (a cycle top).
- It needs no hand-picked cycle dates, it is smooth, and it matches a 3+ year holding horizon directly.

### Alternative labels, to compare in ablations
- **Cycle-position label:** `log(P / prev_low) / log(next_high / prev_low)` using the known cycle extremes.
- **Proximity classification:** within ±N days *and* within X% of a cycle extreme → class `top`, `bottom` or `neither`.

### Leakage note
The label looks up to *H* days into the future. In walk-forward testing, **labels from the last H days before a test cycle would see into that test cycle**. Those rows must be **purged** (masked), with an extra embargo gap. The newest *H* days of data have no complete label and are never used for training.

---

## 3. From model score to a 0–10 strength (hitting the "10 per cycle" budget)

The model outputs a continuous score *s* in [−1, +1]. It is turned into strength like this:

1. On the **training** cycles only, compute the empirical distribution of *s*.
2. Choose thresholds so that each cycle averages the target number of signals at each level. For example:
   - Strength 10 = the most extreme ~10 days per cycle (~0.7% of days)
   - Strength 8–9 = the next ~30 days per cycle
   - …down to 0 = neutral
3. Apply those **frozen** thresholds to the test cycle. Never recalibrate on test data.
4. Optional **de-clustering**: a cooldown of N days or a minimum price move between strength-10 signals. This spreads the 10 signals across the top zone so they don't all land in one week.

This design makes the "10 per cycle" goal something you **control directly**, not something you hope the model produces. Whether those 10 signals land *near* the extremes is what the backtest measures.

Output format: one **signed** scale from −10 (max sell) to +10 (max buy), shown in the GUI as "Sell 7" or "Buy 10".

---

## 4. Inputs (features)

**Rules for every feature:**
- **Causal:** a feature at day *t* uses only data up to *t*. An automated test enforces this (§8).
- **Scale-free:** use log ratios, z-scores and percentiles. No raw prices, because $60k means nothing without context.
- **Available early enough.** Features that start in 2018 or later cannot train fold 1, so features are grouped into tiers (§4.6).

### 4.1 Your proposed inputs, with recommendations

| Input | Keep? | Notes |
|---|---|---|
| Full 1W / 1D candles | ✅ | Used as derived features and as a short weekly window for the sequence model |
| 4h candles | ⚠️ optional | Mostly noise for a 3-year horizon, and 4h history before ~2014 is poor. Use only for derived features (e.g., 4h RSI extremes, intraday volatility), or drop |
| Current price | ✅ as ratios | Never raw. Use price / 200-week MA, price / realized price, and similar |
| Current date | ❌ as raw input | A model memorizes dates. Use cycle-relative time instead |
| Daily / weekly / monthly RSI | ✅ | Monthly RSI is a strong cycle-top feature. Add RSI divergence flags |
| Day of week | ❌ | No relevance to a 3-year horizon. Pure noise, so an overfitting risk |
| Week / month of year | ⚠️ ablation only | With 4 cycles the model can learn "December = top". Test with and without |
| Days since halving | ✅ | A strong and meaningful clock. Tops came ~370 / 526 / 549 / 535 days after halving |
| Days since cycle low / top (1 and 2 back) | ✅ **but causal** | See 4.2. The naive version leaks future information |

### 4.2 The "days since cycle low/top" leak (important)
At the time, you don't know a low was *the* cycle low. That becomes clear only months later. Using hindsight dates makes the backtest look like magic, and it will fail live. Causal replacements:
- **Days since all-time high** and **drawdown from ATH**. Both are fully causal and very powerful.
- **Days since the 2-year rolling minimum**, and the rise above it.
- **Confirmed-extreme rule:** a low counts as "confirmed" once price is ≥ X% above it (e.g., 100%) *and* ≥ Y days have passed. Days-since is counted only from the confirmation date. The same rule, mirrored, applies to tops.

### 4.3 Additional price and technical features (free, full history)
- **Mayer Multiple:** price / 200-day MA
- **Price / 200-week MA**, historically close to the floor at cycle lows
- **Pi Cycle Top indicator:** the gap between the 111-day MA and 2 × the 350-day MA
- **2-Year MA Multiplier:** price relative to the 2-year MA and to 5 × the 2-year MA
- **Log-regression residual:** distance from a power-law or log-growth fit, **refit on an expanding window** so it stays causal
- **Weekly MACD, Bollinger Band width, realized volatility (30/90/365-day)**
- **Rate-of-change** over 30, 90, 180 and 365 days
- **Drawdown from ATH**, **days since ATH**, **% above 2-year low**

### 4.4 On-chain features (high value for cycle extremes)
- **MVRV and MVRV Z-score** (market value / realized value). One of the best historical cycle indicators.
- **NUPL** (net unrealized profit/loss)
- **SOPR / aSOPR / LTH-SOPR**
- **Puell Multiple** (miner revenue / its 365-day average)
- **Realized price** and **price / realized price**
- **Reserve Risk, RHODL ratio, Long-Term Holder supply**
- **Hash rate and Hash Ribbons** (miner capitulation, which often marks lows)
- **Active addresses, transaction value** (network growth)

### 4.5 Macro and sentiment features (partial history)
- **Global M2 / Fed net liquidity, DXY, US 10-year yield** (FRED, full history)
- **Fear & Greed Index** (2018+)
- **Funding rates and open interest** (≈2019+)
- **Google Trends "bitcoin"** (full history, but pulling the data is unreliable)
- **Spot ETF flows** (2024+ only; for later use)

### 4.6 Feature tiers
- **Tier A (2011+):** price, technicals, halving clock and most on-chain metrics. Trains every fold.
- **Tier B (2015+ / 2018+):** Fear & Greed, funding rates, some on-chain detail. Only for later folds, or a separate model.
- **Tier C (2024+):** ETF data. Monitoring only; too short to train on.

### 4.7 Normalizing for shrinking cycles
Where possible, also include each feature as a **percentile of its expanding history** or **relative to the previous cycle's extreme** (computed causally). This lets "MVRV-Z = 3" mean "hot for this era" and not "lukewarm by 2017 standards."

---

## 5. Models, from simplest to deepest

Build the baselines **first**. The neural net must beat them in walk-forward tests to be used.

1. **Rule-based composite (baseline 0):** the average of z-scored Mayer Multiple, MVRV-Z, Pi Cycle gap, monthly RSI and price/200-week MA, with no training. This is surprisingly hard to beat.
2. **Regularized linear / logistic model (baseline 1):** interpretable and hard to overfit.
3. **Gradient-boosted trees (LightGBM):** with **monotonic constraints**, e.g., higher MVRV can only push toward "sell". This greatly improves robustness on small data.
4. **Deep neural net (the requested model):**
   - Input: the current feature vector plus a **weekly window of the last ~52–104 weeks** of key features.
   - Architecture: a small **temporal CNN or GRU** encoder with an MLP head. Roughly 10k–50k parameters, not millions.
   - Heavy regularization: dropout, weight decay, early stopping on an inner validation split, noise injection and feature dropout.
   - **Ensemble of 10–20 seeds.** The spread between seeds gives a confidence estimate, shown in the GUI.
5. **Final ensemble:** a weighted blend of the models that pass the walk-forward tests.

**Selection policy:** the model (or ensemble) with the best **selection score** (§6) becomes the **production** model and drives the live signal and notifications. The neural net is **always** trained and evaluated as a **reference track** with every run, even when it loses. The GUI shows its signal next to production, so you can see how it improves over time.

Training is small. CPU or Apple Silicon (PyTorch MPS) is plenty.

---

## 6. Walk-forward backtesting

### Folds (expanding window)
| Fold | Train | Test |
|---|---|---|
| 1 | C1 (2011‑11 → 2015‑01) | C2 (2015‑01 → 2018‑12) |
| 2 | C1–C2 | C3 (2018‑12 → 2022‑11) |
| 3 | C1–C3 | C4 (2022‑11 → today) |

- Pre-2011 data is used only for warm-up (moving averages and similar).
- Each fold **purges** training labels that look into the test period, plus an embargo.
- Hyperparameters are tuned **only on inner splits of the training cycles**.
- **Discipline rule:** each time you view fold results and change something, the test cycles become a bit more like training data. Keep a log of experiments. Once a design is frozen, the honest test is forward paper-trading of the live signal.

### Trading simulation
Configurable, with sensible defaults:
- **Cash flow:** a fixed monthly contribution, e.g., $1,000 per month.
- **Buy rule:** at strength *k*, deploy a fraction of the cash reserve, f(k), rising steeply near 10. Unused cash accumulates.
- **Sell rule:** at strength *k*, sell a fraction of holdings, g(k), always keeping a **core position floor** (e.g., never below 50% of peak BTC held).
- Fees, slippage and an optional capital-gains tax model.

### Benchmarks
- Lump-sum HODL
- Plain monthly DCA
- DCA plus a simple rule (buy 2× below the 200-week MA, sell 10% above a Mayer Multiple of 2.4)
- Baseline 0 (the rule composite)

### Metrics (per cycle and per fold)
- **Signal quality (primary, used to choose the model):**
  - **Price error:** for strength-10 sells, the median % below the cycle-top price. For strength-10 buys, the median % above the cycle-low price. This is measured in **log terms**, so cycles of different sizes are comparable.
  - **Timing error:** the median number of days from the extreme.
  - **Hit rate:** the % of strength ≥ 9 signals within ±60 days *and* within 25% of the extreme.
  - **Wrong-side penalty:** a sell signal given in the bottom half of the cycle's log price range, or a buy signal in the top half. These are the costly mistakes.
- **Selection score:** one pre-declared number, the **worst fold's** combined signal-quality score. Using the worst fold, not the average, rewards models that work in *every* cycle, not just one lucky one. The score is fixed *before* comparing models, so the rule can't be changed to suit a favored result.
- **Signal budget:** the count of strength-10 signals per cycle vs. the target of 10.
- **Portfolio (secondary):** final BTC held, final USD value, **after-tax** proceeds, average buy cost vs. DCA, average sell price vs. the cycle top, and max drawdown.
- **Stability:** results by seed and by fold, with bootstrapped confidence intervals.

### Tax-aware sell simulation
Sells are real trims, so the simulator tracks **individual purchase lots**:
- **Lot selection:** configurable. FIFO, or HIFO (sell the highest-cost lots first, which usually reduces tax).
- **Tax rates:** short-term and long-term rates, plus the holding period for long-term treatment (e.g., 1 year in the US, 0 in some jurisdictions). All configurable.
- **Option:** prefer selling lots already past the long-term holding threshold.
- **Core floor:** never sell below X% of peak BTC held.

---

## 7. GUI

**Recommendation: Streamlit plus Plotly.** It is pure Python, fast to build, and runs locally in the browser.

Pages:
1. **Dashboard / Live signal:** today's signed strength (−10…+10), model agreement or confidence, top contributing features (SHAP), and a log-price chart with recent signals.
2. **Data:** source status, last update, a "refresh data" button, and gap and quality checks.
3. **Feature explorer:** plot any feature against log price with cycle shading.
4. **Train:** pick the model type, feature tier, label horizon *H* and hyperparameters. Run training with progress and save a versioned run.
5. **Backtest:** pick a run and a fold. Show a candle chart with buy/sell markers colored by strength, equity curves vs. benchmarks, a metrics table and a per-cycle breakdown.
6. **Compare runs:** runs side by side, to answer "did that change actually help?"

---

## 8. Engineering

### Stack
Python 3.12 · `uv` · pandas · **SQLite** (stdlib `sqlite3` + pandas) · PyTorch · LightGBM · scikit-learn · SHAP · Streamlit · Plotly · pytest · `requests` · `launchd` for scheduling

### Storage: SQLite (`data/btc_neural.db`)
The data is small (~5.5k daily rows, ~30k 4h rows and ~100 features), so one SQLite file holds everything tabular. That replaces the Parquet/DuckDB idea. Model weights (`.pt`, `.txt`) stay as files under `runs/`, and the database stores their paths.

| Table | Purpose |
|---|---|
| `candles(timeframe, ts, open, high, low, close, volume, source)` | 4h / 1D / 1W OHLCV. PK `(timeframe, ts)` |
| `series(name, date, value, source, fetched_at)` | On-chain, macro and sentiment series, in long format |
| `series_meta(name, source, tier, first_date, description)` | Feature catalog and tier info |
| `ingest_log(source, started_at, finished_at, rows, status, error)` | Data freshness and failures |
| `feature_cache(feature_set_hash, date, name, value)` | Optional cache. Features can always be recomputed |
| `runs(run_id, created_at, model_type, role, config_json, feature_set, label_cfg, code_version, weights_path)` | `role` is `production`, `reference_nn` or `experiment` |
| `fold_results(run_id, fold, metric, value)` | Walk-forward metrics |
| `predictions(run_id, fold, date, score, strength)` | Out-of-sample backtest predictions |
| `live_signals(date, run_id, role, score, strength, confidence, price, created_at)` | **Append-only** daily signal log |
| `notifications(id, created_at, channel, kind, message, status, error)` | What was sent, and whether delivery worked |

**Point-in-time integrity:** free on-chain providers sometimes **revise past values**. `live_signals` is append-only and never rewritten, and `series.fetched_at` records when each value was seen. Together they make the forward paper-trading record auditable, and it becomes the most honest test the project has. Useful timing: as of Sep 2026, cycle C4 is past its Oct 2025 top, so the live log may capture this cycle's low as it happens.

### Proposed layout
```
BTC_Neural/
├── pyproject.toml
├── config/
│   ├── cycles.yaml          # halving dates, known extremes (labels/eval only!)
│   └── default.yaml         # horizons, signal budget, sim params
├── data/                    # gitignored: btc_neural.db (SQLite)
├── src/btc_neural/
│   ├── ingest/              # price (Bitstamp/Kraken/Binance/CoinMetrics), onchain, macro
│   ├── features/            # technicals, onchain, cycle_clock, normalization
│   ├── labels/              # forward-balance, cycle-position, proximity
│   ├── models/              # baseline_rules, linear, lgbm, nn (tcn/gru), ensemble
│   ├── calibrate/           # score → 0–10 strength, declustering
│   ├── backtest/            # walk-forward folds, purging, simulator, metrics, benchmarks
│   └── app/                 # Streamlit pages
├── tests/
│   ├── test_no_lookahead.py # key test (see below)
│   └── ...
└── runs/                    # versioned model runs + reports
```

### The key test: no lookahead
For random dates *t*, compute every feature using **only data up to *t***. Assert it equals the value from the full-history pipeline at *t*. This catches the most common way projects like this fool their creators, whether through centered windows, full-history normalization, hindsight cycle dates or backfilled on-chain data.

### Data sources (free first)
- **Price, daily from 2010:** Coin Metrics Community API (`PriceUSD`) and Bitstamp OHLC (2011+).
- **4h / weekly OHLC:** Bitstamp and Kraken (2013+), Binance (2017+). Weekly bars are resampled from daily.
- **On-chain:** Coin Metrics Community (market and realized cap, hash rate, addresses) and BGeometrics / bitcoin-data.com (MVRV-Z, NUPL, SOPR, Puell). Glassnode is the paid, higher-quality option.
- **Macro:** FRED (M2, DXY, yields, Fed balance sheet).
- **Sentiment:** alternative.me Fear & Greed.
- *Which metrics the free tiers include changes over time. Verify each one in Phase 1.*

---

## 8a. Daily update and notifications

### Scheduling: macOS `launchd` (LaunchAgent)
- Runs daily at **00:30 UTC**, after the daily candle closes and the free sources update. Some on-chain data lags by a day, so the job retries at 06:00 UTC if anything is stale.
- `launchd` is better than `cron` on a Mac: if the Mac was asleep at run time, **the job runs on wake**, not skipped.
- Pipeline: `ingest → validate freshness → features → predict (production + reference NN) → calibrate → append live_signals → notify`.

### Channel: **Telegram bot (recommended)**, with ntfy as a simpler alternative

| Option | Cost | Phone push | Sends chart image | Setup | Notes |
|---|---|---|---|---|---|
| **Telegram bot** ✅ | Free | Yes | Yes | ~5 min (BotFather → token + chat id) | Private chat, rich formatting, can add commands later (`/signal`, `/chart`) |
| ntfy.sh | Free | Yes | Attachments | ~1 min, no account | Public topics are readable by anyone who guesses the name. Use a long random topic, or self-host |
| Pushover | One-time app fee | Yes | Yes | Easy | Very reliable, but not free |
| Email (SMTP) | Free | Slow | Yes | App password needed | Fine for a weekly digest, poor for alerts |
| macOS notification | Free | No | No | None | Only seen when you're at the Mac. Good as a secondary channel |

**Recommendation: Telegram as the main channel, plus a local macOS notification.** The bot token and chat id live in a local `.env` file, which is gitignored and never committed.

### What triggers a message (avoid alert fatigue)
The model gives ~20 strength-10 days per 4-year cycle, so most days are unremarkable. Messages go out only when useful:

| Kind | Trigger |
|---|---|
| 🚨 **Strength-10** | Any day at ±10. Always sent, with a chart image |
| ⚠️ **Strong signal** | \|strength\| ≥ 7 (configurable) |
| ↕️ **Regime change** | Strength moves ≥ 3 levels in 7 days, or changes sign between buy and sell zones |
| 🧪 **NN disagreement** | The reference NN differs from production by ≥ 5 levels. Informational only |
| 📅 **Weekly digest** | Sundays: current strength, 7-day trend, top drivers and data health |
| ❌ **Pipeline failure / stale data** | Ingest error, or any feature source older than N days. **This one matters:** a silent failure could hide a strength-10 day |

Each message includes: date, price, signed strength, confidence (how much the models agree), the top 3 driving features, and a link to the local Streamlit dashboard.

---

## 9. Phased roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0. Setup** | repo, uv env, config, CI for tests | `pytest` green |
| **1. Data** | ingestion into SQLite + validation, cycle/halving table | Continuous daily series 2010–today, gap report |
| **2. Features** | Tier A features + no-lookahead test | Test passes; feature explorer plots look sane |
| **3. Labels + calibration** | forward-balance label, strength mapping, purging | Label plot matches intuition across all cycles |
| **4. Backtester + baselines** | folds, simulator, metrics, benchmarks, baseline 0/1 | Full walk-forward report for baselines |
| **5. Models** | LightGBM (monotone) + small TCN/GRU ensemble | Report vs. baselines, per fold |
| **6. GUI** | Streamlit app, all pages | Live signal + backtest viewer work end to end |
| **7. Live** | launchd job, append-only signal log, Telegram + macOS notifications | A test notification arrives on your phone. Signals are logged daily |
| **8. Iterate** | Tier B features, label/horizon ablations, calendar-feature ablation | Documented experiment log |

---

## 10. Honest expectations

- The model will probably recognize **"we are in a top zone / bottom zone"** well, because these regimes look very distinctive on-chain. It will **not** reliably pick the exact day. Aim for "within ~15–25% and a few months of the extreme."
- The biggest risk is a **regime change**: ETFs, institutional flows or a lengthening or broken 4-year cycle. The halving-clock features are the most exposed. Ablations with and without them will show how much the model relies on the clock.
- If the neural net doesn't beat the rule composite and LightGBM out of sample, that is a valid result. The framework still delivers a well-tested signal.

---

## 11. Remaining open questions

The main decisions are recorded in §0. These are still open and have working defaults:
1. **Tax settings:** which jurisdiction's rules to model by default (holding period and rates)? Default: 1-year long-term threshold, rates set by the user in config.
2. **Core floor:** what minimum % of peak holdings should never be sold? Default: 50%.
3. **Notification channel:** Telegram (recommended) or ntfy?
