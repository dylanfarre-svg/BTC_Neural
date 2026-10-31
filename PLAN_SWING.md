# BTC_Neural: Swing Signal Plan (7–28 day holds)

A second signal alongside the cycle model: a **discrete trade proposal** — enter, target, stop,
deadline — with an intended holding period of **7 to 28 days** and a target move of **more than 5%**.
Designed so that a third signal (4 hours to 4 days) slots in later as configuration rather than a rewrite.

> Research tool, not financial advice. Nothing here is a tax opinion; §8 flags where a practitioner's
> input is needed before the numbers mean anything.

**Status: plan only. Nothing is built. §14 lists the open questions to settle first.**

---

## 0. Summary of recommendations

| Question | Recommendation |
|---|---|
| Same app or separate UI? | **Same repo, same Streamlit app**, new page group with its own vocabulary. Split only the *execution* concern later, never the display |
| Same package? | New `src/btc_neural/trade/`, parameterised by a `Horizon` spec. Cycle code is left untouched |
| Target width | **Not 5% — 5% is the floor.** A 7–28 day hold implies a ~10% barrier (§2). Use vol-scaled with a 5% floor |
| Label | Triple barrier resolved on 4h bars, with a fixed-horizon variant as the ablation |
| Validation | Purged + embargoed walk-forward, ~45 test blocks, judged against buy-and-hold **and** a random-entry bootstrap |
| Direction | Long/flat to start. Shorting is a separate decision with its own venue and tax consequences |
| First build | A rules baseline that must be beaten, exactly as `rules` does for the cycle model |
| Deployment | Paper-trade through the append-only log for an agreed period before any real money |

---

## 1. Why this is not the cycle model with smaller numbers

The two problems differ in almost every dimension that drives design:

| | Cycle signal (built) | Swing signal (this plan) |
|---|---|---|
| Independent events | ~8 turning points in 15 years | ~300–600 non-overlapping trades |
| Label | Position within a low→top leg, hindsight-anchored | Forward path outcome, computed per bar |
| Statistical problem | Desperately few examples; overfitting is the enemy | Plenty of examples; **a tiny edge against a large benchmark** is the enemy |
| Folds | 4 halving epochs | ~45 purged time blocks |
| Metric | Distance from the true extreme | Expectancy per trade, net of costs and tax |
| Cost sensitivity | Negligible (a 0.5% fee against a 10× move) | **Decisive** (a 0.2–1.0% round trip against a 10% move) |
| Output | Continuous strength −10…+10 | Discrete proposal with entry, target, stop, deadline |
| What "wrong" looks like | Sells 40% below the top | Twenty small losses that quietly underperform holding |

The consequence: **almost none of the modelling code transfers, but almost all of the plumbing does.**
That asymmetry is what §3 is built around.

---

## 2. Reality check: the spec as written is internally inconsistent

Measured on this repo's own data before designing anything.

### 2a. BTC volatility by era

| Era | Annualised vol | 14-day σ | 28-day σ |
|---|---|---|---|
| 2011+ | 85.8% | 16.8% | 23.8% |
| 2015+ | 66.7% | 13.1% | 18.5% |
| 2019+ | 61.3% | 12.0% | 17.0% |
| 2023+ | 45.7% | **9.0%** | 12.7% |

A 5% move is roughly **0.4–0.6 standard deviations** over two weeks. It is not a rare event; it is the
market breathing.

### 2b. How long a ±5% barrier actually takes to hit

First touch of either barrier, daily bars:

| Era | Up first | Down first | Median days | Within 7 days |
|---|---|---|---|---|
| 2015+ | 53.2% | 44.0% | **3.0** | 78.5% |
| 2019+ | 52.3% | 45.7% | **3.0** | 78.4% |
| 2023+ | 52.7% | 45.2% | **5.0** | 69.3% |

**Three-quarters of ±5% trades are over inside a week.** A 5% take-profit does not produce a 7–28 day
hold; it produces a 2–5 day hold. As specified, the swing signal would land squarely on top of the
intraday signal you want next.

### 2c. The barrier width that does give a 7–28 day hold

Time to first touch, by barrier width:

| Band | Era | Median days | Resolves in 7–28d | Resolves <7d |
|---|---|---|---|---|
| 3% | 2015+ / 2023+ | 2 / 2 | 7.5% / 8.9% | 92% / 91% |
| 5% | 2015+ / 2023+ | 3 / 5 | 24% / 35% | 74% / 63% |
| **8%** | 2015+ / 2023+ | **7 / 10** | 43% / 54% | 48% / 32% |
| **10%** | 2015+ / 2023+ | **10 / 15** | 47% / 54% | 36% / 20% |
| 12% | 2015+ / 2023+ | 12 / 19 | 48% / 48% | 28% / 12% |
| 20% | 2015+ / 2023+ | 21 / 31 | 39% / 22% | 10% / 3% |

**A 7–28 day hold means a barrier of roughly 8–12%, and it has to be vol-scaled** — 8% was a 7-day trade
in 2015 and is a 10-day trade now. The same table says the future intraday horizon (4h–4d) is a **2–3%**
band. The two horizons separate cleanly at those widths and collide at 5%.

### 2d. The base rate you have to beat

If instead the exit is a scheduled date (fixed-horizon formulation), the unconditional odds are:

| Era | H | P(return > +5%) | P(return < −5%) | Mean return | P(>0) |
|---|---|---|---|---|---|
| 2015+ | 14 | 35.9% | 24.5% | +2.78% | 56.2% |
| 2019+ | 14 | 33.6% | 23.7% | +2.26% | 54.7% |
| 2023+ | 14 | 30.3% | 19.9% | +1.98% | 54.9% |
| 2023+ | 28 | 39.8% | 23.9% | +3.83% | 57.7% |

Two things follow, and both are uncomfortable:

1. **A coin flip already "works".** Mean 14-day return is positive (+2.0–2.8%) because BTC drifts up.
   A random long with a 5% target and 5% stop will show a profit from drift alone. Any claimed edge must
   be measured against a **random-entry bootstrap with the same trade count and holding period**, not
   against zero.
2. **Buy-and-hold returned 47–59% a year** over every era measured. A long/flat strategy that is in cash
   half the time starts ~25% a year behind. Beating *zero* is easy; beating *holding* is the real test,
   and it is the only comparison that matters given the cycle model already tells you to accumulate.

This does not mean the signal is a bad idea — a swing overlay that is *right about when not to hold*
has real value, and the tax-free-ish compounding argument differs from the cycle strategy. It means the
success criterion must be set honestly up front (§9, §12), before any model is fitted.

---

## 3. Architecture: same app, same repo, separate namespace

### 3a. The recommendation

**One repo. One Streamlit app. A new top-level package `src/btc_neural/trade/`.**

What is genuinely shared, and should never be duplicated:

- `db.py`, `ingest/`, `data.py` — the DB already carries **33,316 four-hour bars back to 2011-08** and
  5,949 daily bars. The intraday horizon's data is already ingested and backfilled nightly.
- `models/` — the `Model` API (`fit`/`predict`/`explain`/`save`) is horizon-agnostic and fine as is.
- `features/` — the *machinery* (group registry, `columns_for`, the no-lookahead test harness) transfers
  completely. The *features* mostly do not (§6).
- `notify.py` — alert dispatch, de-duplication, the ntfy/macOS channels.
- `app/common.py` — the colour system, hover-spine layout, chart helpers.
- `config.py`, `cli.py` — one config file, one `btcn` command.

What must not be shared, because they encode the cycle assumption:

`labels.py` · `backtest/walkforward.py` (halving-epoch folds) · `backtest/metrics.py`
(distance-to-extreme) · `calibrate.py` (days-per-cycle signal budget) · `backtest/simulator.py`
(DCA accumulate-and-trim).

### 3b. Why not a separate Streamlit app

- `load_dataset()` is a `@st.cache_resource` build of the full feature frame. Two apps means two builds
  of the same data, two caches, two ports, two things to restart.
- `notify.dashboard_url` is a single link in every push notification. Two apps means choosing which one
  the phone opens.
- The cycle context is *useful next to* a trade proposal — "enter long, and by the way the cycle model
  reads Sell 7" is exactly the information you want on one screen.
- `common.py` would be forked within a month.

### 3c. Why not a separate repo

The data layer and ingest are the bulk of the existing value and are identical. A separate repo means
either a shared SQLite file across two checkouts (fragile) or duplicated ingest (worse).

### 3d. What *should* be separated, and when

**Execution, not display.** If a swing signal ever places real orders, the component holding an exchange
API key should be its own process with its own credentials, its own restricted config, and no import of
the Streamlit app. That is a later decision (§14 Q10) and does not change the layout now.

### 3e. Keeping the two signals from being confused

This is a safety concern, not a cosmetic one: "BUY 8" (accumulate, 3-year horizon) and "ENTER LONG"
(14-day trade) must never look alike.

- Convert the app from the `pages/` directory convention to **`st.navigation` with sections** —
  "Cycle" and "Swing" as separate labelled groups. Supported in the pinned Streamlit 1.64; a contained,
  low-risk change.
- Different vocabulary throughout: the cycle model has *strength*; the swing model has *trades*,
  *conviction* and *state* (flat / long / cooling off). No −10…+10 scale on the swing pages.
- Different visual encoding: the cycle model owns the blue/red diverging ramp. The swing pages use entry
  and exit markers on price plus an equity curve, and leave the ramp alone.

### 3f. Proposed layout

```
src/btc_neural/
  trade/
    __init__.py
    horizons.py        # Horizon specs: SWING, later INTRADAY
    labels.py          # triple barrier (4h-resolved) + fixed-horizon variant
    features.py        # short-horizon feature groups, registered like the cycle ones
    folds.py           # purged + embargoed walk-forward block generator
    metrics.py         # expectancy, hit rate, MAE/MFE, profit factor, vs-benchmark tests
    simulator.py       # discrete round trips: costs, slippage, tax lots, position state
    calibrate.py       # score -> take / don't take, plus the re-entry cooldown
    pipeline.py        # train / save / load per horizon, mirroring the cycle pipeline
    live.py            # position state machine, trade log, alerts
  app/
    pages/  (existing cycle pages, unchanged)
    trade_pages/       # Swing overview, Trade blotter, Research
```

`Horizon` is a frozen dataclass holding: bar timeframe (`1d` / `4h`), resolution timeframe for paths
(`4h`), min/max hold, barrier rule, cost model, feature groups, fold block length, purge and embargo,
live cadence, and alert kinds. `SWING` and `INTRADAY` are two instances. Everything downstream takes a
`Horizon` argument — the seam you asked for, created once rather than retrofitted.

### 3g. Database: add tables, never alter them

`db.connect()` runs `executescript(SCHEMA)` with `CREATE TABLE IF NOT EXISTS`. **There is no migration
path: adding a column to an existing table silently does nothing on your existing database.** So the
swing signal adds new tables and touches none of the current ones.

```sql
CREATE TABLE trade_runs (            -- mirrors `runs`, with horizon + cost assumptions
    run_id TEXT PRIMARY KEY, created_at TEXT, horizon TEXT, model_type TEXT,
    feature_set TEXT, role TEXT, config_json TEXT, selection_metric REAL,
    weights_path TEXT, notes TEXT);
CREATE TABLE trade_folds (run_id TEXT, block TEXT, metric TEXT, value REAL, PRIMARY KEY (run_id, block, metric));
CREATE TABLE trade_backtest (        -- one row per simulated trade
    run_id TEXT, trade_id TEXT, side TEXT, entry_date TEXT, entry_price REAL,
    exit_date TEXT, exit_price REAL, exit_reason TEXT, gross_ret REAL, net_ret REAL,
    hold_days REAL, mae REAL, mfe REAL, PRIMARY KEY (run_id, trade_id));
CREATE TABLE trade_positions (       -- live state machine
    trade_id TEXT PRIMARY KEY, horizon TEXT, run_id TEXT, side TEXT, state TEXT,
    opened_at TEXT, entry_price REAL, target_price REAL, stop_price REAL, deadline TEXT,
    closed_at TEXT, exit_price REAL, exit_reason TEXT, size_fraction REAL);
CREATE TABLE trade_signals (         -- append-only, one row per horizon per bar
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, horizon TEXT, run_id TEXT,
    score REAL, conviction REAL, action TEXT, price REAL, drivers_json TEXT, created_at TEXT,
    UNIQUE (ts, horizon, run_id));
```

`trade_signals` gets the same append-only `BEFORE UPDATE` / `BEFORE DELETE` triggers that protect
`live_signals`. That log is the only unbiased evidence the project will ever have, exactly as the README
already argues for the cycle signal.

If a column does later need adding to an existing table, add a tiny additive `_migrate()` helper in
`db.py` (`PRAGMA table_info` → `ALTER TABLE ADD COLUMN`) — but prefer not to need it.

---

## 4. Configuration

A new top-level `trade:` block in `config/default.yaml`, one sub-block per horizon:

```yaml
trade:
  swing:
    bars: 1d                  # decision frequency
    resolve_on: 4h            # path resolution for barriers (removes same-bar ambiguity)
    hold_days: {min: 7, max: 28}
    barrier:
      mode: vol_scaled        # fixed | vol_scaled
      sigma_mult: 1.0         # target = sigma_mult x 14d sigma ...
      floor: 0.05             # ... but never less than 5%
      stop_ratio: 0.6         # stop distance as a fraction of the target distance
    cooldown_days: 5          # no re-entry for N days after an exit
    max_concurrent: 1
    costs:
      fee_rate: 0.001         # PLACEHOLDER - see open question Q4
      slippage: 0.0005
      funding_rate_daily: 0.0 # only if shorts/leverage are ever used
    folds:
      block_days: 90
      purge_days: 28          # = hold_days.max
      embargo_days: 10
      min_train_days: 730
    backtest_start: "2015-01-01"
```

Nothing in the existing config is modified, so the cycle model's behaviour is bit-for-bit unchanged.

---

## 5. Label and exit design

### 5a. Primary: triple barrier, resolved on 4h bars

For each decision bar *t*:

- **Target** `+max(floor, sigma_mult × σ₁₄)` from entry; **stop** at `stop_ratio ×` that distance below;
  **deadline** at `hold_days.max`.
- The path is walked on **4h bars**, not daily. This matters: at a ±5% band, 1.4% of daily bars touch
  both barriers within the same bar and the outcome is genuinely ambiguous. 4h resolution cuts that
  roughly sixfold and is already in the database at no cost.
- Label = `+1` target first, `−1` stop first, `0` deadline reached. Also record **time to touch**,
  **maximum adverse excursion** and **maximum favourable excursion** — MAE in particular tells you
  whether the stop is placed sanely, and it is free to compute here.

### 5b. Ablation: fixed horizon

Exit on a scheduled date in the 7–28 day band. Label = forward return over *H*; "win" = `> +5%`.
Hold time is 7–28 days by construction, which is arguably the more literal reading of your spec
(§14 Q1). Base rates are in §2d. Run both and report both; the project's culture is to decide by
ablation, not by argument.

### 5c. Sample weighting for overlapping labels

Consecutive days share most of their forward path, so naive fitting badly overstates the effective
sample size. Weight each observation by its **uniqueness** (the inverse of how many other labels overlap
its path) — the standard treatment, and cheap to compute once the barrier touch times are known. The
cycle model already does something analogous in `sample_weights`, balancing epochs.

### 5d. Direction

Build **long/flat** first: "enter long" and "stand aside". Shorting changes the venue, adds funding
cost, and in South Africa has its own tax characterisation — a separate decision, not a parameter
(§14 Q2). The cycle model's sells are already trims rather than shorts, so long/flat is also the
consistent choice.

---

## 6. Features

The existing 122 features are built for a 3-year horizon: 200-week averages, 4-year percentiles,
halving clocks. Most will be near-useless at 14 days, and a few are actively harmful (a cycle clock
invites the model to memorise 2017). The cycle feature groups stay available as **context**, but the
new groups carry the work.

| Group | Examples | Source |
|---|---|---|
| `fast_price` | 1/3/5/10/20-day returns z-scored; position in the 20-day range; distance from the 20/50-day MA in σ units; consecutive up/down days | have |
| `fast_vol` | 10d/30d realised-vol ratio; Parkinson and Garman-Klass from OHLC; ATR%; vol-of-vol; vol regime percentile | have |
| `fast_struct` | Distance to prior swing high/low; breakout and failed-breakout flags; gap statistics | have |
| `fast_volume` | Volume z-score; volume–price divergence; up-volume share | have (Bitstamp) |
| `fast_intraday` | Overnight vs intraday return split; 4h RSI extremes per day; intraday range share; 4h realised vol | have (4h bars) |
| `fast_flow` | Exchange netflow 7/30-day z-scores; fee pressure; active-address z — the ingested flow data used *fast* rather than as 4-year percentiles | have |
| `fast_sentiment` | Fear & Greed **level and 7-day change**. Currently ingested and marked display-only; this is its natural use and a genuine short-horizon contrarian input | have |
| `fast_macro` | DXY 5-day change, 10-year yield change, risk-on proxies — lag-aware via the existing `series_meta` machinery | have |
| `cycle_context` | The cycle model's own score and strength | **see warning** |

**Leakage warning on `cycle_context`.** The production cycle model is fitted on all data to its cutoff,
so its in-sample score must never be a swing feature. Only the **walk-forward out-of-sample** score
(`predictions` table, via `pipeline.oos_scores`) is admissible, and only for dates inside that run's
test epoch. This is a subtle trap and worth its own test.

**Optional new free sources** (§14 Q11), all genuinely useful at this horizon and all free:
perpetual **funding rates** and **open interest** (Binance/Bybit public endpoints), and **Deribit DVOL**
for implied volatility. Funding in particular is one of the few honest crowding measures available.
They add new ingest dependencies, so they are a decision rather than an assumption.

Every new feature goes through `tests/test_no_lookahead.py` — the test that makes the existing backtest
honest, extended to the 4h grid.

---

## 7. Validation

### 7a. Purged, embargoed walk-forward

Halving epochs are wrong here: four folds, and the real problem is within-epoch overlap, not regime.

- Expanding-window training, **90-day test blocks** → roughly **45 blocks** from 2015 to today.
- **Purge** the 28 days before each test block out of training (their labels peek into the test period).
- **Embargo** 10 days after each block before it may be used for training again.
- Report **per block and per regime** (bull / bear / chop, defined from the cycle model's hindsight legs),
  never as a single pooled number. One good year must not hide two bad ones — the same discipline the
  cycle model applies per cycle.

### 7b. Benchmarks — all three, every time

1. **Buy-and-hold** over the identical period (47–59% a year; the number that matters).
2. **Random-entry bootstrap**: same number of trades, same holding distribution, entries shuffled,
   1,000 draws. This is what separates skill from BTC's upward drift. A strategy that does not clear
   the 95th percentile of this distribution has shown nothing.
3. **The rules baseline** (§10 Phase 2) — a simple published-style rule the learned models must beat,
   mirroring how `rules` anchors the cycle leaderboard.

### 7c. Multiple testing

A grid of models × feature sets × barrier settings × thresholds is hundreds of configurations, and the
best will look good by chance. Mitigations, all pre-registered:

- Fix the selection metric and the promotion rule **before** looking at results.
- Record the number of configurations tried in the run metadata and apply a deflated-Sharpe-style
  haircut when reporting.
- Hold out the final 12 months entirely from all selection, mirroring the existing `training_cutoff`
  holdout discipline.
- The README's own standard applies: seed discipline (≥6 seeds, ≥10 where it matters), and promotion
  only on the worst block and the mean together.

### 7d. Metrics

Per block and pooled: trade count, hit rate, mean and median net return per trade, expectancy,
profit factor, mean/median hold days (**is it actually 7–28?**), MAE/MFE distribution, exit-reason mix,
turnover, time in market, max drawdown, and return vs buy-and-hold over the same window.

---

## 8. Costs and tax — where the edge lives or dies

With a ~10% target, the round trip is not a rounding error.

| Item | Effect on a 10% gross move |
|---|---|
| Fees at 0.1% each way (offshore taker) | −0.2pp |
| Fees at 0.5% each way (`simulation.fee_rate` today) | **−1.0pp** |
| Spread + slippage, 0.05% each way | −0.1pp |
| **Tax, if gains are revenue in nature** | **−45% of the remainder** |

The existing `simulation.fee_rate: 0.005` is calibrated for a DCA accumulation model where it is
immaterial. Carrying it into a swing backtest would consume a tenth of every target. The real venue and
fee schedule must be an input, not an inherited default (§14 Q4).

**The tax point is the one to settle first.** The cycle model assumes South African **capital** gains:
40% inclusion × 45% marginal ≈ 18% effective. For assets bought and sold on a 7–28 day view, a revenue
characterisation (full marginal rate, ~45%) is a realistic possibility, and the difference is roughly
**2.5× the tax drag**. A strategy with a genuine 1.5% net-of-fee edge per trade can be profitable under
one treatment and not the other.

Two further consequences, both easy to miss:

- **Lot contamination.** If swing trades use the same BTC as the long-term holding, a swing sell consumes
  lots under the configured FIFO rule and disturbs the cycle strategy's tax position. Keeping the two
  strategies in **separate accounts, or at minimum separate modelled lot pools**, is the clean answer
  (§14 Q3, Q5).
- A short-hold trading pattern alongside a long-term holding can affect how the whole position is
  characterised. That is a question for your tax practitioner, not for this repo; the simulator should
  be able to model either treatment via a config flag so you can see both numbers.

---

## 9. Success criteria and kill criteria — agreed before building

Decided in advance, in the spirit of "promote only on the worst fold and the seed mean together".

**Ship it if**, on purged out-of-sample blocks, net of the agreed cost and tax model:

- expectancy per trade is positive with a bootstrap lower 95% bound above zero; **and**
- it clears the 95th percentile of the random-entry bootstrap; **and**
- it is positive in at least 3 of the 4 regime buckets; **and**
- median hold lands inside 7–28 days (otherwise it is a different signal than the one specified); **and**
- it beats buy-and-hold on the agreed basis (§14 Q8 — total return, or risk-adjusted, or
  "return per unit of time in market" — these give different answers and the choice must be yours).

**Abandon it if** the best configuration fails the bootstrap test after the full grid, or if the edge
exists gross and vanishes net of costs and tax. That outcome is a real possibility and a legitimate
result; the plan should be able to end in "don't deploy this" without that counting as failure.

---

## 10. Phases

| Phase | Work | Deliverable |
|---|---|---|
| **0** | Settle §14 | Decisions recorded in this file, as `PLAN.md` §0 does |
| **1** | `trade/` skeleton: `Horizon`, new tables, config block, purged fold generator, cost model, no-lookahead tests extended to 4h | `btcn trade folds --horizon swing` prints the block structure |
| **2** | Labels (both formulations), MAE/MFE, uniqueness weights, **rules baseline**, simulator with costs and both tax treatments | A baseline on the leaderboard with honest net numbers |
| **3** | Fast feature groups, no-lookahead tests, grid over models × feature sets, ablations with seed discipline | `btcn trade backtest`, a swing leaderboard |
| **4** | Threshold calibration (take / don't take), re-entry cooldown, position sizing | Trade count per year matches the agreed budget |
| **5** | Significance: random-entry bootstrap, regime breakdown, deflated reporting, final 12-month holdout | A go/no-go verdict against §9 |
| **6** | UI pages, `st.navigation` sections, live position state machine, alerts, append-only log | Swing signal visible and pushing notifications |
| **7** | Paper trading for the agreed period, reviewed against the backtest | Decision on real capital |

Phases 1–5 are research and may end at "no". Only Phase 6 touches the daily job.

---

## 11. Live operation

- The existing launchd agent runs at 00:30 and 06:00 UTC. A daily swing decision fits with no change;
  it just becomes another step in `run_daily`.
- **Position state machine**, which the repo has no concept of today: `flat → long → (target | stop |
  deadline) → cooling_off → flat`. State lives in `trade_positions`; every transition appends to
  `trade_signals` and emits exactly one alert.
- **Idempotency.** `notifications` currently de-duplicates on `(date, kind)`. Trade events need
  `(trade_id, event)` instead — a rerun of the daily job must never duplicate an ENTER.
- Alert kinds: `trade_enter`, `trade_target`, `trade_stop`, `trade_deadline`, `trade_skipped`
  (conviction near the threshold — useful while paper trading), plus the existing health alerts.
- Every notification states the horizon in the title, so a swing alert can never be mistaken for a
  cycle alert on a phone lock screen.

---

## 12. UI

Three pages in a **Swing** section:

1. **Swing overview** — current state (flat / long / cooling off), the live proposal with entry, target,
   stop and deadline, conviction, top drivers, the cycle model's current reading shown as *context*, and
   price with entry/exit markers for the last 6 months.
2. **Trade blotter** — every trade, live and backtested: dates, prices, exit reason, hold days, gross and
   net return, MAE/MFE. Equity curve against buy-and-hold and against the random-entry band.
3. **Swing research** — the leaderboard, per-block and per-regime breakdown, ablations, and the
   configuration-count disclosure from §7c.

Reuse `common.py` for colour and layout; introduce no new chart idioms beyond entry/exit markers and the
equity curve.

---

## 13. How the 4-hour to 4-day horizon slots in

The measurements in §2c say that horizon is a **2–3% band** (median time to touch: 2 days). With the
`Horizon` abstraction, most of it is configuration:

```yaml
  intraday:
    bars: 4h
    resolve_on: 4h
    hold_days: {min: 0.17, max: 4}
    barrier: {mode: vol_scaled, sigma_mult: 1.0, floor: 0.02, stop_ratio: 0.6}
    folds: {block_days: 30, purge_days: 4, embargo_days: 2}
```

What genuinely differs, and should not be pretended away:

- **Cadence.** Six decisions a day means a launchd agent every 4 hours and six times the notification
  volume. Quiet hours become a real requirement (§14 Q13).
- **Costs dominate even harder.** A 0.2% round trip against a 2.5% target is 8% of the move, and at 0.5%
  each way the strategy is dead on arrival. The cost question is not shared between horizons.
- **Microstructure appears.** Funding at 8-hour intervals, session and weekend effects, exchange
  outages, and the fact that `ingest_bitstamp` correctly drops the still-forming bar — which at 4h
  granularity means the newest decision can be up to 4 hours stale.
- **Daily-frequency features must be shifted carefully** onto the 4h grid: a daily on-chain metric is
  not known at 04:00 UTC the same day. The existing `publication_lag_days` machinery extends to this,
  but the test harness must cover it.
- **More data, not more signal.** 33,316 bars sounds like plenty until you account for overlap and the
  fact that the edge per trade is six times smaller.

Building the swing horizon with the `Horizon` seam costs perhaps 10% extra effort now and saves a
rewrite later. Building it *without* the seam, then retrofitting, is the expensive path.

---

## 14. Open questions

Each has my recommended default. "Defaults are fine" is a complete answer to most of them.

### Specification

**Q1. What does "7 to 28 days" mean?** The measurements in §2 say 5% and 7–28 days cannot both hold.
Three coherent readings:
  - (a) **Hold 7–28 days, exit on a schedule, want >5%** → fixed-horizon label (§5b). Literal reading.
  - (b) **Target ~10% (vol-scaled, 5% floor), hold whatever it takes up to 28 days** → triple barrier.
    Hold averages 7–28 days.
  - (c) 5% target, accept a 2–5 day hold → this *is* the intraday signal, and the two merge.
  *Recommended: (b), with (a) as a reported ablation.*

**Q2. Long-only, or long and short?** Shorting needs a venue that offers it, funding cost modelling, and
its own tax characterisation. *Recommended: long/flat first; revisit once there is evidence of an edge.*

**Q3. Same capital as the cycle strategy, or a separate pot?** If shared, the two models will contradict
each other (cycle says accumulate, swing says exit) and you need a priority rule. *Recommended:
a separate sleeve with its own cash, so the strategies never fight over the same coins.*

**Q4. Which venue, and what are the real fees?** Maker/taker rates and typical spread. The current
0.5% default would eat 10% of a 10% target. *No default — this one genuinely needs your answer.*

**Q5. Tax treatment to model?** Capital (40% inclusion × 45% ≈ 18%) or revenue (~45%)? And are swing
trades in a separate account from the long-term holding, or do they share a FIFO lot pool? *Recommended:
model both treatments and show both numbers; confirm the characterisation with your practitioner before
any capital is committed.*

### Modelling

**Q6. Barrier: fixed percentage or vol-scaled?** Vol-scaled adapts to the regime (8% was a 7-day trade
in 2015 and is a 10-day trade now); fixed is simpler to reason about. *Recommended: vol-scaled with a 5%
floor.*

**Q7. Stop distance and risk–reward.** `stop_ratio: 0.6` means a stop at 6% against a 10% target
(1.67:1). *Recommended: 0.6, then tune it as an ablation rather than by intuition.*

**Q8. What does "beats buy-and-hold" mean to you?** Total return, risk-adjusted return, or return per
unit of time in market? These rank strategies differently and the choice determines what gets promoted.
*Recommended: risk-adjusted as primary, total return reported alongside.*

**Q9. How many trades per year do you want?** This is the swing analogue of the cycle model's signal
budget and it drives threshold calibration. *Recommended: 15–25, i.e. roughly one or two a month.*

**Q10. Manual or automated execution?** *Recommended: manual indefinitely — the notification tells you
what to do and you place the order. If that ever changes, the executor becomes a separate process with
its own credentials (§3d).*

**Q11. May we add funding rate, open interest and Deribit DVOL?** All free, all genuinely informative at
this horizon, all new ingest dependencies. *Recommended: yes, in Phase 3, behind the existing
per-source error isolation so a failure cannot block the daily job.*

**Q12. May the swing model use the cycle model's out-of-sample score as a feature?** Leak-safe only
(§6). *Recommended: yes, and test it as an ablation — it may add nothing.*

**Q13. Backtest start date?** 2011 data exists, but pre-2015 BTC microstructure is a different market.
*Recommended: 2015-01-01, with 2019+ reported separately as the modern-regime check.*

### Operations

**Q14. Alert cadence and quiet hours?** Daily is fine for swing; the future 4h horizon will push at
03:00 local unless constrained. *Recommended: no quiet hours for swing, decide them before intraday.*

**Q15. Confirm the UI arrangement** — one app, `st.navigation` sections, separate vocabulary (§3e)?
*Recommended: yes.*

**Q16. Paper-trading period before real capital?** *Recommended: 3 months minimum, or 20 trades,
whichever is longer — with the append-only log as the record.*

**Q17. What would make you decide not to deploy it?** §9 proposes the kill criteria. Confirm or amend
them now, while nothing is invested in the answer.
