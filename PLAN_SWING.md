# BTC_Neural: Swing Signal Plan (7–28 day holds)

A second signal alongside the cycle model: a **discrete trade proposal** — enter, target, stop,
deadline — with an intended holding period of **7 to 28 days** and a target move of **more than 5%**.
Designed so that a third signal (4 hours to 4 days) slots in later as configuration rather than a rewrite.

> Research tool, not financial advice. Nothing here is a tax opinion; §8 flags where a practitioner's
> input is needed before the numbers mean anything.

**Status: built (phases 1–6); research results in §16.** The selection rule was frozen in §15 before
the grid ran. Swing signals are logged to the append-only paper log and are never sent to ntfy.

---

## 0. Decisions (confirmed 2026-10-31)

| # | Topic | Decision |
|---|---|---|
| Q1 | Target and hold | **Vol-scaled barrier**: target = 1σ of 14-day vol, floor 5% (≈10% today), 28-day deadline. 5% is the floor, not the target |
| Q2 | Direction | **Long and short** |
| Q2b | Short instrument | **Undecided — build sign-agnostic.** Funding hook is zero until a venue is named (§14) |
| Q3 | Capital | **Separate sleeve**: its own cash and coins, no shared lot pool with the cycle holding |
| Q4 | Venue and fees | **Outstanding — you are supplying exact figures** (§14). Built with an explicit **assumption of 0.1% fee + 0.05% slippage per side**, and every result is also reported across a fee sweep (0–0.5%) |
| Q5 | Tax | **Model both** capital (~18% effective) and revenue (~45%), side by side |
| Q6 | Barrier mode | Vol-scaled (settled by Q1) |
| Q7 | Stop | **0.6 × target distance** (6% stop against a 10% target, 1.67:1), tuned as an ablation |
| Q8 | Benchmark | **Must beat buy-and-hold on all three**: total return, risk-adjusted, and return per time in market |
| Q9 | Trade frequency | **15–25 a year** |
| Q10 | Execution | **Manual, indefinitely.** No exchange credentials anywhere in the repo |
| Q11 | New data | **Funding rates, open interest, Deribit DVOL** |
| Q12 | Cycle context | **Yes, leak-safe OOS score only, as an ablation.** No hard regime filter |
| Q13 | Backtest start | **2015-01-01**, with 2019+ reported separately |
| Q14 | Quiet hours | Deferred to the intraday horizon (daily cadence makes it moot for swing) |
| Q15 | UI | **Separate Streamlit app**, same repo, own port |
| Q16 | Paper trading | **Until live results clear the backtest's lower bound**, with a minimum-sample floor (§9c) |
| Q17 | Kill criteria | §9 as drafted, **plus the result must hold on the 2019+ subsample independently** |

### What these decisions do to the project

Three of them interact in ways worth stating plainly before any work starts.

**The bar is now very high, deliberately.** Q8 (beat buy-and-hold on all three measures) and Q17 (and
survive on 2019+ alone) together mean a real probability that this ends in *don't deploy*. That is a
legitimate outcome and the plan is built to reach it cheaply rather than expensively.

**Q2 makes that bar achievable.** Long-only timing against a 47–59%/yr drift is close to unwinnable on
total return. A long/short strategy can earn in drawdowns, which is where the three benchmarks diverge
and where an honest edge would show up. Q2 and Q8 are coherent together in a way that long-only and Q8
would not have been.

**Q2 and Q13 are in tension, and §7e resolves it.** Shorting BTC in 2015 was barely practicable — perp
venues arrive from 2016 (BitMEX) and liquid ones from 2019. A 2015-start backtest that happily shorts
is counterfactual for its first years.

---

## 1. Why this is not the cycle model with smaller numbers

| | Cycle signal (built) | Swing signal (this plan) |
|---|---|---|
| Independent events | ~8 turning points in 15 years | ~300–600 non-overlapping trades |
| Label | Position within a low→top leg, hindsight-anchored | Forward path outcome, computed per bar |
| Statistical problem | Desperately few examples; overfitting is the enemy | Plenty of examples; **a tiny edge against a large benchmark** is the enemy |
| Folds | 4 halving epochs | ~45 purged time blocks |
| Metric | Distance from the true extreme | Expectancy per trade, net of costs, funding and tax |
| Cost sensitivity | Negligible (a 0.5% fee against a 10× move) | **Decisive** (a 0.2–1.0% round trip against a 10% move) |
| Output | Continuous strength −10…+10 | Discrete proposal with entry, target, stop, deadline |
| What "wrong" looks like | Sells 40% below the top | Twenty small losses that quietly underperform holding |

**Almost none of the modelling code transfers; almost all of the plumbing does.** That asymmetry drives §3.

---

## 2. Reality check: measured before designing

### 2a. BTC volatility by era

| Era | Annualised vol | 14-day σ | 28-day σ |
|---|---|---|---|
| 2011+ | 85.8% | 16.8% | 23.8% |
| 2015+ | 66.7% | 13.1% | 18.5% |
| 2019+ | 61.3% | 12.0% | 17.0% |
| 2023+ | 45.7% | **9.0%** | 12.7% |

A 5% move is 0.4–0.6 standard deviations over two weeks. Not a rare event; the market breathing.

### 2b. How long a ±5% barrier actually takes to hit

| Era | Up first | Down first | Median days | Within 7 days |
|---|---|---|---|---|
| 2015+ | 53.2% | 44.0% | **3.0** | 78.5% |
| 2019+ | 52.3% | 45.7% | **3.0** | 78.4% |
| 2023+ | 52.7% | 45.2% | **5.0** | 69.3% |

Three-quarters of ±5% trades are over inside a week. This is why Q1 moved the target to vol-scaled.

### 2c. The barrier width that gives a 7–28 day hold

| Band | Era | Median days | Resolves in 7–28d | Resolves <7d |
|---|---|---|---|---|
| 3% | 2015+ / 2023+ | 2 / 2 | 7.5% / 8.9% | 92% / 91% |
| 5% | 2015+ / 2023+ | 3 / 5 | 24% / 35% | 74% / 63% |
| **8%** | 2015+ / 2023+ | **7 / 10** | 43% / 54% | 48% / 32% |
| **10%** | 2015+ / 2023+ | **10 / 15** | 47% / 54% | 36% / 20% |
| 12% | 2015+ / 2023+ | 12 / 19 | 48% / 48% | 28% / 12% |
| 20% | 2015+ / 2023+ | 21 / 31 | 39% / 22% | 10% / 3% |

8% was a 7-day trade in 2015 and is a 10-day trade now — which is exactly why the barrier is vol-scaled
rather than fixed. The same table puts the future intraday horizon at a **2–3%** band, so the two
separate cleanly.

### 2d. The base rate to beat

| Era | H | P(return > +5%) | P(return < −5%) | Mean return | P(>0) |
|---|---|---|---|---|---|
| 2015+ | 14 | 35.9% | 24.5% | +2.78% | 56.2% |
| 2019+ | 14 | 33.6% | 23.7% | +2.26% | 54.7% |
| 2023+ | 14 | 30.3% | 19.9% | +1.98% | 54.9% |
| 2023+ | 28 | 39.8% | 23.9% | +3.83% | 57.7% |

**A coin flip already "works"** — mean 14-day return is positive because BTC drifts up, so a random long
with a target and stop shows a profit from drift alone. Every claimed edge is measured against a
random-entry bootstrap, not against zero.

**Buy-and-hold returned 47–59% a year** in every era measured. Per Q8, the strategy must beat it on all
three measures.

---

## 3. Architecture: two apps, one repo, one package

### 3a. The shape

**One repo. One `trade/` package. Two Streamlit apps** — the existing cycle dashboard and a new swing
dashboard on its own port (Q15).

Shared by import, never duplicated:

- `db.py`, `ingest/`, `data.py` — the DB already carries **33,316 four-hour bars back to 2011-08** and
  5,949 daily bars. The intraday horizon's data is already ingested and backfilled nightly.
- `models/` — the `Model` API (`fit`/`predict`/`explain`/`save`) is horizon-agnostic.
- `features/` — the group registry, `columns_for`, and the no-lookahead test harness.
- `notify.py` — alert dispatch, de-duplication, ntfy and macOS channels.
- `app/common.py` — colours, hover-spine layout, chart helpers. **Imported by the swing app, not forked.**
- `config.py`, `cli.py` — one config file, one `btcn` command.

Not shared, because they encode the cycle assumption: `labels.py`, `backtest/walkforward.py`,
`backtest/metrics.py`, `calibrate.py`, `backtest/simulator.py`.

### 3b. Two apps: the costs, and how each is mitigated

Separating the UI is your call and it buys the strongest possible separation between a 3-year
accumulation signal and a 14-day trade — which is a genuine safety property, not a cosmetic one. Three
costs come with it, and each needs an explicit answer rather than being discovered later:

| Cost | Mitigation |
|---|---|
| **The feature frame gets built twice.** `load_dataset()` is a `@st.cache_resource` build, and a cache cannot span processes | **Measured, and not a real cost: the cycle frame builds in 0.3 s and the trade frame in under 1 s.** The shared on-disk cache this row originally proposed was dropped: it would save nothing, and it carried a real hazard — `FEATURE_GROUPS` is filled as a side effect of *building* features, so a cache hit would leave it empty and `columns_for` would silently return no columns |
| **`notify.dashboard_url` is one link.** A swing alert must not open the cycle dashboard | Config becomes `dashboard_url` (cycle) and `trade_dashboard_url` (swing). Each alert carries the URL for its own horizon |
| **`common.py` will drift toward a fork** | The swing app imports it. A rule in the plan and in review: the swing app adds no chart helper that belongs in `common.py`. If both apps need it, it goes in `common.py` |

A fourth, smaller one: two launchd/`preview_start` entries and two ports. Both apps go in
`.claude/launch.json` so either can be started by name.

### 3c. Why not a separate repo

The data layer and ingest are the bulk of the existing value and are identical. A separate repo means
either a shared SQLite file across two checkouts (fragile under WAL with two writers) or duplicated
ingest (worse). Two apps in one repo gives the separation you want without that cost.

### 3d. Execution stays out of both apps

Q10 is manual indefinitely, so no exchange credentials exist anywhere in this repo. If that ever
changes, the executor becomes its own process with its own restricted config and no import of either
app.

### 3e. Keeping the two signals from being confused

- Separate apps, separate ports, separate URLs in notifications.
- Different vocabulary: the cycle model has *strength*; the swing model has *trades*, *conviction* and
  *state* (flat / long / short / cooling off). No −10…+10 scale anywhere in the swing app.
- Different visual encoding: the cycle model owns the blue/red diverging ramp. The swing app uses entry
  and exit markers on price plus an equity curve, and leaves the ramp alone.
- Every notification title names its horizon, so the two can't be confused on a phone lock screen.

### 3f. Proposed layout

```
src/btc_neural/
  trade/
    __init__.py
    horizons.py        # Horizon specs: SWING, later INTRADAY
    labels.py          # triple barrier (4h-resolved), sign-agnostic; fixed-horizon variant
    features.py        # short-horizon feature groups, registered like the cycle ones
    folds.py           # purged + embargoed walk-forward block generator
    metrics.py         # expectancy, hit rate, MAE/MFE, profit factor, benchmark tests
    simulator.py       # discrete round trips: costs, funding, slippage, tax, position state
    calibrate.py       # score -> take / don't take, plus the re-entry cooldown
    pipeline.py        # train / save / load per horizon
    live.py            # position state machine, trade log, alerts
  features/
  app/                 # existing cycle app, unchanged apart from reading the shared cache
  app_trade/
    Home.py            # Swing overview
    pages/
      1_Blotter.py
      2_Research.py
```

`Horizon` is a frozen dataclass: bar timeframe, path-resolution timeframe, min/max hold, barrier rule,
cost model, feature groups, fold geometry, live cadence, alert kinds. `SWING` and `INTRADAY` are two
instances; everything downstream takes a `Horizon`.

### 3g. Database: add tables, never alter them

`db.connect()` runs `executescript(SCHEMA)` with `CREATE TABLE IF NOT EXISTS`. **There is no migration
path: adding a column to an existing table silently does nothing on your existing database.** So the
swing signal adds new tables and touches none of the current ones.

```sql
CREATE TABLE trade_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, horizon TEXT, model_type TEXT,
    feature_set TEXT, role TEXT, config_json TEXT, selection_metric REAL,
    weights_path TEXT, notes TEXT);
CREATE TABLE trade_folds (run_id TEXT, block TEXT, metric TEXT, value REAL, PRIMARY KEY (run_id, block, metric));
CREATE TABLE trade_backtest (        -- one row per simulated trade
    run_id TEXT, trade_id TEXT, side TEXT, entry_date TEXT, entry_price REAL,
    exit_date TEXT, exit_price REAL, exit_reason TEXT, gross_ret REAL, funding_cost REAL,
    net_ret REAL, hold_days REAL, mae REAL, mfe REAL, PRIMARY KEY (run_id, trade_id));
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
`live_signals`. Per Q16 that log is the gate on real capital, so it must be tamper-evident.

---

## 4. Configuration

A new top-level `trade:` block in `config/default.yaml`. Nothing existing is modified, so the cycle
model's behaviour is bit-for-bit unchanged.

```yaml
trade:
  swing:
    bars: 1d
    resolve_on: 4h              # path resolution: removes same-bar barrier ambiguity
    hold_days: {min: 7, max: 28}
    sides: [long, short]        # Q2; simulator and labels are sign-agnostic
    barrier:
      mode: vol_scaled          # Q1/Q6
      sigma_mult: 1.0           # target = 1 sigma of 14d vol ...
      floor: 0.05               # ... never less than 5%
      stop_ratio: 0.6           # Q7: 6% stop against a 10% target
    cooldown_days: 5
    max_concurrent: 1
    target_trades_per_year: 20  # Q9: calibration aims at 15-25
    costs:
      fee_rate: null            # Q4 OUTSTANDING - no default, backtest refuses to run without it
      slippage: 0.0005
      funding:
        enabled: false          # Q2b: zero until a short venue is named
        source: null            # binance | bybit | <local venue>
    tax:
      treatments: [capital, revenue]   # Q5: both, reported side by side
      capital:  {inclusion_rate: 0.40, marginal_rate: 0.45}
      revenue:  {marginal_rate: 0.45}
    sleeve:
      separate_lots: true       # Q3: own lot pool, never touches the cycle holding
    folds:
      block_days: 90
      purge_days: 28
      embargo_days: 10
      min_train_days: 730
    backtest_start: "2015-01-01"        # Q13
    short_start: "2019-01-01"           # §7e: shorts not simulated before this
    modern_subsample_start: "2019-01-01" # Q17: must also pass on this alone
  dashboard_url: http://localhost:8503   # Q15: the swing app's own URL
```

`fee_rate: null` is deliberate. The backtest raises rather than silently inheriting the cycle model's
0.5%, which would consume a tenth of every target.

---

## 5. Label and exit design

### 5a. Primary: triple barrier, sign-agnostic, resolved on 4h bars

For each decision bar *t* and each side:

- **Target** at `max(floor, sigma_mult × σ₁₄)` in the direction of the trade; **stop** at
  `stop_ratio ×` that distance against it; **deadline** at `hold_days.max`.
- The path is walked on **4h bars**. At a ±5% band, 1.4% of daily bars touch both barriers within the
  same bar and the outcome is genuinely ambiguous; 4h resolution cuts that roughly sixfold, using data
  already in the database.
- Label `+1` target first, `−1` stop first, `0` deadline. Also record **time to touch**, **maximum
  adverse excursion** and **maximum favourable excursion** — MAE tells us whether the 0.6 stop ratio is
  being hit by noise, and it is free to compute here.
- **Sign-agnostic throughout** (Q2b): the same code labels a long and a short, with funding applied by
  side once a venue exists.

### 5b. Ablation: fixed horizon

Exit on a scheduled date in the 7–28 day band. Label = forward return over *H*; win = `> +5%`. Base
rates in §2d. Reported alongside, not used for promotion unless it wins outright.

### 5c. Sample weighting for overlapping labels

Consecutive days share most of their forward path, so naive fitting badly overstates the effective
sample size. Weight each observation by its **uniqueness** — the inverse of how many other labels
overlap its path — computed from the barrier touch times. The cycle model does something analogous in
`sample_weights`.

---

## 6. Features

The existing 122 features target a 3-year horizon. They stay available as *context*; the new groups
carry the work.

| Group | Examples | Availability |
|---|---|---|
| `fast_price` | 1/3/5/10/20-day returns z-scored; position in the 20-day range; distance from 20/50-day MA in σ units; consecutive up/down days | 2010+ |
| `fast_vol` | 10d/30d realised-vol ratio; Parkinson and Garman-Klass from OHLC; ATR%; vol-of-vol; vol regime percentile | 2010+ |
| `fast_struct` | Distance to prior swing high/low; breakout and failed-breakout flags; gap statistics | 2010+ |
| `fast_volume` | Volume z-score; volume–price divergence; up-volume share | 2011+ (Bitstamp) |
| `fast_intraday` | Overnight vs intraday return split; 4h RSI extremes per day; intraday range share; 4h realised vol | 2011-08+ |
| `fast_flow` | Exchange netflow 7/30-day z-scores; fee pressure; active-address z — ingested flow data used *fast* rather than as 4-year percentiles | 2011-04+ |
| `fast_sentiment` | Fear & Greed **level and 7-day change**. Currently ingested and marked display-only; this is its natural use | 2018-02+ |
| `fast_macro` | DXY 5-day change, 10-year yield change, risk-on proxies — lag-aware via `series_meta` | 2010+ |
| `derivatives` | **Funding rate** (level, 7d mean, z-score), **open interest** (level, change, OI-vs-price divergence), **DVOL** and the implied-vs-realised vol spread | **~2019+ — see below** |
| `cycle_context` | The cycle model's walk-forward OOS score | 2012+ |

### 6a. The derivatives data starts late, and that matters

Q11 adds funding, open interest and DVOL; Q13 starts the backtest in 2015. Those don't overlap for the
first four years. Major-venue perp history begins around 2019 (BitMEX from 2016, with thin and
idiosyncratic funding), and DVOL is later still. Exact start dates get verified at ingest and recorded
in `series_meta` like every other source.

The consequence is a **feature availability window**, which the plan handles explicitly rather than by
hoping the models cope:

- Feature sets are declared with a valid-from date. A set containing `derivatives` is only backtested
  from its own start, so a 2015–2019 block is never scored on a model fed four years of NaN.
- The leaderboard shows **two populations**: sets valid from 2015, and sets valid from 2019. They are
  not directly comparable and are never ranked against each other in one table.
- This dovetails with Q17 (the 2019+ subsample must stand alone anyway), so the derivatives sets are
  judged exactly where they are valid.

### 6b. Leakage warning on `cycle_context`

The production cycle model is fitted on all data to its cutoff, so its in-sample score must never be a
swing feature. Only the **walk-forward out-of-sample** score (`predictions` table, via
`pipeline.oos_scores`) is admissible, and only for dates inside that run's test epoch. Per Q12 it is an
ablation, not a hard filter, and it gets its own test.

Every new feature goes through `tests/test_no_lookahead.py`, extended to the 4h grid.

---

## 7. Validation

### 7a. Purged, embargoed walk-forward

- Expanding-window training, **90-day test blocks** → roughly **45 blocks** from 2015.
- **Purge** the 28 days before each test block out of training; **embargo** 10 days after.
- Report **per block and per regime** (bull / bear / chop, from the cycle model's hindsight legs), never
  as one pooled number.

### 7b. Benchmarks — Q8 requires all three

Promotion requires beating buy-and-hold on **total return, risk-adjusted return (Sharpe/Sortino), and
return per unit of time in market**, plus:

1. **Random-entry bootstrap**: same trade count, same holding distribution, same long/short mix,
   entries shuffled, 1,000 draws. Must clear the 95th percentile. This is what separates skill from
   BTC's drift.
2. **The rules baseline** (Phase 2) — a simple published-style rule the learned models must beat,
   mirroring how `rules` anchors the cycle leaderboard.

### 7c. The 2019+ requirement (Q17)

Every criterion above must hold **on the 2019+ subsample evaluated on its own**, not merely on the
pooled 2015+ result. This guards against an edge that existed only in the earlier, less efficient
market — and it is also where the derivatives features live (§6a).

### 7d. Multiple testing

A grid of models × feature sets × barrier settings × thresholds is hundreds of configurations, and the
best will look good by chance. Pre-registered mitigations:

- Fix the selection metric and promotion rule **before** looking at results.
- Record the configuration count in run metadata; apply a deflated-Sharpe-style haircut when reporting.
- Hold the final 12 months out of all selection, mirroring the existing `training_cutoff` discipline.
- Seed discipline per the README: ≥6 seeds, ≥10 where it matters; promote on the worst block and the
  mean together.

### 7e. Shorts before shorting was practicable

Q2 wants shorts; Q13 starts in 2015. Shorting BTC in 2015 meant thin, expensive venues with real
counterparty risk, and a backtest that shorts freely from 2015 is describing trades you could not have
placed. Resolution:

- **Shorts are simulated only from `short_start` (2019-01-01).** Before that the strategy is long/flat.
- The long-only 2015–2019 segment is still useful — it tests the long side over an extra cycle.
- Reported separately so the two periods are never silently pooled into one performance claim.

### 7f. Metrics

Per block and pooled: trade count, hit rate by side, mean and median net return per trade, expectancy,
profit factor, mean/median hold days (**is it actually 7–28?**), MAE/MFE distribution, exit-reason mix,
funding paid, turnover, time in market, max drawdown, and all three benchmark comparisons.

---

## 8. Costs, funding and tax

With a ~10% target, the round trip is not a rounding error.

| Item | Effect on a 10% gross move |
|---|---|
| Fees at 0.1% each way | −0.2pp |
| Fees at 0.5% each way (the cycle model's current default) | **−1.0pp** |
| Spread + slippage, 0.05% each way | −0.1pp |
| Funding on a 14-day perp long at 0.01%/8h | **−0.4pp** (shorts receive it) |
| **Tax** | **−18% (capital) or −45% (revenue) of the remainder** |

Per Q4 the fee rate has no default and the backtest refuses to run without one. Per Q2b funding is zero
until a venue is named, and the hook is wired from the start so enabling it is config, not code.

**Tax (Q5): both treatments, side by side.** The cycle model assumes capital gains (40% inclusion × 45%
≈ 18% effective). For 7–28 day trades a revenue characterisation (~45%) is a realistic possibility, and
the difference is roughly **2.5× the tax drag** — enough to flip the verdict. Derivatives positions, if
shorts are ever placed, carry their own characterisation. None of this is a tax opinion: the simulator
shows you both numbers, and your practitioner decides which one is real before any capital moves.

**Q3 removes the lot-contamination problem.** With a separate sleeve, swing trades have their own lot
pool and never consume FIFO lots belonging to the long-term holding, so the cycle strategy's tax
position is undisturbed. The simulator enforces this rather than assuming it.

---

## 9. Success and kill criteria

Set now, while nothing is invested in the answer.

### 9a. Ship it if — on purged out-of-sample blocks, net of fees, funding and tax

- expectancy per trade is positive with a bootstrap lower 95% bound above zero; **and**
- it clears the 95th percentile of the random-entry bootstrap; **and**
- it is positive in at least 3 of the 4 regime buckets; **and**
- median hold lands inside 7–28 days (otherwise it is a different signal than the one specified); **and**
- it beats buy-and-hold on **all three** measures (Q8); **and**
- every one of the above **also holds on the 2019+ subsample alone** (Q17).

### 9b. Abandon it if

the best configuration fails the random-entry bootstrap after the full grid, or the edge exists gross
and vanishes net of costs and tax, or it passes pooled but fails on 2019+.

Given 9a, *don't deploy* is a likely outcome. The phase order (§10) is arranged so the cheapest tests
run first and that answer arrives early rather than after the UI is built.

### 9c. Paper trading (Q16), with the goalposts fixed in advance

You chose "until live results clear the backtest's lower bound". That rule is only meaningful if the
comparison is specified before going live, so:

- The gate is: **live mean net return per trade ≥ the backtest's 95% bootstrap lower bound**, with the
  backtest figure **frozen and recorded in `trade_runs` on the day paper trading starts**.
- **Minimum 10 closed trades** before the test may be evaluated at all, so two lucky trades cannot
  trigger deployment. At 15–25 trades a year that is roughly 5–8 months.
- Evaluated **once a quarter**, not continuously — a test checked daily will eventually pass by chance.
- If the live mean is below the bound after 25 trades, stop rather than extend.

---

## 10. Phases

| Phase | Work | Deliverable |
|---|---|---|
| **0** | ✅ Decisions (§0). Outstanding: Q4 fees, Q2b instrument | This document |
| **1** | `trade/` skeleton: `Horizon`, new tables, config block, purged fold generator, shared feature cache, cost model, no-lookahead tests extended to 4h | `btcn trade folds --horizon swing` prints the block structure |
| **2** | Labels (both formulations, sign-agnostic), MAE/MFE, uniqueness weights, **rules baseline**, simulator with costs, funding hook and both tax treatments | A baseline on the leaderboard with honest net numbers |
| **3** | Fast feature groups; funding/OI/DVOL ingest with availability windows; grid over models × feature sets; ablations with seed discipline | `btcn trade backtest`, a swing leaderboard |
| **4** | Threshold calibration to 15–25 trades/year, re-entry cooldown, position sizing | Trade count matches the budget |
| **5** | Significance: random-entry bootstrap, regime breakdown, 2019+ subsample, deflated reporting, 12-month holdout | **Go/no-go against §9** |
| **6** | `app_trade/` scaffold, swing pages, live position state machine, alerts, append-only log | Swing signal visible and pushing notifications |
| **7** | Paper trading to the §9c gate | Decision on real capital |

Phases 1–5 are research and may end at "no". Only Phase 6 touches the daily job or builds a second app
— deliberately, so a negative verdict costs the least.

---

## 11. Live operation

- The existing launchd agent runs at 00:30 and 06:00 UTC. A daily swing decision fits with no schedule
  change; it becomes another step in `run_daily`.
- **Position state machine**, which the repo has no concept of today: `flat → long|short → (target |
  stop | deadline) → cooling_off → flat`. State lives in `trade_positions`; every transition appends to
  `trade_signals` and emits exactly one alert.
- **Idempotency.** `notifications` de-duplicates on `(date, kind)`. Trade events need
  `(trade_id, event)` instead — a rerun of the daily job must never duplicate an ENTER.
- Alert kinds: `trade_enter`, `trade_target`, `trade_stop`, `trade_deadline`, `trade_skipped`
  (conviction near the threshold, useful while paper trading), plus the existing health alerts.
- Every alert names its horizon in the title and links to `trade.dashboard_url`, not the cycle
  dashboard.
- Execution is manual (Q10): the notification states side, entry, target, stop and deadline, and you
  place the order. The simulator assumes a slippage allowance for the delay.

---

## 12. UI — the swing app

`app_trade/`, own port (8503), own launch entry, importing `app/common.py`.

1. **Home — Swing overview**: current state (flat / long / short / cooling off), the live proposal with
   entry, target, stop, deadline and conviction, top drivers, the cycle model's current reading shown
   as *context only*, and price with entry/exit markers for the last 6 months.
2. **Blotter**: every trade, live and backtested — dates, side, prices, exit reason, hold days, gross
   return, funding, net return under both tax treatments, MAE/MFE. Equity curve against buy-and-hold
   and against the random-entry band.
3. **Research**: the leaderboard (split into 2015-valid and 2019-valid populations), per-block and
   per-regime breakdown, ablations, and the configuration-count disclosure from §7d.

---

## 13. How the 4-hour to 4-day horizon slots in

§2c puts that horizon at a **2–3% band** (median time to touch: 2 days). With the `Horizon`
abstraction most of it is configuration:

```yaml
  intraday:
    bars: 4h
    resolve_on: 4h
    hold_days: {min: 0.17, max: 4}
    barrier: {mode: vol_scaled, sigma_mult: 1.0, floor: 0.02, stop_ratio: 0.6}
    folds: {block_days: 30, purge_days: 4, embargo_days: 2}
```

It joins the **swing app** as a third section rather than becoming a third app — the separation that
matters is cycle-versus-trading, and two trading horizons share vocabulary, state machine and blotter.

What genuinely differs:

- **Cadence.** Six decisions a day means a launchd agent every 4 hours and six times the notification
  volume. Quiet hours (Q14, deferred) become a real requirement at that point.
- **Costs dominate harder.** A 0.2% round trip against a 2.5% target is 8% of the move; at 0.5% each
  way the strategy is dead on arrival.
- **Funding is per-position, not per-day** — 8-hourly settlement is a material fraction of a 2-day hold.
- **Microstructure appears**: session and weekend effects, exchange outages, and the fact that
  `ingest_bitstamp` correctly drops the still-forming bar — which at 4h granularity means the newest
  decision can be up to 4 hours stale.
- **Daily features must be shifted carefully** onto the 4h grid: a daily on-chain metric is not known at
  04:00 UTC the same day. The `publication_lag_days` machinery extends to this; the test harness must
  cover it.
- **More data, not more signal.** 33,316 bars sounds like plenty until you account for overlap and an
  edge per trade roughly six times smaller.

---

## 14. Outstanding

Two items block Phase 2 and Phase 6 respectively. Everything else in §0 is settled.

**Q4 — venue, maker/taker rates and typical spread.** You are supplying these. Until then
`costs.fee_rate` is `null` and the backtest refuses to run, by design: inheriting the cycle model's
0.5% would quietly consume a tenth of every target and produce confident, wrong numbers. *Blocks
Phase 2.*

**Q2b — the short instrument.** Research proceeds sign-agnostic with funding set to zero, so this does
not block modelling. It does block live deployment of the short side, and it changes the cost model:
perps pay 8-hourly funding, dated futures pay basis and a roll. *Blocks Phase 6 for shorts.*

**Q14 — quiet hours.** Deferred to the intraday horizon by agreement; swing's daily cadence makes it
moot.

One thing to flag rather than ask: with Q8 and Q17 set where they are, and `don't deploy` a likely
outcome, it is worth deciding now whether a *partial* result — say, a long side that works and a short
side that doesn't — is interesting to you. If it is, Phase 5 should report the sides separately rather
than only as a combined strategy. That costs nothing extra if decided before Phase 2.

---

## 15. Pre-registration (frozen 2026-10-31, before the grid ran)

Written and committed before any grid result existed, so the selection cannot be fitted to the answer.

**Fixed inputs.** Labels as configured: vol-scaled target (1σ of 14-day vol, 5% floor), stop at 0.6× the
target distance, 28-day deadline, walked on 4h bars. Costs: **0.1% fee + 0.05% slippage per side
(assumed — Q4)**. Longs are spot; shorts are perps and pay or receive the recorded Binance/Bybit funding.
Shorts are not simulated before 2019-01-01 (§7e). Trading sleeve is separate, 1× notional, one position
at a time, 5-day re-entry cooldown.

**Grid.**

| Dimension | Values |
|---|---|
| Models | `srules_momentum`, `srules_pullback` (published baselines, no training), `linear` (ridge), `lgbm` × 10 seeds, `ens` (ridge + LightGBM) × 10 seeds, `nn` × 3 seeds on `fast_core` and `fast_all` (reference track) |
| Population A (valid 2015+) | `fast_core`, `fast_flow`, `fast_macro`, `fast_cycle`, `fast_all` |
| Population B (valid 2019+) | `fast_sent`, `fast_funding`, `fast_deriv` |
| Trade threshold | today's score beats the {0.75, 0.80, 0.85, 0.90, 0.95} quantile of the model's own trailing 730 days of out-of-sample scores; learned models must also expect to clear their round-trip cost |

**Windows.** *Full* = evaluation start (2015-01-01 for population A) to 2025-10-29. *Modern* = 2019-01-01
to 2025-10-29. *Holdout* = 2025-10-30 onward — **never used for selection**.

**Selection metric.** For each (model, feature set, quantile): the **mean across seeds of
min(Sharpe full, Sharpe modern)**, pre-tax, net of fees, slippage and funding, long and short combined.

**Eligibility.** Seed-mean trades per year on the full window between 12 and 30 (the 15–25 target, Q9,
with a margin).

**Promotion.** The eligible population-A configuration with the highest selection metric. The verdict
is run on its **seed ensemble** (out-of-sample scores averaged across its seeds) against every §9a
criterion, separately for the capital and revenue tax treatments, on the full and modern windows.

**Population B.** A late feature set is adopted only if it beats `fast_core` — same model, same quantile,
**same window** — by more than two standard errors across seeds.

**Reported but not used to select:** the holdout year; long-only and short-only results (decided in
conversation: a working long side is of interest even if the short side fails); a fee sweep from 0% to
0.5% per side; funding switched off; a stop ratio of 1.0 (Q7); a deflated Sharpe ratio over every
population-A configuration tried.

**Disclosure.** Before this section was written, a smoke test ran three configurations (`srules_momentum`,
`linear`, `lgbm` seed 7, all on `fast_core`, quantiles 0.90 and 0.95). It showed quantiles ≥ 0.90 produce
only 7–13 trades a year — below the Q9 target — so the quantile grid was widened downward from the
originally planned {0.90, 0.95, 0.975}. Their Sharpe ratios were seen (0.25–0.62, against 1.13 for
buy-and-hold). Those runs were deleted and are re-run inside the grid like everything else.
