# Stock Movement Predictor

Predicting **whether** a stock rises during the next trading session — with
development-only model selection, a final test that can only be opened once,
execution costs charged the way an intraday strategy actually pays them, and
bootstrap intervals on every headline number.

> **Research and education only. Not investment advice.** No claim of
> profitability is made or implied. See [Limitations](#limitations).

**The result is negative, and the project is built to establish that credibly.**
On AAPL the selected model clears the predictive-edge gate on development data and
then fails on the sealed final test. Every bootstrap interval includes zero. That
is the expected outcome for daily direction on a liquid large-cap; the point of
the work is the apparatus that can tell the difference between a real edge and a
convincing one.

---

## Contents

- [What it predicts](#what-it-predicts)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [The two-stage protocol](#the-two-stage-protocol)
- [Execution and cost models](#execution-and-cost-models)
- [Leakage controls](#leakage-controls)
- [Baselines](#baselines)
- [Models and selection](#models-and-selection)
- [Metrics and uncertainty](#metrics-and-uncertainty)
- [Results](#results)
- [Reproducing the results](#reproducing-the-results)
- [Model persistence and prediction](#model-persistence-and-prediction)
- [Limitations](#limitations)
- [Adding a new ticker](#adding-a-new-ticker)
- [Testing and quality gates](#testing-and-quality-gates)
- [Project layout](#project-layout)

---

## What it predicts

Using only information available at the close of session `t`, estimate the
probability that the stock rises during session `t+1`:

```
next_session_return_t = Close(t+1) / Open(t+1) - 1
target_t              = 1 if next_session_return_t > 0 else 0
```

```
signal generated after Close(t)  ->  enter at Open(t+1)  ->  exit at Close(t+1)
```

This is the **executable** target: it measures the return a trader could actually
capture by acting on the signal. The older close-to-close convention
(`Close(t+1) > Close(t)`) assumes trading at the very close used to make the
decision, which is not possible. It is retained as an explicitly labelled
research-only experiment, never as the default.

The model outputs a **probability**, and two distinct thresholds act on it:

| Threshold | Default | Purpose |
|---|---|---|
| `classification_value` | 0.50 | probability → predicted label, for metrics |
| `trading_value` | 0.55 | probability → position, for the backtest |

Keeping them separate matters: conflating them means a trading decision silently
changes a reported accuracy.

Prices are adjusted for splits and dividends (`auto_adjust: true`, always passed
explicitly, because yfinance has changed this default before). A session with
exactly zero return is labelled **down** — the test is `> 0`.

---

## Architecture

```
                        configs/*.yaml  (pydantic; unknown keys and
                              │          incoherent combinations rejected)
                              ▼
    yfinance ──► Data ingestion ──► Parquet cache + SHA-256 (verified on read)
                              │
                              ▼
                    Exchange calendar ──► drop the final bar only if the
                              │            session is still open
                              ▼
                      Data validation ──► impossible OHLC bars fail;
                              │            market anomalies are recorded
                              ▼
                   Feature engineering  (29 causal, scale-free features)
                              │
                              ▼
                     Label generation   (the only forward-looking module)
                              │
                              ▼
              ┌───────────── Split ─────────────┐
              │                                 │
     DEVELOPMENT (first 80%)              FINAL TEST (last 20%)
              │                                 │   sealed
    shared walk-forward folds                   │
              │                                 │
    baselines + every candidate                 │
              │                                 │
    deterministic ranking                       │
              │                                 │
    predictive-edge gate                        │
              │                                 │
    selection_decision.json  ──────────────────►│  opened ONCE
                                                ▼
                                    cost-aware backtest,
                                    block bootstrap,
                                    saved model, final_test.lock.json
                                                │
                                                ▼
                              artifacts/runs/<run_id>/  +  predict
```

---

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

`uv.lock` is committed, so this resolves to the exact versions the published
results were produced with. The `lstm` extra pulls TensorFlow (a large download)
and is needed only for the optional LSTM candidate.

Without the LSTM phase:

```bash
uv sync --extra dev
```

---

## Usage

The full flow, in the order it is meant to be run:

```bash
uv run stock-movement download --config configs/default.yaml
```

```bash
uv run stock-movement build-features --config configs/default.yaml
```

```bash
uv run stock-movement evaluate-candidates --config configs/default.yaml
```

```bash
uv run stock-movement select-model --config configs/default.yaml
```

```bash
uv run stock-movement final-test --run-id <RUN_ID>
```

```bash
uv run stock-movement predict --run-id <RUN_ID> --latest
```

`select-model` prints the run id to pass to `final-test`. To do both stages at
once:

```bash
uv run stock-movement run-all --config configs/default.yaml
```

Other commands: `evaluate` and `backtest` print metrics from a finished run,
`show-run` summarises its provenance and status, `list-runs` enumerates them.
Every failure exits non-zero.

| Command | What it does |
|---|---|
| `download` | fetch, validate and cache OHLCV with a digest |
| `build-features` | build the feature/label table and print the manifest order |
| `evaluate-candidates` | compare candidates on development folds; creates no run |
| `select-model` | **stage 1** — pick one candidate, write the decision, no test metrics |
| `final-test` | **stage 2** — score the sealed holdout once, then lock it |
| `run-all` | both stages |
| `evaluate` / `backtest` | read a finished run |
| `predict` | predict from the saved model, no retraining |
| `show-run` | run id, hashes, git SHA, edge status, lock status, model path |

Shipped experiments in `configs/experiments/`:

| Config | Purpose |
|---|---|
| `research_close_to_close.yaml` | the non-executable research convention, clearly labelled |
| `spy.yaml`, `msft.yaml` | robustness on other instruments |
| `benchmark_features.yaml` | adds SPY-relative features |
| `tuned_thresholds.yaml` | tunes both thresholds on development out-of-fold predictions |
| `lstm_aapl.yaml` | adds the optional LSTM as a third candidate family |

---

## The two-stage protocol

### The problem it solves

An earlier version of this project ran one model per config and immediately
reported its test metrics. Running logistic regression, gradient boosting, an
LSTM and a benchmark-feature variant therefore **exposed the same final test four
times**, and the best number got written up. That is multiple comparisons against
a holdout, and it inflates results whether or not anyone intends it. Those earlier
runs should be read as exploratory, not as out-of-sample evidence.

### Stage 1 — `select-model`

Everything happens inside the development window (the first 80% of rows):

- all candidates and all baselines are scored on **identical** walk-forward folds;
- ranking is a total order — mean balanced accuracy, then fold-to-fold standard
  deviation, then log loss, then model complexity, then hyperparameter simplicity,
  then name — so a rerun selects the same model;
- thresholds, if tuned, are tuned on the winner's out-of-fold development
  predictions;
- the decision is written to `selection_decision.json`, which records
  `max_row_index_used` and certifies `test_data_used_for_selection: false`.

Stage 1 writes no final-test metric, no backtest, and no saved model. A test
asserts those files do not exist afterwards.

### Stage 2 — `final-test`

Loads the locked decision, verifies the config hash matches, refits the winner on
all development rows, scores the holdout **once**, backtests, bootstraps, saves the
model, and writes `final_test.lock.json`.

A second `final-test` on the same run **fails**:

```bash
uv run stock-movement final-test --run-id <RUN_ID>
# error: FinalTestAlreadyCompletedError: the final test ... already completed
```

Overriding requires both a flag and a reason, and every rerun is appended to the
lock's history:

```bash
uv run stock-movement final-test --run-id <RUN_ID> --allow-test-rerun --rerun-reason "found a data bug"
```

This is what makes "the test set was used once" a property of the code rather than
a claim in a README.

### The predictive-edge gate

`edge_detected` is true only when **all four** hold on development data:

1. balanced accuracy at least **+0.01** over the best baseline;
2. beats the best baseline in at least **4 of 5** folds;
3. mean ROC-AUC **> 0.50**;
4. mean MCC **> 0**.

Condition 2 carries most of the weight. A model can clear an average margin by
winning one fold enormously and losing the rest — noise in a convincing costume.

A winner is always saved, but when the gate fails the artifacts say so plainly,
and `predict` refuses to imply otherwise.

---

## Execution and cost models

The two modes compute **different numbers**. Using the wrong one understates cost
by roughly two orders of magnitude on a fully-invested strategy.

### `next_open` (default, executable)

Enter at the open, exit at the close of the same session. Every active session is
a *complete round trip*, so every active session pays:

```
cost_t = |position_t| * round_trip_cost_rate      # 10 bps default
```

Holding `position = [1, 1, 1]` is three separate intraday trades and three
round-trip costs — not one.

> **This was a real bug.** The previous implementation charged cost only when the
> *position changed*, so an always-invested intraday strategy paid a single 10 bps
> entry instead of 10 bps × 819 sessions. It reported +81% where the truth is
> −8.9%: an 82-percentage-point error, entirely inside the cost model.

### `close_to_close` (research only)

A position is held across sessions, so cost follows position *changes*:

```
cost_t = |position_t - position_{t-1}| * one_way_cost_rate     # 5 bps default
```

`execution_mode` must match `labels.target_definition`, and config validation
rejects a mismatch at load time.

### Benchmark naming

| Name | What it is |
|---|---|
| `always_long_intraday` | re-enters every session under `next_open`; pays a round trip each time |
| `buy_and_hold_close_to_close` | genuinely holds the asset; pays one entry cost |
| `cash` | never invested; exactly zero |

The intraday always-active strategy is **not** buy-and-hold and is never labelled
as such. Both are always reported, because "would holding have done better?" is
the question a reader actually has.

### Trade counters

Three distinct quantities, which coincide only by accident:

| Counter | Meaning |
|---|---|
| `n_active_sessions` | sessions with a non-zero position |
| `n_completed_trades` | completed round trips (== active sessions under `next_open`) |
| `n_position_changes` | times the position changed |

All per-trade statistics — average, median, win rate, best, worst, profit factor —
are computed from **net** returns, after costs.

---

## Leakage controls

Leakage is the failure mode of this problem class. A model scoring 85% here has
found a bug, not an edge. Every control is enforced by a test.

| Control | Mechanism | Test |
|---|---|---|
| No shuffling, ever | only chronological splits exist in `split.py` | `test_split.py` |
| Preprocessing fitted per fold | scaler/imputer live *inside* the `Pipeline` | `test_leakage.py::test_scaler_is_fitted_on_training_rows_only` |
| No forward-looking features | static check for `shift(-n)` and `center=True` | `test_features.py::test_no_feature_uses_a_negative_shift` |
| Rewriting the future cannot change the past | rebuild on tampered data, compare bit-for-bit | `test_features.py::test_changing_a_future_price_cannot_change_a_past_feature` |
| Gap at every boundary | `gap >= horizon`, enforced by config validation | `test_leakage.py::test_gap_removes_the_row_whose_label_overlaps_the_next_segment` |
| Labels never in `X` | `assert_no_leakage_columns` on every dataset | `test_leakage.py` |
| Selection never reads the holdout | asserted on row indices, recorded in the decision | `test_selection.py::test_candidate_selection_never_reads_test_rows` |
| Test opened once | `final_test.lock.json` + rerun guard | `test_artifacts.py`, `test_pipeline.py` |
| Benchmark not forward-filled | inner join on shared sessions; >2% loss fails | `test_features.py::test_benchmark_does_not_forward_fill_missing_session` |
| The harness can detect leakage | a planted oracle feature must score >95% | `test_leakage.py::test_a_deliberately_leaked_feature_is_detectably_too_good` |

**Why the gap matters.** With a horizon-1 label, the label of the last training
row *is* the return of the first evaluation row. One row is dropped at each
boundary and belongs to no split.

**Why random cross-validation is absent.** K-fold puts future data in training and
past data in validation. It is not implemented, so it cannot be selected by
accident.

---

## Baselines

Four baselines, scored on identical dates through the identical protocol.

| Baseline | Hard rule | Probability |
|---|---|---|
| **Majority** | training majority class | training positive frequency |
| **Always up** | always 1 | training positive frequency |
| **Last direction** | sign of the current session's open-to-close return | P(up \| current direction), from training |
| **Random** | seeded Bernoulli draw at the training prior | the constant training prior |

Each baseline reports its *own* hard rule for threshold metrics (accuracy, F1,
MCC, confusion matrix) and its probability for log loss, Brier and ROC-AUC.
Thresholding an invented probability at 0.5 would silently change what a baseline
asserts — a "majority" baseline with a 0.48 prior would flip to predicting down
every day.

The random baseline emits the **constant prior**, not uniform noise. Emitting
noise (the previous behaviour) gave it a log loss of ~1.06 against a possible
~0.69, making a diagnostic look pathological for reasons unrelated to its
predictions.

---

## Models and selection

| Family | Complexity rank | Notes |
|---|---|---|
| Logistic regression | 1 | interpretable, well-calibrated, hard to overfit |
| HistGradientBoosting | 2 | nonlinearity and interactions; permutation importance |
| LSTM *(optional)* | 3 | multivariate sequences, sigmoid, BCE, early stopping, `shuffle=False` |

On an exact tie the simpler family wins, because a tie on ~3,000 noisy rows is not
evidence for the more flexible model. Grids are deliberately tiny (3–5 candidates
per family): every extra candidate raises the chance one looks good by luck.

Gradient boosting exposes neither `coef_` nor `feature_importances_`, so it falls
back to permutation importance — without which it reported no importance at all.

### 29 causal, scale-free features

Lagged returns (1/2/3/5/10/20) · skip-a-day momentum (independently configured
from volatility windows) · fraction of up-sessions · close-to-SMA and SMA-to-SMA
ratios (5/10/20/50) · rolling volatility and a short/long vol ratio · intraday
range, overnight gap, open-to-close, close position in range · relative volume and
volume z-score · optional SPY-relative returns, rolling beta and correlation.

Ratios rather than raw prices: a model trained on a $30 stock should still mean
something at $200.

---

## Metrics and uncertainty

**Primary metric: balanced accuracy.** If 54% of sessions are up, a model that
always says "up" scores 54% plain accuracy and looks informative. Balanced
accuracy averages per-class recall, so that model scores exactly 0.500.

Also reported: accuracy, ROC-AUC, PR-AUC, F1-macro, precision/recall, MCC, log
loss, Brier score, confusion matrix.

**Calibration** uses 10 **fixed-width** bins with every sample assigned exactly
once (`bin_id = min(int(p * 10), 9)`), yielding ECE and MCE. The previous
quantile-based implementation resized a separately-computed histogram and silently
dropped or double-counted samples whenever predictions were concentrated — which,
for this problem, they always are. Bin counts now sum to the sample count exactly,
and a test asserts it.

**Uncertainty** is reported, not implied:

- fold metrics carry a **Student-t 95% interval** (with 5 folds the normal
  approximation is badly wrong in the tails);
- final-test daily returns get a **moving-block bootstrap** (block 20 sessions,
  2000 resamples, seed 42) for mean daily return, annualised return, Sharpe, and
  the paired difference against `always_long_intraday`. Blocks preserve the
  autocorrelation an i.i.d. bootstrap would destroy, which otherwise makes
  intervals far too narrow.

---

## Results

Reproduced from `configs/reproduction/readme_aapl_2026_07.yaml`
(AAPL, 2010-01-01 → 2026-07-01, `end_date` pinned).
Config SHA-256 `ba6f7d67…`, data SHA-256 `03c39ee5…`, run
`20260726T122235Z_ba6f7d67`.

4,097 labelled rows, 29 features. Development: 2010-03-16 → 2023-03-21 (3,277
rows). Final test: 2023-03-23 → 2026-06-29 (819 rows), opened once.

### Stage 1 — development walk-forward (5 expanding folds)

| Candidate | Balanced acc. | ± std | 95% t-interval | ROC-AUC | Log loss |
|---|---|---|---|---|---|
| **logistic** `C=1.0, balanced` | **0.5277** | 0.0261 | **[0.4953, 0.5601]** | 0.5240 | 0.7066 |
| logistic `C=0.1, balanced` | 0.5216 | 0.0263 | [0.4889, 0.5543] | 0.5210 | 0.7040 |
| logistic `C=0.01, none` | 0.5168 | 0.0153 | [0.4979, 0.5358] | 0.5147 | 0.6979 |
| logistic `C=0.01, balanced` | 0.5075 | 0.0237 | [0.4780, 0.5369] | 0.5147 | 0.6987 |
| hgb `lr=0.03, leaves=7` | 0.5067 | 0.0202 | [0.4817, 0.5317] | 0.5091 | 0.7078 |
| logistic `C=0.001, none` | 0.5054 | 0.0109 | [0.4918, 0.5189] | 0.5058 | 0.6939 |
| hgb `lr=0.03, leaves=15` | 0.5018 | 0.0162 | [0.4817, 0.5219] | 0.5017 | 0.7318 |
| hgb `lr=0.1, leaves=7` | 0.5013 | 0.0311 | [0.4627, 0.5400] | 0.5074 | 0.7475 |
| `baseline_majority` | 0.5000 | 0.0000 | — | 0.5000 | 0.6927 |
| `baseline_always_up` | 0.5000 | 0.0000 | — | 0.5000 | 0.6927 |
| `baseline_random` | 0.4865 | 0.0164 | [0.4661, 0.5069] | 0.5000 | 0.6927 |
| `baseline_last_direction` | 0.4769 | 0.0263 | [0.4442, 0.5096] | 0.5231 | 0.6923 |

The gate passed: margin +0.0277, 4/5 fold wins, ROC-AUC 0.5240, MCC 0.0557 →
`edge_detected: true`.

**But read the interval.** The winner's 95% t-interval is [0.4953, 0.5601] — it
**includes 0.50**. The gate is a decision rule, not a significance test, and here
it fired on a result that is not statistically distinguishable from a coin flip.
Every candidate's interval straddles 0.50.

### Stage 2 — final test (scored once)

| Model | Accuracy | Balanced acc. | ROC-AUC | MCC | Log loss | Brier |
|---|---|---|---|---|---|---|
| `model_logistic` | 0.5031 | **0.4982** | 0.4798 | −0.0037 | 0.7050 | 0.2554 |
| `baseline_majority` | 0.5421 | 0.5000 | 0.5000 | 0.0000 | 0.6906 | 0.2487 |
| `baseline_always_up` | 0.5421 | 0.5000 | 0.5000 | 0.0000 | 0.6906 | 0.2487 |
| `baseline_last_direction` | 0.5311 | **0.5278** | 0.4722 | 0.0556 | 0.6938 | 0.2503 |
| `baseline_random` | 0.5006 | 0.4986 | 0.5000 | −0.0028 | 0.6906 | 0.2487 |

**The final test contradicted the gate.** 0.4982 versus the best baseline's
0.5278 — a margin of **−0.0296**. The development edge did not survive.

Note `baseline_last_direction`: *worst* on development (0.4769) and *best* on the
final test (0.5278). The sign of the relationship flipped between the two periods.
That is non-stationarity, and it is why a single split cannot be trusted.

### Backtest, net of 10 bps per round trip

| Strategy | Cumulative | Annualised | Sharpe | Max DD | Exposure | Active sessions | Trades | Cost paid |
|---|---|---|---|---|---|---|---|---|
| `model_logistic` | +1.43% | +0.44% | 0.09 | −13.4% | 12.6% | 103 | 103 | 0.103 |
| `baseline_last_direction` | +7.72% | +2.31% | 0.23 | −17.3% | 54.2% | 444 | 444 | 0.444 |
| `always_long_intraday` | **−8.95%** | −2.84% | −0.02 | −26.5% | 100% | 819 | 819 | **0.819** |
| **`buy_and_hold_close_to_close`** | **+81.17%** | **+20.06%** | **0.84** | −33.4% | 100% | 819 | **1** | 0.0005 |
| `cash` | 0.00% | 0.00% | — | 0.0% | 0% | 0 | 0 | 0 |

### Bootstrap intervals (block 20, 2000 resamples)

| Statistic | Point | 95% CI | Excludes 0 |
|---|---|---|---|
| Mean daily net return | +0.000043 | [−0.000472, +0.000629] | no |
| Annualised return | +0.44% | [−11.46%, +15.60%] | no |
| Sharpe ratio | 0.093 | [−1.860, +0.951] | no |
| vs `always_long_intraday` | +0.000064 | [−0.000693, +0.000725] | no |

**Every interval includes zero.** On this window the strategy is statistically
indistinguishable from being always active, and from doing nothing.

### Robustness across every shipped config

Each row is an independent run with its own sealed holdout. `edge (dev)` is the
development-only gate; `margin` is the model's final-test balanced accuracy minus
the best baseline's.

| Config | Ticker | Selected candidate | edge (dev) | margin | Model return | `always_long_intraday` | Buy & hold | Exposure |
|---|---|---|---|---|---|---|---|---|
| `readme_aapl_2026_07` | AAPL | logistic `C=1.0` | yes | −0.0296 | +1.4% | −8.9% | **+81.2%** | 12.6% |
| `lstm_aapl` | AAPL | logistic `C=1.0` | yes | −0.0308 | +4.5% | −0.1% | **+105.7%** | 12.2% |
| `tuned_thresholds` | AAPL | logistic `C=1.0` | yes | −0.0389 | +16.1% | −0.1% | **+105.7%** | 0.9% |
| `benchmark_features` | AAPL | logistic `C=1.0` | yes | −0.0524 | +16.1% | −0.1% | **+105.7%** | 13.9% |
| `research_close_to_close` | AAPL | logistic `C=1.0` | no | −0.0346 | +14.5% | +103.9% | **+105.7%** | 6.2% |
| `spy` | SPY | hgb `lr=0.03` | yes | −0.0132 | −22.0% | −45.3% | **+90.0%** | 57.3% |
| `msft` | MSFT | logistic `C=0.01` | no | **+0.0364** | **−19.1%** | −56.4% | **+41.5%** | 15.9% |

Three things this table shows that a single run cannot:

- **Six of seven configs fail on the holdout.** The one that does not is MSFT.
- **MSFT is the most instructive row.** It *beats* the baselines on classification
  by +0.0364 — the only positive margin here — and *loses 19.1%* while simply
  holding MSFT returned +41.5%. Better direction calls, substantially worse money.
  This is what "accuracy is not profit" looks like in numbers.
- **Buy-and-hold wins every single row**, by margins between 33 and 114 points.
- **The `edge (dev)` column has no relationship to the `margin` column.** Runs that
  passed the gate span margins from −0.0524 to −0.0132; the two runs that *failed*
  the gate include both the worst and the only positive margin. A development-side
  gate does not predict holdout performance — which is exactly why the holdout is
  sealed rather than consulted.

The `tuned_thresholds` run also shows how threshold tuning can go quietly wrong:
optimising net Sharpe on development pushed the trading threshold so high that the
strategy takes only 7 positions in 822 sessions (0.9% exposure). The return looks
respectable and the sample is far too small to mean anything.

`bootstrap_summary.json` for every run above reports `interval_excludes_zero:
false` on mean daily return. Not one of them is statistically distinguishable from
zero.

### Reading these numbers honestly

- **The gate passed and the holdout disagreed.** This is the single most useful
  thing the project demonstrates. Under the old approach — score the test set for
  every variant, report the best — this would have been written up as a win.
- **Costs dominate.** `always_long_intraday` paid **0.819** in cumulative cost
  (81.9% of starting capital) across 819 round trips and turned a rising market
  into a −8.95% loss. The same asset, simply held, returned +81.17% for 0.0005 in
  cost. Trading frequency, not prediction quality, is the dominant term.
- **Nothing beat holding.** The model returned +1.43% against +81.17%.
- **Low exposure flatters risk metrics.** The model is in cash 87% of the time, so
  its drawdown (−13.4%) looks better than buy-and-hold's (−33.4%) while earning a
  fiftieth of the return. Sharpe and drawdown read without exposure mislead.
- **Fold spread swamps every difference.** Development balanced accuracy ranges
  0.4974–0.5269 across candidates, with per-candidate std up to 0.026. A ±0.005
  gap between models is not a result.
- **The close-to-close variant agrees.** `research_close_to_close.yaml` gives
  0.4754 versus a 0.5100 best baseline (`edge_detected: false`), so the conclusion
  is not an artifact of the executable target.
- **The LSTM competed and lost.** With `include_lstm: true` the LSTM is scored on
  the same development folds as everything else and did not rank first, so logistic
  regression was selected. Per the complexity ordering, the simpler model wins ties
  — and here it won outright.

The honest summary: **no stable predictive edge was found.**

---

## Reproducing the results

The default config uses `end_date: null`, meaning "through the last completed
session" — so it produces different numbers every day and cannot reproduce a
published table. The frozen config pins the end date:

```bash
uv run stock-movement run-all --config configs/reproduction/readme_aapl_2026_07.yaml
```

Every run records what it depended on: SHA-256 of the raw Parquet, the resolved
config and the feature manifest; the git commit, branch and **dirty flag**; the
Python version, OS, CPU/GPU and package versions; and start/finish timestamps. A
result produced from a modified working tree says so in its model card, because it
cannot be reproduced from the recorded commit alone.

Cached data is verified against its digest on every read, so a hand-edited or
corrupted Parquet fails loudly instead of quietly changing a result.

> **A pinned `end_date` is necessary but not sufficient.** Running this config on
> two different days produced two different data digests (`1b9f663a…` then
> `03c39ee5…`) for the same date range: Yahoo Finance returned one extra row and
> slightly different adjusted values. Development balanced accuracy moved 0.5269 →
> 0.5277 and final-test 0.4970 → 0.4982; the backtest figures were identical to
> four decimals and no conclusion changed. Full byte-level reproducibility needs
> the cached Parquet itself, which is why its SHA-256 is recorded in
> `data_manifest.json` and in every model card. Keep `data/raw/` to reproduce a
> published number exactly.

Run IDs are `<UTC timestamp>_<config hash>` (e.g. `20260726T115818Z_ba6f7d67`).
The hash comes from canonical JSON of the resolved config, so it ignores YAML key
order. Run directories are never overwritten.

---

## Model persistence and prediction

The final-test stage saves the fitted model, and **a failed save fails the run** —
a run that reports a metric but leaves nothing usable has not produced a model.

| Format | Files |
|---|---|
| sklearn | `model/model.joblib` (the whole fitted `Pipeline`, imputer and scaler included) |
| Keras/LSTM | `model/model.keras` + `imputer.joblib` + `scaler.joblib` + `lstm_state.joblib` |

`model/model_metadata.json` records the **ordered** feature list, feature version,
both thresholds, target definition, execution mode, training window, config and
data hashes, package versions, and per-file digests. Saving is verified by an
immediate reload.

```bash
uv run stock-movement predict --run-id <RUN_ID> --latest
```

```json
{
  "signal_date": "2026-06-29",
  "expected_execution_date": "2026-06-30",
  "ticker": "AAPL",
  "probability_up": 0.463364,
  "predicted_class": 0,
  "trading_signal": "cash",
  "classification_threshold": 0.5,
  "trading_threshold": 0.55,
  "model_run_id": "20260726T115818Z_ba6f7d67",
  "model_candidate": "logistic[C=1.0,class_weight=balanced]",
  "feature_version": "v2",
  "edge_detected": true,
  "final_test_confirmed_edge": false,
  "warnings": ["the development-only edge gate passed, but the FINAL TEST CONTRADICTED it ..."]
}
```

`final_test_confirmed_edge` exists because `edge_detected` alone would let a
prediction imply a working model. When the holdout contradicted the gate, the
prediction says so.

Inference refuses rather than guesses when: the run has no saved model; the feature
version differs from the current code; a manifest feature cannot be built; the
manifest and model disagree; the requested date is not a completed session; or any
feature is NaN or infinite. Features are reindexed to the manifest's exact order —
a permuted column order would otherwise produce confident nonsense.

---

## Limitations

1. **No demonstrated edge.** The headline result is negative and every bootstrap
   interval includes zero. Nothing here should be traded.
2. **Execution.** `next_open` is executable in principle but assumes fills at the
   official open and close and a flat cost per round trip. Real slippage varies
   with size and volatility.
3. **One ticker, one history.** A single asset over a single period is one sample.
   `spy.yaml` and `msft.yaml` exist for this reason.
4. **Non-stationarity.** `baseline_last_direction` went from worst on development
   to best on the final test. Relationships invert.
5. **Small differences are noise.** With 819 test rows the standard error on
   accuracy is about ±1.7 points.
6. **Accuracy is not profit.** Accuracy weights every session equally; returns do
   not.
7. **Backtest profit is not skill.** Long-biased exposure to a rising market
   produces profit with no predictive content, which is why always-long and
   buy-and-hold are always shown.
8. **Multiple comparisons.** Eight candidates were evaluated on development data.
   Grids are small, but the risk is not zero — and it is why the holdout is locked.
9. **Vendor data.** Prices are revised in practice — the same pinned date range
   returned a different digest on two consecutive days (see
   [Reproducing the results](#reproducing-the-results)). Survivorship and delisting
   effects are not modelled. Adjusted OHLC also carries float64 rounding: on SPY,
   `High` came out 1.17e-16 below `Close` on a session that closed at its high,
   which is one unit in the last place, so the OHLC consistency checks use an
   explicit relative tolerance (1e-8) rather than exact comparison.
10. **No macro, fundamental, news, or order-book information.**
11. **The published results were produced from a dirty working tree**, as the model
    card records. Re-run from a clean commit for a fully reproducible artifact.

---

## Adding a new ticker

```bash
uv run stock-movement run-all --config configs/default.yaml --ticker NVDA
```

Or copy a config, set `data.ticker` and `run_name`, and run it. Configs inherit
via `extends:`; list-valued fields are replaced wholesale rather than merged, so
switching model families does not drag along the parent's settings.

Check the run's `data_manifest.json` for validation warnings — a short history or
a long calendar gap affects the rolling windows, and `min_rows` (1000 *after*
warm-up) must still be satisfiable.

---

## Testing and quality gates

```bash
make quality
```

runs exactly what CI runs:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not slow" --cov=stock_movement --cov-fail-under=90
```

Current state: **443 tests passing (447 with the LSTM extra), 95% coverage, ruff
clean, mypy strict clean** across 25 modules. No test touches the network —
everything runs on synthetic data with a mocked clock, so the guarantees hold
deterministically.

The suite passes both **with and without** the optional `lstm` extra. Tests that
depend on TensorFlow's absence simulate it rather than branching on whether it
happens to be installed, so neither environment can quietly skip an assertion.

The `slow` marker covers TensorFlow training and is excluded from the fast gate;
sequence-construction tests are not marked slow and always run. GitHub Actions
runs the gate on every push and pull request, with the LSTM job on demand.

```bash
uv run pytest -m slow      # LSTM phase (needs the lstm extra)
```

---

## Project layout

```
stock-movement-predictor/
├── configs/
│   ├── default.yaml              # the executable setup
│   ├── experiments/              # variants, via `extends:`
│   └── reproduction/             # frozen configs with pinned end dates
├── src/stock_movement/
│   ├── config.py                 # pydantic schema, cross-field validation, hashing
│   ├── data.py                   # download, normalise, cache with verified digest
│   ├── market_calendar.py        # partial-session logic via exchange_calendars
│   ├── validation.py             # impossible bars fail; anomalies are recorded
│   ├── features.py               # pure causal feature functions
│   ├── labels.py                 # the only forward-looking module
│   ├── dataset.py                # assembly, benchmark inner join, contract checks
│   ├── split.py                  # chronological holdout and walk-forward
│   ├── baselines.py              # the four baselines
│   ├── models.py                 # factory, complexity, simplicity tie-breaks
│   ├── selection.py              # development-only selection and the edge gate
│   ├── evaluation.py             # metrics with explicit hard predictions
│   ├── calibration.py            # fixed-width bins, ECE/MCE, threshold selection
│   ├── backtest.py               # two cost models, net trade metrics
│   ├── statistics.py             # t-intervals and moving-block bootstrap
│   ├── persistence.py            # model saving; failures fail the run
│   ├── inference.py              # predict from a saved model, or refuse
│   ├── provenance.py             # hashes, git state, environment
│   ├── artifacts.py              # immutable runs, final-test lock
│   ├── report.py                 # model card
│   ├── plots.py                  # figures
│   ├── pipeline.py               # the two stages
│   ├── lstm.py                   # optional Keras classifier
│   └── cli.py                    # command line
├── tests/                        # 443 tests, no network access
├── notebooks/                    # EDA, features, costs, run comparison
└── artifacts/runs/<run_id>/      # immutable per-run output
```

Each run writes `resolved_config.{yaml,json}`, `environment.json`,
`data_manifest.json`, `feature_manifest.json`, `split_manifest.json`,
`candidate_summary.csv`, `candidate_fold_metrics.csv`,
`candidate_oof_predictions.parquet`, `selection_decision.json`,
`selected_model_spec.json`, `final_test.lock.json`, `final_test_metrics.json`,
`final_test_predictions.parquet`, `backtest_metrics.json`,
`backtest_daily.parquet`, `statistical_summary.json`, `bootstrap_summary.json`,
`model_card.md`, `model/`, and nine figures.

---

## License

MIT — see [LICENSE](LICENSE).
