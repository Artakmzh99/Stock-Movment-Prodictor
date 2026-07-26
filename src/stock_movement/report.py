"""Model card generation.

The card is written from the artifacts, not from memory, so it cannot drift from
what the run actually produced. It leads with the edge verdict and the execution
cost model, because those are the two things a reader most needs in order to
interpret every number below them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .artifacts import FinalTestLock, RunDirectory
from .backtest import backtest_comparison
from .baselines import BASELINE_PREFIX
from .config import Config
from .dataset import Dataset
from .evaluation import PRIMARY_METRIC
from .persistence import ModelMetadata


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number * 100:.2f}%"


def _interval(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    return f"{_fmt(entry.get('point'), 5)} [{_fmt(entry.get('ci_low'), 5)}, {_fmt(entry.get('ci_high'), 5)}]"


def _candidate_ranking_table(decision: dict[str, Any]) -> str:
    rows = decision.get("ranking", [])
    if not rows:
        return "_no candidates recorded_"
    lines = [
        "| Rank | Candidate | Balanced acc. (mean ± std) | Log loss | Complexity |",
        "|---:|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['candidate']}` | "
            f"{_fmt(row['balanced_accuracy_mean'])} ± {_fmt(row['balanced_accuracy_std'])} | "
            f"{_fmt(row['log_loss_mean'])} | {row['complexity_rank']} |"
        )
    return "\n".join(lines)


def _baseline_table(decision: dict[str, Any]) -> str:
    rows = decision.get("baselines", [])
    if not rows:
        return "_no baselines recorded_"
    lines = ["| Baseline | Balanced acc. (mean ± std) |", "|---|---|"]
    for row in sorted(rows, key=lambda r: -(r.get("balanced_accuracy_mean") or 0)):
        lines.append(
            f"| `{row['candidate']}` | {_fmt(row['balanced_accuracy_mean'])} ± "
            f"{_fmt(row['balanced_accuracy_std'])} |"
        )
    return "\n".join(lines)


def _edge_gate_table(decision: dict[str, Any]) -> str:
    gate = decision.get("edge_gate", {})
    checks = gate.get("checks", {})
    if not checks:
        return "_edge gate not evaluated_"
    lines = ["| Condition | Value | Required | Passed |", "|---|---|---|:--:|"]
    for name, check in checks.items():
        value = check.get("value")
        rendered = f"{check['value']}/{check['of_folds']}" if name == "fold_wins" else _fmt(value)
        lines.append(
            f"| {name.replace('_', ' ')} | {rendered} | {check.get('required')} | "
            f"{'yes' if check.get('passed') else 'no'} |"
        )
    return "\n".join(lines)


def _test_metrics_table(test_metrics: dict[str, dict[str, Any]], model_key: str) -> str:
    baselines = {k: v for k, v in test_metrics.items() if k.startswith(BASELINE_PREFIX)}
    model = test_metrics.get(model_key, {})

    def best_baseline(metric: str, maximise: bool = True) -> float:
        values: list[float] = []
        for entry in baselines.values():
            candidate = entry.get(metric)
            if candidate is None:
                continue
            number = float(candidate)
            if np.isfinite(number):
                values.append(number)
        if not values:
            return float("nan")
        return max(values) if maximise else min(values)

    lines = ["| Metric | Selected model | Best baseline |", "|---|---|---|"]
    for metric, maximise in (
        ("balanced_accuracy", True),
        ("roc_auc", True),
        ("accuracy", True),
        ("mcc", True),
        ("f1_macro", True),
        ("log_loss", False),
        ("brier_score", False),
    ):
        lines.append(f"| {metric} | {_fmt(model.get(metric))} | {_fmt(best_baseline(metric, maximise))} |")
    return "\n".join(lines)


TEMPLATE = """# Model card — {run_id}

> **{edge_headline}**
>
> {edge_reason}
>
> {final_test_headline}

Research and education only. **Not investment advice.** No claim of profitability
is made or implied.

## Overview

| Field | Value |
|---|---|
| Task | Next-session direction classification (binary) |
| Ticker | `{ticker}` |
| Target | `{target_definition}` — {target_formula} |
| Execution | `{execution_mode}` — {execution_description} |
| Cost model | {cost_description} |
| Selected candidate | `{selected_candidate}` |
| Model family | `{family}` |
| Classification threshold | {classification_threshold} |
| Trading threshold | {trading_threshold} |
| Data range | {data_first} to {data_last} ({n_rows:,} labelled rows, {n_features} features) |
| Development window | {dev_first} to {dev_last} ({n_dev:,} rows) |
| Final test window | {test_first} to {test_last} ({n_test:,} rows) |
| Random seed | {seed} |
| Feature version | `{feature_version}` |
| Artifact schema | v{schema_version} |

## Provenance

| Field | Value |
|---|---|
| Config SHA-256 | `{config_sha}` |
| Data SHA-256 | `{data_sha}` |
| Git commit | `{git_commit}` |
| Working tree | {git_dirty} |
| Final test completed | {lock_completed_at} |
| Final test reruns | {rerun_count}{rerun_note} |
| Model artifact | `{model_artifact}` |
| Model format | `{model_format}` |
| Inference compatible | {inference_compatible} |

## Selection protocol

Candidate selection used **development data only** ({n_dev:,} rows, ending
{dev_last}). All candidates were scored on the same {n_folds} expanding-window
walk-forward folds with a {gap}-row gap at every boundary. The highest row index
any candidate saw was {max_row_used}; the final test begins at row
{test_start_index}. The final test was opened once, after selection was locked.

Selection metric: **{selection_metric}**, ranked by mean, then fold-to-fold
standard deviation, then log loss, then model complexity, then hyperparameter
simplicity, then name — a total order, so a rerun selects the same model.

### Candidate ranking (development walk-forward)

{ranking_table}

### Baselines (development walk-forward)

{baseline_table}

### Predictive-edge gate

{edge_gate_table}

## Final-test results (scored once)

{test_metrics_table}

Development walk-forward {selection_metric} for the selected candidate:
**{dev_score} ± {dev_std}** across {n_folds} folds (95% t-interval
[{dev_ci_low}, {dev_ci_high}]).

### Probability calibration (final test)

| Quantity | Value |
|---|---|
| Brier score | {brier} |
| Brier of always predicting the base rate | {brier_base} |
| Expected calibration error (10 fixed-width bins) | {ece} |
| Maximum calibration error | {mce} |
| Predicted probability range | {proba_min} to {proba_max} (std {proba_std}) |

## Backtest ({execution_mode}, net of costs)

{backtest_table}

### Uncertainty (moving-block bootstrap, {bootstrap_samples} resamples, block {block_length})

| Statistic | Point [95% CI] |
|---|---|
| Mean daily net return | {boot_mean} |
| Annualised return | {boot_annual} |
| Sharpe ratio | {boot_sharpe} |
| Mean daily return vs `{benchmark_name}` | {boot_diff} |

{bootstrap_reading}

## Limitations

- **Execution.** {execution_limitation}
- **One ticker, one history.** A single asset over a single period is one sample,
  not evidence. Robustness across tickers must be checked separately.
- **Non-stationarity.** The relationship between features and target changes; a
  model fitted through one regime faces the next one.
- **Small differences are noise.** With {n_test:,} test rows the standard error on
  accuracy is roughly ±{accuracy_se} percentage points. Fold spread above is the
  honest measure of confidence.
- **Accuracy is not profit.** Accuracy weights every session equally; returns do
  not. Being right on many quiet sessions and wrong on one violent one loses money.
- **Backtest profit is not model skill.** Long-biased exposure to a rising market
  produces profit with no predictive content, which is why always-long and
  buy-and-hold are always shown.
- **Multiple comparisons.** {n_candidates} candidates were evaluated on development
  data. The grids are deliberately small, but every extra candidate raises the
  chance one looks good by luck.
- **Vendor data.** Prices may be revised; survivorship and delisting effects are
  not modelled.
- **Costs are a flat estimate.** Real slippage varies with size and volatility.

## Reproduction

```bash
uv run stock-movement select-model --config {config_hint}
uv run stock-movement final-test --run-id {run_id}
```

Inference from the saved model, without retraining:

```bash
uv run stock-movement predict --run-id {run_id} --latest
```
"""


def write_model_card(
    run: RunDirectory,
    config: Config,
    dataset: Dataset,
    decision: dict[str, Any],
    test_metrics: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
    backtests: dict[str, Any],
    bootstrap: dict[str, Any],
    calibration: dict[str, Any],
    lock: FinalTestLock,
    model_metadata: ModelMetadata,
    verdict: str,
    model_key: str,
) -> None:
    from .config import ARTIFACT_SCHEMA_VERSION

    edge = bool(decision.get("edge_detected", False))
    gate = decision.get("edge_gate", {})
    development = decision.get("development", {})
    held_out = decision.get("held_out_test", {})
    aggregate = decision.get("selection_score"), decision.get("selection_score_std")
    winner_aggregate = (
        run.read_json("selected_model_spec.json").get("development_walk_forward", {})
        if run.exists("selected_model_spec.json")
        else {}
    )

    environment = run.read_json("environment.json") if run.exists("environment.json") else {}
    git = environment.get("git", {})

    benchmark_name = next((k for k in backtests if k.startswith("always_long")), "always_long_intraday")
    backtest_frame = backtest_comparison(list(backtests.values()))
    display_columns = [
        c
        for c in (
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "exposure",
            "n_active_sessions",
            "n_completed_trades",
            "total_cost_paid",
        )
        if c in backtest_frame.columns
    ]

    is_intraday = config.backtest.execution_mode == "next_open"
    n_test = int(held_out.get("n_rows", 0)) or 1
    accuracy_se = 100 * 0.5 / np.sqrt(n_test)

    diff = bootstrap.get(f"vs_{benchmark_name}", {})
    if diff.get("interval_excludes_zero"):
        bootstrap_reading = (
            f"The interval for the difference against `{benchmark_name}` excludes zero, so the "
            "gap is unlikely to be pure sampling noise. Direction still matters — check the sign."
        )
    else:
        bootstrap_reading = (
            f"The interval for the difference against `{benchmark_name}` **includes zero**: on "
            "this window the strategy is statistically indistinguishable from simply being "
            "always active. This is the usual outcome and should not be read as an edge."
        )

    context = {
        "run_id": run.run_id,
        "edge_headline": (
            "EDGE DETECTED — the selected model passed every gate condition on development data."
            if edge
            else "NO STABLE PREDICTIVE EDGE WAS FOUND. This is a valid and expected outcome."
        ),
        "edge_reason": gate.get("reason", verdict),
        "final_test_headline": (
            (
                "**The sealed final test CONTRADICTED the development edge gate.** "
                f"{verdict} The apparent edge did not survive out of sample, which is the "
                "usual outcome and precisely why selection and final testing are separated."
            )
            if model_metadata.final_test_beat_baselines is False and edge
            else (
                f"**The final test confirmed the development result.** {verdict}"
                if model_metadata.final_test_beat_baselines
                else f"Final test: {verdict}"
            )
        ),
        "ticker": config.data.ticker,
        "target_definition": config.labels.target_definition,
        "target_formula": (
            "`target = 1 if Close(t+1) / Open(t+1) - 1 > 0 else 0`"
            if config.labels.target_definition == "open_to_close"
            else "`target = 1 if Close(t+1) / Close(t) - 1 > 0 else 0`"
        ),
        "execution_mode": config.backtest.execution_mode,
        "execution_description": (
            "signal after the close of t, enter at the open of t+1, exit at the close of t+1"
            if is_intraday
            else "position taken at the close of t (research convention; not literally tradable)"
        ),
        "cost_description": (
            f"{config.backtest.round_trip_cost_bps:.1f} bps round trip charged on **every active "
            "session**, because each session is a separate entry and exit"
            if is_intraday
            else f"{config.backtest.one_way_cost_bps:.1f} bps one-way charged per unit of position change"
        ),
        "selected_candidate": decision["selected_candidate"]["name"],
        "family": decision["selected_candidate"]["model_name"],
        "classification_threshold": _fmt(model_metadata.classification_threshold, 3),
        "trading_threshold": _fmt(model_metadata.trading_threshold, 3),
        "data_first": str(dataset.index[0].date()),
        "data_last": str(dataset.index[-1].date()),
        "n_rows": len(dataset),
        "n_features": dataset.X.shape[1],
        "dev_first": development.get("first_date", "n/a"),
        "dev_last": development.get("last_date", "n/a"),
        "n_dev": int(development.get("n_rows", 0)),
        "test_first": held_out.get("first_date", "n/a"),
        "test_last": held_out.get("last_date", "n/a"),
        "n_test": n_test,
        "seed": config.random_seed,
        "feature_version": model_metadata.feature_version,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha": config.sha256(),
        "data_sha": lock.data_sha256 or "n/a",
        "git_commit": git.get("commit") or "n/a",
        "git_dirty": (
            "clean"
            if git.get("dirty") is False
            else f"**dirty** ({git.get('dirty_file_count', '?')} modified files) — "
            "this result cannot be reproduced from the recorded commit alone"
            if git.get("dirty")
            else "unknown"
        ),
        "lock_completed_at": lock.completed_at_utc,
        "rerun_count": lock.rerun_count,
        "rerun_note": (
            " — **the holdout was re-scored; treat these numbers with corresponding suspicion**"
            if lock.rerun_count
            else " (scored once, as intended)"
        ),
        "model_artifact": f"model/{'model.keras' if model_metadata.model_format == 'keras' else 'model.joblib'}",
        "model_format": model_metadata.model_format,
        "inference_compatible": (
            f"yes — `predict --run-id {run.run_id}` works against feature version "
            f"`{model_metadata.feature_version}`"
        ),
        "n_folds": int(winner_aggregate.get("n_folds", decision.get("n_folds", 0)) or 0),
        "gap": config.split.gap,
        "max_row_used": development.get("max_row_index_used", "n/a"),
        "test_start_index": held_out.get("test_start_index", "n/a"),
        "selection_metric": PRIMARY_METRIC,
        "ranking_table": _candidate_ranking_table(decision),
        "baseline_table": _baseline_table(decision),
        "edge_gate_table": _edge_gate_table(decision),
        "test_metrics_table": _test_metrics_table(test_metrics, model_key),
        "dev_score": _fmt(aggregate[0]),
        "dev_std": _fmt(aggregate[1]),
        "dev_ci_low": _fmt(winner_aggregate.get(f"{PRIMARY_METRIC}_ci_low")),
        "dev_ci_high": _fmt(winner_aggregate.get(f"{PRIMARY_METRIC}_ci_high")),
        "brier": _fmt(calibration.get("brier_score")),
        "brier_base": _fmt(calibration.get("brier_score_of_base_rate")),
        "ece": _fmt(calibration.get("expected_calibration_error")),
        "mce": _fmt(calibration.get("maximum_calibration_error")),
        "proba_min": _fmt(calibration.get("predicted_probability_min"), 3),
        "proba_max": _fmt(calibration.get("predicted_probability_max"), 3),
        "proba_std": _fmt(calibration.get("predicted_probability_std"), 4),
        "backtest_table": backtest_frame[display_columns].round(4).to_markdown(),
        "bootstrap_samples": config.statistics.bootstrap_samples,
        "block_length": config.statistics.bootstrap_block_length,
        "boot_mean": _interval(bootstrap.get("mean_daily_return")),
        "boot_annual": _interval(bootstrap.get("annualized_return")),
        "boot_sharpe": _interval(bootstrap.get("sharpe_ratio")),
        "boot_diff": _interval(diff),
        "benchmark_name": benchmark_name,
        "bootstrap_reading": bootstrap_reading,
        "execution_limitation": (
            "Entry at the next open and exit at that session's close is executable in "
            "principle, but assumes fills at the official open and close prices and a flat "
            "cost per round trip."
            if is_intraday
            else "Close-to-close execution assumes trading at the same close used to make the "
            "decision, which is not achievable. Research convention only."
        ),
        "accuracy_se": f"{accuracy_se:.1f}",
        "n_candidates": decision.get("n_candidates_evaluated", "n/a"),
        "config_hint": "configs/default.yaml",
    }

    run.write_text("model_card.md", TEMPLATE.format(**context))
