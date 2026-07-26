"""Two-stage orchestration.

**Stage 1 — ``run_selection_stage``.** Builds the dataset, evaluates every
candidate and baseline on shared development folds, tunes thresholds on
out-of-fold *development* predictions, and writes the selection decision. It never
computes a single final-test metric.

**Stage 2 — ``run_final_test_stage``.** Loads the locked decision, verifies it was
made without the holdout, refits the winner on all development rows, scores the
holdout **once**, backtests under the configured execution model, bootstraps
uncertainty, saves the fitted model, and writes the lock.

Splitting them is the point: stage 1 cannot see the test set, and stage 2 cannot
change its mind about the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import plots
from .artifacts import (
    FinalTestLock,
    RunDirectory,
    create_run,
    finalize_environment,
    open_run,
    write_common_artifacts,
)
from .backtest import (
    ALWAYS_LONG_INTRADAY,
    BacktestResult,
    always_active,
    backtest_comparison,
    buy_and_hold_close_to_close,
    cash,
    positions_from_probability,
    run_backtest,
)
from .baselines import BASELINE_PREFIX, build_baselines
from .calibration import calibration_report, select_threshold
from .config import Config, set_global_seed
from .dataset import Dataset, build_dataset
from .evaluation import (
    PRIMARY_METRIC,
    classification_metrics,
    confusion_frame,
    metrics_to_frame,
    summarize_comparison,
)
from .models import build_model, extract_feature_importance
from .persistence import build_metadata_fields, save_model
from .provenance import utc_now_iso
from .selection import (
    SelectionOutcome,
    development_bounds,
    run_selection,
)
from .statistics import block_bootstrap_difference, block_bootstrap_returns

logger = logging.getLogger("stock_movement")

SELECTION_DECISION = "selection_decision.json"
SELECTED_MODEL_SPEC = "selected_model_spec.json"


@dataclass
class SelectionStageOutput:
    run: RunDirectory
    dataset: Dataset
    outcome: SelectionOutcome
    thresholds: dict[str, float]


@dataclass
class FinalTestOutput:
    run: RunDirectory
    dataset: Dataset
    test_metrics: dict[str, dict[str, Any]]
    comparison: pd.DataFrame
    backtests: dict[str, BacktestResult]
    lock: FinalTestLock
    verdict: str
    edge_detected: bool
    selected_candidate: str


# --------------------------------------------------------------------------
# stage 1: selection
# --------------------------------------------------------------------------
def _tune_thresholds(
    outcome: SelectionOutcome, config: Config
) -> tuple[dict[str, float], pd.DataFrame | None]:
    """Tune thresholds on the winner's **out-of-fold development** predictions.

    Out-of-fold predictions within development are genuinely out-of-sample for the
    model that produced them, and contain no holdout rows — so this is a legitimate
    place to choose an operating point.
    """
    thresholds = {
        "classification_value": config.threshold.classification_value,
        "trading_value": config.threshold.trading_value,
    }
    oof = outcome.winner.oof_predictions
    if oof.empty:
        return thresholds, None

    scan: pd.DataFrame | None = None

    if config.threshold.tune_classification_threshold:
        chosen, scan = select_threshold(
            oof["target"],
            oof["proba_up"],
            candidates=config.threshold.candidates,
            objective=config.threshold.objective,
        )
        thresholds["classification_value"] = chosen

    if config.threshold.tune_trading_threshold:
        chosen, trading_scan = select_threshold(
            oof["target"],
            oof["proba_up"],
            candidates=config.threshold.candidates,
            objective="net_sharpe",
            future_return=oof["future_return"],
            cost_rate=(
                config.backtest.round_trip_cost_rate
                if config.backtest.execution_mode == "next_open"
                else config.backtest.one_way_cost_rate
            ),
            execution_mode=config.backtest.execution_mode,
            trading_days=config.backtest.trading_days_per_year,
        )
        thresholds["trading_value"] = chosen
        scan = trading_scan if scan is None else scan

    return thresholds, scan


def run_selection_stage(
    config: Config,
    force_refresh: bool = False,
    run_id: str | None = None,
) -> SelectionStageOutput:
    """Stage 1. Writes no final-test metric of any kind."""
    run_started = utc_now_iso()
    set_global_seed(config.random_seed)

    logger.info("building dataset for %s (%s)", config.data.ticker, config.labels.target_definition)
    dataset = build_dataset(config, force_refresh=force_refresh)
    logger.info("dataset ready: %s", dataset.summary())

    n_development, test_start = development_bounds(dataset, config)
    logger.info(
        "development rows 0-%d (%s..%s); holdout %d rows from %s — sealed until final-test",
        n_development - 1,
        dataset.index[0].date(),
        dataset.index[n_development - 1].date(),
        len(dataset) - test_start,
        dataset.index[test_start].date(),
    )

    outcome = run_selection(dataset, config)
    thresholds, scan = _tune_thresholds(outcome, config)
    outcome.decision["thresholds"] = thresholds
    outcome.decision["thresholds_tuned_on"] = "winner out-of-fold development predictions"

    logger.info(
        "selected %s (%s = %.4f +/- %.4f); edge_detected=%s",
        outcome.winner.name,
        PRIMARY_METRIC,
        outcome.winner.mean(PRIMARY_METRIC),
        outcome.winner.std(PRIMARY_METRIC),
        outcome.edge.edge_detected,
    )
    logger.info("%s", outcome.edge.reason)

    run = create_run(config, run_id=run_id, must_be_new=True)
    write_common_artifacts(run, config, dataset.metadata, dataset.summary(), dataset.manifest, run_started)

    run.write_json(SELECTION_DECISION, outcome.decision)
    run.write_json(
        SELECTED_MODEL_SPEC,
        {
            **outcome.winner.spec.to_dict(),
            "thresholds": thresholds,
            "development_walk_forward": outcome.winner.aggregate,
            "edge_detected": outcome.edge.edge_detected,
        },
    )
    run.write_csv("candidate_summary.csv", outcome.summary_frame, index=False)
    run.write_csv("candidate_fold_metrics.csv", outcome.fold_metrics_frame, index=False)
    run.write_parquet("candidate_oof_predictions.parquet", outcome.oof_predictions)
    run.write_json(
        "split_manifest.json",
        {
            "n_rows": len(dataset),
            "development_end_index": n_development,
            "test_start_index": test_start,
            "gap_rows": config.split.gap,
            "development_first_date": str(dataset.index[0].date()),
            "development_last_date": str(dataset.index[n_development - 1].date()),
            "test_first_date": str(dataset.index[test_start].date()),
            "test_last_date": str(dataset.index[-1].date()),
            "walk_forward_folds": [f.dates(dataset.index) for f in outcome.folds],
        },
    )
    run.write_json(
        "statistical_summary.json",
        {
            "development_walk_forward": {r.name: r.aggregate for r in outcome.ranked + outcome.baselines},
            "note": (
                "Confidence intervals are Student-t over walk-forward folds on development "
                "data. Final-test bootstrap intervals are written by the final-test stage."
            ),
        },
    )
    if scan is not None:
        run.write_csv("threshold_scan.csv", scan, index=False)
        plots.plot_threshold_scan(
            scan,
            thresholds["trading_value"]
            if config.threshold.tune_trading_threshold
            else thresholds["classification_value"],
            run.figure_path("threshold_scan.png"),
            objective=config.threshold.objective,
        )

    plots.plot_class_balance(dataset.y, run.figure_path("class_balance.png"))
    plots.plot_fold_balanced_accuracy(
        outcome.fold_metrics_frame, run.figure_path("fold_balanced_accuracy.png"), winner=outcome.winner.name
    )

    # Invariant, not a comment: nothing test-shaped may exist after stage 1.
    for forbidden in ("final_test_metrics.json", "final_test_predictions.parquet", "backtest_metrics.json"):
        if run.exists(forbidden):  # pragma: no cover - defensive
            raise AssertionError(f"selection stage wrote {forbidden}, which it must never do")

    logger.info("selection artifacts written to %s", run.path)
    return SelectionStageOutput(run=run, dataset=dataset, outcome=outcome, thresholds=thresholds)


# --------------------------------------------------------------------------
# stage 2: final test
# --------------------------------------------------------------------------
def _verify_selection_decision(decision: dict[str, Any], config: Config) -> None:
    if not decision.get("selected_candidate"):
        raise ValueError(f"{SELECTION_DECISION} has no selected_candidate; run select-model first")
    if decision.get("test_data_used_for_selection") is not False:
        raise ValueError(
            f"{SELECTION_DECISION} does not certify development-only selection; refusing to "
            "score the holdout against a decision that may have seen it"
        )
    recorded = decision.get("config_sha256")
    if recorded and recorded != config.sha256():
        raise ValueError(
            "config hash mismatch between the selection decision and the current config:\n"
            f"  selection: {recorded}\n  current:   {config.sha256()}\n"
            "The holdout must be scored against the configuration selection was performed under."
        )


def _score_baselines_on_test(
    dataset: Dataset,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}

    X_dev, y_dev = dataset.X.iloc[development_idx], dataset.y.iloc[development_idx]
    X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]

    baselines = build_baselines(
        random_state=config.random_seed,
        target_definition=config.labels.target_definition,
        include_random=config.selection.include_random_baseline,
    )
    for name, estimator in baselines.items():
        fitted = estimator.fit(X_dev, y_dev)
        proba = np.asarray(fitted.predict_proba(X_test))[:, 1]
        hard = np.asarray(fitted.predict(X_test)).astype(int)

        key = f"{BASELINE_PREFIX}{name}"
        metrics[key] = classification_metrics(y_test, proba, y_pred=hard)
        predictions[key] = pd.DataFrame(
            {
                "proba_up": proba,
                "prediction": hard,
                "target": y_test.to_numpy(),
                "future_return": dataset.future_return.iloc[test_idx].to_numpy(),
            },
            index=X_test.index,
        )
    return metrics, predictions


def _run_backtests(
    model_predictions: pd.DataFrame,
    baseline_predictions: dict[str, pd.DataFrame],
    dataset: Dataset,
    config: Config,
    trading_threshold: float,
    model_key: str,
) -> dict[str, BacktestResult]:
    future_return = model_predictions["future_return"]
    results: dict[str, BacktestResult] = {}

    position = positions_from_probability(
        model_predictions["proba_up"],
        threshold=trading_threshold,
        allow_short=config.backtest.allow_short,
        short_threshold=config.backtest.short_threshold,
    )
    results[model_key] = run_backtest(position, future_return, config.backtest, name=model_key)

    # Baselines trade their own hard rule, which is what they actually assert.
    for name in (f"{BASELINE_PREFIX}last_direction", f"{BASELINE_PREFIX}always_up"):
        if name not in baseline_predictions:
            continue
        rule = baseline_predictions[name]["prediction"].astype(float).clip(0, 1)
        results[name] = run_backtest(rule, future_return, config.backtest, name=name)

    results[ALWAYS_LONG_INTRADAY if config.backtest.execution_mode == "next_open" else "always_long"] = (
        always_active(future_return, config.backtest)
    )
    # Always show genuine buy-and-hold: "would holding have done better?" is the
    # question a reader actually has.
    results["buy_and_hold_close_to_close"] = buy_and_hold_close_to_close(
        dataset.prices["Close"].astype(float), config.backtest, index=future_return.index
    )
    results["cash"] = cash(future_return, config.backtest)
    return results


def run_final_test_stage(
    config: Config,
    run_id: str,
    allow_rerun: bool = False,
    rerun_reason: str | None = None,
    force_refresh: bool = False,
) -> FinalTestOutput:
    """Stage 2. Opens the holdout exactly once, then locks it."""
    set_global_seed(config.random_seed)
    run = open_run(config, run_id)

    decision = run.read_json(SELECTION_DECISION)
    _verify_selection_decision(decision, config)
    previous_lock = run.assert_final_test_allowed(allow_rerun=allow_rerun, rerun_reason=rerun_reason)

    spec = decision["selected_candidate"]
    thresholds = decision.get("thresholds", {})
    classification_threshold = float(
        thresholds.get("classification_value", config.threshold.classification_value)
    )
    trading_threshold = float(thresholds.get("trading_value", config.threshold.trading_value))

    dataset = build_dataset(config, force_refresh=force_refresh)
    n_development, test_start = development_bounds(dataset, config)
    development_idx = np.arange(0, n_development)
    test_idx = np.arange(test_start, len(dataset))

    logger.info(
        "final test: refitting %s on %d development rows, scoring %d holdout rows (%s..%s)",
        spec["name"],
        len(development_idx),
        len(test_idx),
        dataset.index[test_idx[0]].date(),
        dataset.index[test_idx[-1]].date(),
    )

    estimator = build_model(spec["model_name"], spec["params"], random_state=config.random_seed)
    estimator.fit(dataset.X.iloc[development_idx], dataset.y.iloc[development_idx])

    X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]
    proba = np.asarray(estimator.predict_proba(X_test))
    proba_up = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
    hard = (proba_up >= classification_threshold).astype(int)

    model_key = f"model_{spec['model_name']}"
    model_predictions = pd.DataFrame(
        {
            "proba_up": proba_up,
            "prediction": hard,
            "target": y_test.to_numpy(),
            "future_return": dataset.future_return.iloc[test_idx].to_numpy(),
        },
        index=X_test.index,
    )

    test_metrics: dict[str, dict[str, Any]] = {
        model_key: classification_metrics(y_test, proba_up, y_pred=hard, threshold=classification_threshold)
    }
    baseline_metrics, baseline_predictions = _score_baselines_on_test(
        dataset, development_idx, test_idx, config
    )
    test_metrics.update(baseline_metrics)

    comparison = metrics_to_frame(test_metrics)
    verdict = summarize_comparison(comparison, PRIMARY_METRIC)
    logger.info("verdict: %s", verdict)

    backtests = _run_backtests(
        model_predictions, baseline_predictions, dataset, config, trading_threshold, model_key
    )

    # -- uncertainty -------------------------------------------------------
    strategy_returns = backtests[model_key].returns
    benchmark_key = ALWAYS_LONG_INTRADAY if ALWAYS_LONG_INTRADAY in backtests else "always_long"
    bootstrap = {
        name: result.to_dict()
        for name, result in block_bootstrap_returns(
            strategy_returns,
            n_samples=config.statistics.bootstrap_samples,
            block_length=config.statistics.bootstrap_block_length,
            seed=config.statistics.bootstrap_seed,
            confidence_level=config.statistics.confidence_level,
            trading_days_per_year=config.backtest.trading_days_per_year,
        ).items()
    }
    bootstrap["vs_" + benchmark_key] = block_bootstrap_difference(
        strategy_returns,
        backtests[benchmark_key].returns,
        n_samples=config.statistics.bootstrap_samples,
        block_length=config.statistics.bootstrap_block_length,
        seed=config.statistics.bootstrap_seed,
        confidence_level=config.statistics.confidence_level,
    ).to_dict()

    calibration = {
        name: calibration_report(frame["target"], frame["proba_up"])
        for name, frame in ({model_key: model_predictions} | baseline_predictions).items()
    }

    # -- persist -----------------------------------------------------------
    run.write_json("final_test_metrics.json", test_metrics)
    run.write_csv("final_test_comparison.csv", comparison)
    run.write_parquet("final_test_predictions.parquet", model_predictions)
    run.write_json(
        "backtest_metrics.json",
        {name: result.metrics for name, result in backtests.items()},
    )
    run.write_csv("backtest_comparison.csv", backtest_comparison(list(backtests.values())))
    run.write_parquet(
        "backtest_daily.parquet",
        pd.concat({name: r.to_frame() for name, r in backtests.items()}, axis=1),
    )
    run.write_json("bootstrap_summary.json", bootstrap)
    run.write_json("calibration.json", calibration)

    statistical = run.read_json("statistical_summary.json") if run.exists("statistical_summary.json") else {}
    statistical["final_test_bootstrap"] = bootstrap
    statistical["final_test_metrics"] = test_metrics
    run.write_json("statistical_summary.json", statistical)

    # X_test/y_test are passed so gradient boosting can fall back to permutation
    # importance; this is reporting only and feeds into no decision.
    if importance := extract_feature_importance(
        estimator, dataset.feature_names, X=X_test, y=y_test, random_state=config.random_seed
    ):
        run.write_json("feature_importance.json", importance)
        plots.plot_feature_importance(importance, run.figure_path("feature_importance.png"))

    # -- model persistence (a failure here fails the run) -------------------
    data_manifest = run.read_json("data_manifest.json") if run.exists("data_manifest.json") else {}

    # Record whether the holdout actually confirmed the development-only edge gate.
    # A model whose metadata says edge_detected=True while the final test says
    # otherwise would let `predict` imply a working model, so both are stored.
    baseline_balanced: list[float] = [
        float(m["balanced_accuracy"])
        for key, m in test_metrics.items()
        if key.startswith(BASELINE_PREFIX) and np.isfinite(m.get("balanced_accuracy", np.nan))
    ]
    model_balanced = float(test_metrics[model_key].get("balanced_accuracy", np.nan))
    best_baseline_balanced = max(baseline_balanced) if baseline_balanced else None
    beat_baselines = (
        bool(model_balanced > best_baseline_balanced) if best_baseline_balanced is not None else None
    )
    logger.info(
        "final test %s the baselines (%.4f vs %.4f)",
        "CONFIRMS" if beat_baselines else "CONTRADICTS",
        model_balanced,
        best_baseline_balanced if best_baseline_balanced is not None else float("nan"),
    )

    metadata_fields = build_metadata_fields(
        config=config,
        run_id=run.run_id,
        candidate_name=spec["name"],
        family=spec["model_name"],
        params=spec["params"],
        feature_names=dataset.feature_names,
        training_first_date=str(dataset.index[0].date()),
        training_last_date=str(dataset.index[n_development - 1].date()),
        n_training_rows=len(development_idx),
        edge_detected=bool(decision.get("edge_detected", False)),
        data_sha256=data_manifest.get("raw_sha256"),
        git_commit=None,
        final_test_beat_baselines=beat_baselines,
        final_test_balanced_accuracy=model_balanced,
        final_test_best_baseline_balanced_accuracy=best_baseline_balanced,
    )
    model_metadata = save_model(estimator, run.path, metadata_fields)
    logger.info("model saved: %s (%s)", model_metadata.model_format, run.path / "model")

    # -- figures -----------------------------------------------------------
    plots.plot_confusion_matrix(
        confusion_frame(model_predictions["target"], model_predictions["prediction"]),
        run.figure_path("confusion_matrix.png"),
        title=f"{spec['name']} — final test confusion matrix",
    )
    plots.plot_roc_curve(
        model_predictions["target"], model_predictions["proba_up"], run.figure_path("roc_curve.png")
    )
    plots.plot_precision_recall_curve(
        model_predictions["target"],
        model_predictions["proba_up"],
        run.figure_path("precision_recall_curve.png"),
    )
    plots.plot_calibration_curve(
        model_predictions["target"],
        model_predictions["proba_up"],
        run.figure_path("calibration_curve.png"),
    )
    plots.plot_probability_distribution(
        model_predictions["proba_up"], run.figure_path("probability_distribution.png")
    )
    plots.plot_equity_curves(
        {name: r.equity_curve for name, r in backtests.items()},
        run.figure_path("equity_curve.png"),
        title=(f"{config.data.ticker} final test — {config.backtest.execution_mode} execution, net of costs"),
    )
    plots.plot_drawdown({name: r.drawdown for name, r in backtests.items()}, run.figure_path("drawdown.png"))

    # -- lock --------------------------------------------------------------
    lock = run.write_lock(
        config=config,
        test_start=str(dataset.index[test_idx[0]].date()),
        test_end=str(dataset.index[test_idx[-1]].date()),
        n_test_rows=len(test_idx),
        selected_candidate=spec["name"],
        data_sha256=data_manifest.get("raw_sha256"),
        previous=previous_lock,
        rerun_reason=rerun_reason,
    )

    from .report import write_model_card

    write_model_card(
        run=run,
        config=config,
        dataset=dataset,
        decision=decision,
        test_metrics=test_metrics,
        comparison=comparison,
        backtests=backtests,
        bootstrap=bootstrap,
        calibration=calibration[model_key],
        lock=lock,
        model_metadata=model_metadata,
        verdict=verdict,
        model_key=model_key,
    )
    finalize_environment(run)

    logger.info("final-test artifacts written to %s", run.path)
    return FinalTestOutput(
        run=run,
        dataset=dataset,
        test_metrics=test_metrics,
        comparison=comparison,
        backtests=backtests,
        lock=lock,
        verdict=verdict,
        edge_detected=bool(decision.get("edge_detected", False)),
        selected_candidate=spec["name"],
    )


def run_all(
    config: Config,
    force_refresh: bool = False,
    run_id: str | None = None,
) -> tuple[SelectionStageOutput, FinalTestOutput]:
    """Selection followed by a single final test, in one command."""
    selection = run_selection_stage(config, force_refresh=force_refresh, run_id=run_id)
    final = run_final_test_stage(config, selection.run.run_id, force_refresh=False)
    return selection, final
