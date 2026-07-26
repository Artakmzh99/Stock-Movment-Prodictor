"""End-to-end integration, on synthetic data and without touching the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_movement.artifacts import FinalTestAlreadyCompletedError
from stock_movement.backtest import ALWAYS_LONG_INTRADAY, BUY_AND_HOLD
from stock_movement.dataset import build_dataset
from stock_movement.pipeline import (
    SELECTED_MODEL_SPEC,
    SELECTION_DECISION,
    run_all,
    run_final_test_stage,
    run_selection_stage,
)
from stock_movement.selection import development_bounds


# --------------------------------------------------------------------------
# dataset contract
# --------------------------------------------------------------------------
def test_dataset_contract_holds(dataset):
    assert not dataset.X.isna().to_numpy().any(), "no NaN may reach the model"
    assert "target" not in dataset.X.columns
    assert "future_return" not in dataset.X.columns
    assert dataset.X.index.is_monotonic_increasing
    assert not dataset.X.index.has_duplicates
    assert dataset.X.index.equals(dataset.y.index)
    assert dataset.X.index.equals(dataset.future_return.index)
    assert set(dataset.y.unique()) <= {0, 1}


def test_label_and_future_return_agree_row_by_row(dataset):
    pd.testing.assert_series_equal(dataset.y, (dataset.future_return > 0).astype(int), check_names=False)


def test_open_to_close_target_uses_the_next_open(offline_config, dataset):
    """The default target must be Close(t+1)/Open(t+1)-1, not close-to-close."""
    assert offline_config.labels.target_definition == "open_to_close"

    prices = dataset.prices
    row = dataset.index[100]
    position = prices.index.get_loc(row)
    next_open = prices["Open"].iloc[position + 1]
    next_close = prices["Close"].iloc[position + 1]

    assert dataset.future_return.loc[row] == pytest.approx(next_close / next_open - 1.0)


def test_too_few_rows_fails_loudly(offline_config):
    """`min_rows` is a post-warm-up requirement, so the raw check demands more."""
    from stock_movement.validation import DataValidationError

    strict = offline_config.model_copy(
        update={"data": offline_config.data.model_copy(update={"min_rows": 100000})}
    )
    with pytest.raises(DataValidationError, match="need at least"):
        build_dataset(strict)


def test_surviving_rows_exceed_the_configured_minimum(offline_config):
    """The two row-count checks are consistent, and the post-warm-up count wins.

    Raw validation requires `min_rows + warm_up`; the dataset check then requires
    `min_rows` after the rolling windows have been consumed. Because warm-up costs
    at most `max_window + horizon` rows, passing the first implies passing the
    second — the dataset-level check is a defensive guard, and this test pins the
    relationship that makes it redundant rather than asserting it can fire.
    """
    dataset = build_dataset(offline_config)
    warm_up = offline_config.features.max_window + offline_config.labels.horizon + 1

    assert len(dataset) >= offline_config.data.min_rows
    assert dataset.manifest["rows_dropped_for_nan"] <= warm_up


# --------------------------------------------------------------------------
# stage 1: selection writes no test metrics
# --------------------------------------------------------------------------
def test_select_model_writes_no_test_metrics(offline_config):
    output = run_selection_stage(offline_config)
    run = output.run

    for forbidden in (
        "final_test_metrics.json",
        "final_test_predictions.parquet",
        "final_test_comparison.csv",
        "backtest_metrics.json",
        "backtest_daily.parquet",
        "final_test.lock.json",
        "model_card.md",
    ):
        assert not run.exists(forbidden), f"selection stage must not write {forbidden}"

    assert not (run.path / "model").exists(), "no model may be saved before the final test"


def test_selection_stage_writes_its_required_artifacts(offline_config):
    run = run_selection_stage(offline_config).run

    for name in (
        "resolved_config.yaml",
        "resolved_config.json",
        "environment.json",
        "data_manifest.json",
        "feature_manifest.json",
        "split_manifest.json",
        "candidate_summary.csv",
        "candidate_fold_metrics.csv",
        "candidate_oof_predictions.parquet",
        SELECTION_DECISION,
        SELECTED_MODEL_SPEC,
        "statistical_summary.json",
    ):
        assert run.exists(name), f"missing artifact: {name}"

    for figure in ("class_balance.png", "fold_balanced_accuracy.png"):
        assert (run.path / "figures" / figure).exists(), figure


def test_selection_decision_certifies_development_only(offline_config):
    run = run_selection_stage(offline_config).run
    decision = run.read_json(SELECTION_DECISION)

    assert decision["test_data_used_for_selection"] is False
    assert decision["held_out_test"]["opened_during_selection"] is False
    assert decision["development"]["max_row_index_used"] < decision["held_out_test"]["test_start_index"]
    assert decision["config_sha256"] == offline_config.sha256()


def test_run_id_encodes_the_config_hash(offline_config):
    run = run_selection_stage(offline_config).run
    assert run.run_id.endswith(offline_config.short_hash())


def test_a_second_selection_creates_a_new_run_directory(offline_config):
    first = run_selection_stage(offline_config).run
    second = run_selection_stage(offline_config).run

    assert first.run_id != second.run_id or first.path != second.path


# --------------------------------------------------------------------------
# stage 2: final test
# --------------------------------------------------------------------------
def test_final_test_requires_locked_selection(offline_config):
    from stock_movement.artifacts import create_run

    run = create_run(offline_config, run_id="no-selection")
    assert not run.exists(SELECTION_DECISION)

    with pytest.raises(FileNotFoundError, match=SELECTION_DECISION):
        run_final_test_stage(offline_config, run_id="no-selection")


def test_final_test_rejects_an_uncertified_decision(offline_config):
    selection = run_selection_stage(offline_config)
    run = selection.run

    tampered = run.read_json(SELECTION_DECISION)
    tampered["test_data_used_for_selection"] = True
    run.write_json(SELECTION_DECISION, tampered)

    with pytest.raises(ValueError, match="development-only selection"):
        run_final_test_stage(offline_config, run_id=run.run_id)


def test_final_test_rejects_a_config_hash_mismatch(offline_config):
    selection = run_selection_stage(offline_config)

    changed = offline_config.model_copy(update={"random_seed": 999})
    with pytest.raises(ValueError, match="config hash mismatch"):
        run_final_test_stage(changed, run_id=selection.run.run_id)


def test_full_pipeline_writes_every_required_artifact(offline_config):
    _, final = run_all(offline_config)
    run = final.run

    for name in (
        "resolved_config.yaml",
        "resolved_config.json",
        "environment.json",
        "data_manifest.json",
        "feature_manifest.json",
        "split_manifest.json",
        "candidate_summary.csv",
        "candidate_fold_metrics.csv",
        "candidate_oof_predictions.parquet",
        SELECTION_DECISION,
        SELECTED_MODEL_SPEC,
        "final_test.lock.json",
        "final_test_metrics.json",
        "final_test_predictions.parquet",
        "backtest_metrics.json",
        "backtest_daily.parquet",
        "statistical_summary.json",
        "bootstrap_summary.json",
        "model_card.md",
    ):
        assert run.exists(name), f"missing artifact: {name}"

    for figure in (
        "class_balance.png",
        "fold_balanced_accuracy.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "probability_distribution.png",
        "equity_curve.png",
        "drawdown.png",
    ):
        assert (run.path / "figures" / figure).exists(), f"missing figure: {figure}"

    assert (run.path / "model" / "model_metadata.json").exists()


def test_final_test_writes_lock_and_refuses_a_second_run(offline_config):
    _, final = run_all(offline_config)

    assert final.lock.completed is True
    assert final.lock.rerun_count == 0

    with pytest.raises(FinalTestAlreadyCompletedError):
        run_final_test_stage(offline_config, run_id=final.run.run_id)


def test_rerun_with_a_reason_is_recorded(offline_config):
    _, final = run_all(offline_config)

    rerun = run_final_test_stage(
        offline_config,
        run_id=final.run.run_id,
        allow_rerun=True,
        rerun_reason="verifying a data fix",
    )

    assert rerun.lock.rerun_count == 1
    assert rerun.lock.rerun_history[0]["reason"] == "verifying a data fix"


def test_rerun_without_a_reason_is_refused(offline_config):
    _, final = run_all(offline_config)

    with pytest.raises(ValueError, match="non-empty explanation"):
        run_final_test_stage(offline_config, run_id=final.run.run_id, allow_rerun=True)


# --------------------------------------------------------------------------
# comparability and correctness
# --------------------------------------------------------------------------
def test_all_four_baselines_are_evaluated(offline_config):
    _, final = run_all(offline_config)
    baselines = {k for k in final.test_metrics if k.startswith("baseline_")}

    assert baselines == {
        "baseline_majority",
        "baseline_always_up",
        "baseline_last_direction",
        "baseline_random",
    }
    assert any(not k.startswith("baseline_") for k in final.test_metrics)


def test_backtest_uses_the_intraday_cost_model(offline_config):
    """Every active session must pay a round trip under next_open execution."""
    _, final = run_all(offline_config)

    intraday = final.backtests[ALWAYS_LONG_INTRADAY]
    expected = len(intraday.position) * offline_config.backtest.round_trip_cost_rate

    assert intraday.metrics["total_cost_paid"] == pytest.approx(expected)
    assert intraday.metrics["n_active_sessions"] == intraday.metrics["n_completed_trades"]


def test_buy_and_hold_is_reported_separately_from_always_long(offline_config):
    _, final = run_all(offline_config)

    assert ALWAYS_LONG_INTRADAY in final.backtests
    assert BUY_AND_HOLD in final.backtests
    # Holding pays one entry; re-entering daily pays hundreds.
    assert (
        final.backtests[BUY_AND_HOLD].metrics["total_cost_paid"]
        < final.backtests[ALWAYS_LONG_INTRADAY].metrics["total_cost_paid"]
    )


def test_walk_forward_never_reaches_the_test_set(offline_config, dataset):
    _, test_start = development_bounds(dataset, offline_config)
    test_first_date = dataset.index[test_start]

    selection = run_selection_stage(offline_config)
    for fold in selection.outcome.decision["folds"]:
        assert pd.Timestamp(fold["eval_end"]) < test_first_date


def test_backtest_and_predictions_span_the_same_window(offline_config):
    _, final = run_all(offline_config)
    model_key = next(k for k in final.backtests if k.startswith("model_"))

    predictions = pd.read_parquet(final.run.path / "final_test_predictions.parquet")
    backtest = final.backtests[model_key]

    assert backtest.returns.index.equals(predictions.index)
    assert backtest.metrics["n_periods"] == len(predictions)


def test_metadata_records_provenance(offline_config):
    _, final = run_all(offline_config)
    environment = final.run.read_json("environment.json")
    manifest = final.run.read_json("data_manifest.json")

    assert environment["config_sha256"] == offline_config.sha256()
    assert environment["run_finished_utc"] is not None
    assert "git" in environment and "packages" in environment
    assert manifest["ticker"] == "TEST"
    assert len(manifest["raw_sha256"]) == 64


def test_model_card_states_the_verdict_and_provenance(offline_config):
    _, final = run_all(offline_config)
    card = (final.run.path / "model_card.md").read_text()

    assert final.verdict in card
    assert offline_config.sha256() in card
    assert "not investment advice" in card.lower()
    assert "Limitations" in card
    assert final.lock.completed_at_utc in card


def test_model_card_flags_a_contradicted_edge(offline_config):
    """When development says edge and the holdout disagrees, the card must lead with it."""
    _, final = run_all(offline_config)
    card = (final.run.path / "model_card.md").read_text()

    metadata = final.run.read_json("model/model_metadata.json")
    if metadata["edge_detected"] and metadata["final_test_beat_baselines"] is False:
        assert "CONTRADICTED" in card


def test_pipeline_is_reproducible_across_runs(offline_config):
    _, first = run_all(offline_config, run_id="run-a")
    _, second = run_all(offline_config, run_id="run-b")

    assert first.selected_candidate == second.selected_candidate
    assert first.verdict == second.verdict

    a = pd.read_parquet(first.run.path / "final_test_predictions.parquet")
    b = pd.read_parquet(second.run.path / "final_test_predictions.parquet")
    np.testing.assert_allclose(a["proba_up"].to_numpy(), b["proba_up"].to_numpy(), rtol=1e-12)


def test_bootstrap_summary_reports_intervals(offline_config):
    _, final = run_all(offline_config)
    bootstrap = final.run.read_json("bootstrap_summary.json")

    assert "mean_daily_return" in bootstrap
    assert "sharpe_ratio" in bootstrap
    assert any(k.startswith("vs_") for k in bootstrap)
    for entry in bootstrap.values():
        assert "ci_low" in entry and "ci_high" in entry


def test_saved_model_reloads_and_reproduces_test_probabilities(offline_config):
    from stock_movement.persistence import load_model

    _, final = run_all(offline_config)
    predictions = pd.read_parquet(final.run.path / "final_test_predictions.parquet")

    model = load_model(final.run.path)
    dataset = build_dataset(offline_config)
    X_test = dataset.X.loc[predictions.index]

    np.testing.assert_allclose(
        model.predict_proba(X_test)[:, 1], predictions["proba_up"].to_numpy(), rtol=1e-9
    )


def test_close_to_close_config_uses_the_change_based_cost_model(offline_config):
    """The research variant must charge on position changes, not per session."""
    research = offline_config.model_copy(
        update={
            "labels": offline_config.labels.model_copy(update={"target_definition": "close_to_close"}),
            "backtest": offline_config.backtest.model_copy(update={"execution_mode": "close_to_close"}),
        }
    )
    _, final = run_all(research, run_id="research-run")

    always = final.backtests[BUY_AND_HOLD]
    assert always.execution_mode == "close_to_close"
    assert always.metrics["n_completed_trades"] == 1
    assert always.metrics["total_cost_paid"] == pytest.approx(research.backtest.one_way_cost_rate)
