"""Metrics (P1.1) and calibration binning (P1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from stock_movement.calibration import (
    bin_ids,
    calibration_bins,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_data,
    select_threshold,
)
from stock_movement.evaluation import (
    aggregate_fold_metrics,
    classification_metrics,
    confusion_frame,
    metrics_to_frame,
    summarize_comparison,
)


# --------------------------------------------------------------------------
# P1.1 explicit y_pred
# --------------------------------------------------------------------------
def test_metrics_use_the_supplied_hard_predictions():
    """A baseline's hard rule must drive threshold metrics, not its probabilities.

    Here the probabilities are a constant 0.45, so a 0.5 threshold would predict
    all-zeros. The hard rule says all-ones. The metrics must reflect the rule.
    """
    y_true = np.array([1, 1, 1, 0])
    constant_proba = np.full(4, 0.45)
    hard_rule = np.ones(4, dtype=int)

    with_rule = classification_metrics(y_true, constant_proba, y_pred=hard_rule)
    with_threshold = classification_metrics(y_true, constant_proba, threshold=0.5)

    assert with_rule["accuracy"] == pytest.approx(0.75)
    assert with_rule["positive_rate_pred"] == pytest.approx(1.0)
    assert with_threshold["accuracy"] == pytest.approx(0.25)
    assert with_threshold["positive_rate_pred"] == pytest.approx(0.0)


def test_confusion_counts_match_the_supplied_predictions():
    y_true = np.array([1, 1, 0, 0])
    proba = np.full(4, 0.9)
    hard = np.array([1, 0, 1, 0])

    metrics = classification_metrics(y_true, proba, y_pred=hard)

    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    total = sum(
        metrics[k] for k in ("true_positives", "true_negatives", "false_positives", "false_negatives")
    )
    assert total == len(y_true)


def test_probability_metrics_ignore_the_hard_rule():
    """Log loss and Brier depend only on probabilities, whatever the rule says.

    y_true is deliberately imbalanced (3 up, 1 down) so that all-ones and
    all-zeros give *different* accuracies — on balanced labels both score 0.5 and
    the contrast would prove nothing.
    """
    y_true = np.array([1, 1, 1, 0])
    proba = np.array([0.8, 0.7, 0.6, 0.3])

    a = classification_metrics(y_true, proba, y_pred=np.ones(4, dtype=int))
    b = classification_metrics(y_true, proba, y_pred=np.zeros(4, dtype=int))

    assert a["log_loss"] == pytest.approx(b["log_loss"])
    assert a["brier_score"] == pytest.approx(b["brier_score"])
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    assert a["accuracy"] == pytest.approx(0.75)
    assert b["accuracy"] == pytest.approx(0.25)


def test_mismatched_prediction_length_is_rejected():
    with pytest.raises(ValueError, match="y_pred has"):
        classification_metrics(np.array([1, 0]), np.array([0.5, 0.5]), y_pred=np.array([1]))


def test_perfect_predictions_score_perfectly():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.1, 0.95, 0.05, 0.8, 0.2])

    metrics = classification_metrics(y_true, proba)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


def test_balanced_accuracy_punishes_the_always_up_strategy():
    """Plain accuracy rewards guessing the majority class; balanced accuracy does not."""
    y_true = np.array([1] * 70 + [0] * 30)
    metrics = classification_metrics(y_true, np.full(100, 0.99))

    assert metrics["accuracy"] == pytest.approx(0.70)
    assert metrics["balanced_accuracy"] == pytest.approx(0.50)


def test_single_class_window_yields_nan_auc_not_a_crash():
    y_true = np.ones(20, dtype=int)
    metrics = classification_metrics(y_true, np.linspace(0.4, 0.9, 20))

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["mcc"])
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["accuracy"])


# --------------------------------------------------------------------------
# P1.2 calibration binning
# --------------------------------------------------------------------------
def test_bin_assignment_is_fixed_width():
    assignments = bin_ids(np.array([0.0, 0.05, 0.15, 0.5, 0.95, 1.0]))
    np.testing.assert_array_equal(assignments, [0, 0, 1, 5, 9, 9])


def test_calibration_handles_zero_and_one_probabilities():
    """p == 1.0 belongs in the last bin, not an eleventh bin that cannot exist."""
    y_true = np.array([0, 1])
    bins = calibration_bins(y_true, np.array([0.0, 1.0]))

    assert {b.bin_id for b in bins} == {0, 9}
    assert sum(b.count for b in bins) == 2


def test_probabilities_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bin_ids(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bin_ids(np.array([0.5, 1.2]))


def test_bin_counts_sum_to_sample_count():
    """The invariant the previous quantile-plus-resize implementation broke."""
    rng = np.random.default_rng(0)
    for n in (1, 7, 50, 913):
        y_true = rng.integers(0, 2, n)
        proba = rng.uniform(0, 1, n)
        assert sum(b.count for b in calibration_bins(y_true, proba)) == n


def test_constant_probabilities_create_one_nonempty_bin():
    """The realistic case for this problem: predictions concentrated near 0.5."""
    y_true = np.array([1, 0, 1, 0, 1])
    bins = calibration_bins(y_true, np.full(5, 0.52))

    assert len(bins) == 1
    assert bins[0].bin_id == 5
    assert bins[0].count == 5


def test_constant_probability_ece_matches_manual_value():
    """One bin, so ECE is exactly |predicted - observed|.

    Six of ten outcomes are up and the model always says 0.50, giving |0.50 - 0.60|.
    """
    y_true = np.array([1] * 6 + [0] * 4)
    proba = np.full(10, 0.50)

    assert expected_calibration_error(y_true, proba) == pytest.approx(0.10)
    assert maximum_calibration_error(y_true, proba) == pytest.approx(0.10)


def test_ece_is_a_weighted_average_over_bins():
    # Bin 1 (0.15): 2 samples, observed 0.0 -> gap 0.15
    # Bin 8 (0.85): 8 samples, observed 1.0 -> gap 0.15
    y_true = np.array([0, 0] + [1] * 8)
    proba = np.array([0.15, 0.15] + [0.85] * 8)

    assert expected_calibration_error(y_true, proba) == pytest.approx(0.15)


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(3)
    proba = rng.uniform(0.05, 0.95, 40000)
    y_true = rng.binomial(1, proba)

    assert expected_calibration_error(y_true, proba) < 0.01


def test_calibration_report_preserves_every_sample():
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 500)
    proba = rng.uniform(0, 1, 500)

    report = calibration_report(y_true, proba)

    assert report["n"] == 500
    assert sum(b["count"] for b in report["bins"]) == 500
    assert report["n_bins"] == 10


def test_calibration_report_rewards_predicting_the_base_rate():
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.55, 2000)
    report = calibration_report(y_true, np.full(2000, 0.55))

    assert report["brier_score"] == pytest.approx(report["brier_score_of_base_rate"], abs=0.01)
    assert report["predicted_probability_std"] == pytest.approx(0.0)


def test_calibration_report_flags_an_overconfident_model():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    overconfident = rng.choice([0.02, 0.98], 500)

    report = calibration_report(y_true, overconfident)

    assert report["brier_score"] > report["brier_score_of_base_rate"]
    assert report["expected_calibration_error"] > 0.3


def test_reliability_frame_is_empty_for_no_samples():
    assert reliability_data(np.array([]), np.array([])).empty


def test_mismatched_calibration_lengths_are_rejected():
    with pytest.raises(ValueError, match="proba_up has"):
        calibration_bins(np.array([1, 0]), np.array([0.5]))


# --------------------------------------------------------------------------
# aggregation and verdicts
# --------------------------------------------------------------------------
def test_fold_aggregation_reports_spread_and_intervals():
    folds = [
        {"balanced_accuracy": 0.60, "roc_auc": 0.62, "accuracy": 0.6},
        {"balanced_accuracy": 0.40, "roc_auc": 0.44, "accuracy": 0.4},
        {"balanced_accuracy": 0.50, "roc_auc": 0.51, "accuracy": 0.5},
    ]
    aggregate = aggregate_fold_metrics(folds)

    assert aggregate["balanced_accuracy_mean"] == pytest.approx(0.50)
    assert aggregate["balanced_accuracy_std"] > 0.09
    assert aggregate["balanced_accuracy_min"] == pytest.approx(0.40)
    assert aggregate["balanced_accuracy_max"] == pytest.approx(0.60)
    assert aggregate["n_folds"] == 3
    assert aggregate["balanced_accuracy_ci_low"] < 0.50 < aggregate["balanced_accuracy_ci_high"]


def test_empty_aggregation_is_empty():
    assert aggregate_fold_metrics([]) == {}


def test_verdict_compares_the_model_against_the_best_baseline():
    frame = metrics_to_frame(
        {
            "baseline_majority": {"balanced_accuracy": 0.50},
            "baseline_last_direction": {"balanced_accuracy": 0.52},
            "model_logistic": {"balanced_accuracy": 0.51},
        }
    )
    verdict = summarize_comparison(frame, "balanced_accuracy")

    # Beats one baseline but not the best one; that is not a win.
    assert "DOES NOT BEAT" in verdict
    assert "baseline_last_direction" in verdict


def test_verdict_recognises_a_genuine_win():
    frame = metrics_to_frame(
        {"baseline_majority": {"balanced_accuracy": 0.50}, "model_logistic": {"balanced_accuracy": 0.58}}
    )
    assert "BEATS" in summarize_comparison(frame, "balanced_accuracy")


def test_verdict_without_baselines_says_so():
    frame = metrics_to_frame({"model_logistic": {"balanced_accuracy": 0.58}})
    assert "no baselines" in summarize_comparison(frame, "balanced_accuracy")


def test_verdict_for_a_missing_metric():
    frame = metrics_to_frame({"model_logistic": {"balanced_accuracy": 0.58}})
    assert "not present" in summarize_comparison(frame, "nonexistent")


def test_confusion_frame_orientation_is_actual_by_predicted():
    frame = confusion_frame([1, 1, 0], [1, 0, 0])
    assert frame.loc["actual_up", "pred_up"] == 1
    assert frame.loc["actual_up", "pred_down"] == 1
    assert frame.loc["actual_down", "pred_down"] == 1


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------
def test_threshold_selection_returns_a_candidate_and_the_full_scan():
    rng = np.random.default_rng(3)
    y_true = rng.binomial(1, 0.5, 400)
    proba = np.clip(y_true * 0.25 + rng.normal(0.4, 0.15, 400), 0.01, 0.99)

    threshold, scan = select_threshold(
        y_true, proba, candidates=[0.3, 0.4, 0.5, 0.6, 0.7], objective="balanced_accuracy"
    )

    assert threshold in {0.3, 0.4, 0.5, 0.6, 0.7}
    assert len(scan) == 5
    assert scan.loc[scan["threshold"] == threshold, "balanced_accuracy"].iloc[0] == pytest.approx(
        scan["balanced_accuracy"].max()
    )


def test_threshold_ties_resolve_to_the_lower_threshold():
    """Determinism: an exact tie must not depend on iteration order."""
    y_true = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1])

    threshold, _ = select_threshold(y_true, proba, candidates=[0.2, 0.5, 0.8])
    assert threshold == 0.2


def test_net_sharpe_objective_uses_the_intraday_cost_model():
    """A per-session cost must penalise a high-exposure threshold more than a rare one."""
    rng = np.random.default_rng(7)
    proba = rng.uniform(0.3, 0.7, 300)
    future_return = rng.normal(0.0, 0.01, 300)

    _, scan = select_threshold(
        np.array((future_return > 0).astype(int)),
        proba,
        candidates=[0.3, 0.65],
        objective="net_sharpe",
        future_return=future_return,
        cost_rate=0.001,
        execution_mode="next_open",
    )

    low, high = scan.set_index("threshold").loc[[0.3, 0.65], "predicted_up_rate"]
    assert low > high  # a lower threshold trades far more and pays far more cost


def test_unknown_threshold_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown threshold objective"):
        select_threshold(np.array([0, 1]), np.array([0.4, 0.6]), [0.5], objective="profit")


def test_empty_candidate_list_is_rejected():
    with pytest.raises(ValueError, match="no threshold candidates"):
        select_threshold(np.array([0, 1]), np.array([0.4, 0.6]), [])
