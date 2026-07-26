"""Development-only candidate selection (P0.2) and deterministic ranking (P1.9)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stock_movement.baselines import BASELINE_PREFIX
from stock_movement.config import config_from_dict
from stock_movement.models import simplicity_key
from stock_movement.selection import (
    CandidateResult,
    CandidateSpec,
    baseline_candidates,
    build_candidates,
    development_bounds,
    evaluate_edge_gate,
    rank_candidates,
    run_selection,
)
from tests.conftest import small_config_payload


def _result(
    name: str,
    balanced: float,
    std: float = 0.01,
    log_loss: float = 0.69,
    complexity: int = 1,
    family: str = "logistic",
    params: dict[str, Any] | None = None,
    fold_scores: list[float] | None = None,
    roc_auc: float = 0.55,
    mcc: float = 0.05,
) -> CandidateResult:
    """A CandidateResult with hand-set aggregates, for ranking and gate tests."""
    scores = fold_scores if fold_scores is not None else [balanced] * 5
    spec = CandidateSpec(name=name, model_name=family, params=params or {}, complexity_rank=complexity)
    return CandidateResult(
        spec=spec,
        fold_metrics=[{"balanced_accuracy": s, "fold": f"fold_{i + 1}"} for i, s in enumerate(scores)],
        aggregate={
            "n_folds": len(scores),
            "balanced_accuracy_mean": balanced,
            "balanced_accuracy_std": std,
            "log_loss_mean": log_loss,
            "roc_auc_mean": roc_auc,
            "mcc_mean": mcc,
        },
    )


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def test_selection_prefers_higher_balanced_accuracy():
    ranked = rank_candidates([_result("low", 0.51), _result("high", 0.55), _result("mid", 0.53)])
    assert [r.name for r in ranked] == ["high", "mid", "low"]


def test_selection_uses_lower_std_as_tie_break():
    """Equal means: the more stable candidate wins."""
    ranked = rank_candidates([_result("volatile", 0.53, std=0.08), _result("stable", 0.53, std=0.01)])
    assert ranked[0].name == "stable"


def test_selection_uses_log_loss_as_second_tie_break():
    ranked = rank_candidates(
        [
            _result("worse_probs", 0.53, std=0.02, log_loss=0.72),
            _result("better_probs", 0.53, std=0.02, log_loss=0.68),
        ]
    )
    assert ranked[0].name == "better_probs"


def test_selection_prefers_simpler_model_when_tied():
    """Identical performance: the simpler family wins. A tie is not evidence."""
    ranked = rank_candidates(
        [
            _result("complex", 0.53, std=0.02, log_loss=0.69, complexity=3, family="lstm"),
            _result("simple", 0.53, std=0.02, log_loss=0.69, complexity=1, family="logistic"),
        ]
    )
    assert ranked[0].name == "simple"


def test_logistic_tie_prefers_smaller_c():
    """Within a family, stronger regularisation breaks the tie."""
    ranked = rank_candidates(
        [
            _result("loose", 0.53, std=0.02, log_loss=0.69, params={"C": 10.0}),
            _result("tight", 0.53, std=0.02, log_loss=0.69, params={"C": 0.01}),
        ]
    )
    assert ranked[0].name == "tight"


def test_hgb_tie_prefers_simpler_tree():
    ranked = rank_candidates(
        [
            _result(
                "deep",
                0.53,
                std=0.02,
                log_loss=0.69,
                complexity=2,
                family="hist_gradient_boosting",
                params={"max_leaf_nodes": 31, "max_iter": 300},
            ),
            _result(
                "shallow",
                0.53,
                std=0.02,
                log_loss=0.69,
                complexity=2,
                family="hist_gradient_boosting",
                params={"max_leaf_nodes": 7, "max_iter": 100},
            ),
        ]
    )
    assert ranked[0].name == "shallow"


def test_hgb_simplicity_prefers_larger_leaves_and_more_regularisation():
    small_leaf = simplicity_key("hist_gradient_boosting", {"min_samples_leaf": 10})
    large_leaf = simplicity_key("hist_gradient_boosting", {"min_samples_leaf": 80})
    assert large_leaf < small_leaf

    weak = simplicity_key("hist_gradient_boosting", {"l2_regularization": 0.1})
    strong = simplicity_key("hist_gradient_boosting", {"l2_regularization": 5.0})
    assert strong < weak


def test_logistic_simplicity_prefers_no_class_weight_on_an_exact_tie():
    assert simplicity_key("logistic", {"C": 1.0, "class_weight": None}) < simplicity_key(
        "logistic", {"C": 1.0, "class_weight": "balanced"}
    )


def test_search_order_is_deterministic():
    """Name is the final tie-break, so ranking never depends on input order."""
    candidates = [
        _result("bbb", 0.53, std=0.02, log_loss=0.69),
        _result("aaa", 0.53, std=0.02, log_loss=0.69),
        _result("ccc", 0.53, std=0.02, log_loss=0.69),
    ]
    assert [r.name for r in rank_candidates(candidates)] == ["aaa", "bbb", "ccc"]
    assert [r.name for r in rank_candidates(list(reversed(candidates)))] == ["aaa", "bbb", "ccc"]


def test_nan_scores_rank_last_rather_than_crashing():
    ranked = rank_candidates([_result("broken", float("nan")), _result("fine", 0.51)])
    assert ranked[0].name == "fine"


# --------------------------------------------------------------------------
# edge gate
# --------------------------------------------------------------------------
def _gate_config(**overrides: Any):
    """The documented gate needs 4 wins out of 5 folds, so 5 splits are required.

    Config validation enforces min_fold_wins <= walk_forward_splits, which is why
    this helper must set both together.
    """
    return config_from_dict(
        small_config_payload(
            split={"walk_forward_splits": 5, "gap": 1},
            selection={"min_fold_wins": 4, **overrides},
        )
    )


def test_edge_gate_requires_four_of_five_fold_wins():
    """A big average margin earned in one fold is not an edge."""
    config = _gate_config()
    baseline = _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5)

    # Wins one fold enormously, loses the rest: mean clears the margin, folds do not.
    spiky = _result("spiky", 0.53, fold_scores=[0.49, 0.49, 0.49, 0.49, 0.69])
    gate = evaluate_edge_gate(spiky, [baseline], config)
    assert gate.edge_detected is False
    assert gate.checks["fold_wins"]["passed"] is False
    assert gate.checks["margin_over_best_baseline"]["passed"] is True

    consistent = _result("consistent", 0.53, fold_scores=[0.52, 0.53, 0.54, 0.53, 0.49])
    gate = evaluate_edge_gate(consistent, [baseline], config)
    assert gate.checks["fold_wins"]["value"] == 4
    assert gate.edge_detected is True


def test_edge_gate_requires_the_balanced_accuracy_margin():
    config = _gate_config()
    baseline = _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5)
    narrow = _result("narrow", 0.505, fold_scores=[0.505] * 5)

    gate = evaluate_edge_gate(narrow, [baseline], config)

    assert gate.edge_detected is False
    assert gate.checks["margin_over_best_baseline"]["passed"] is False


def test_edge_gate_requires_roc_auc_above_half():
    config = _gate_config()
    baseline = _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5)
    backwards = _result("backwards", 0.53, fold_scores=[0.53] * 5, roc_auc=0.48)

    gate = evaluate_edge_gate(backwards, [baseline], config)

    assert gate.edge_detected is False
    assert gate.checks["roc_auc"]["passed"] is False


def test_edge_gate_requires_positive_mcc():
    config = _gate_config()
    baseline = _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5)
    negative = _result("negative_mcc", 0.53, fold_scores=[0.53] * 5, mcc=-0.01)

    gate = evaluate_edge_gate(negative, [baseline], config)

    assert gate.edge_detected is False
    assert gate.checks["mcc"]["passed"] is False


def test_edge_gate_compares_against_the_strongest_baseline():
    config = _gate_config()
    baselines = [
        _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5),
        _result(f"{BASELINE_PREFIX}last_direction", 0.54, fold_scores=[0.54] * 5),
    ]
    model = _result("model", 0.52, fold_scores=[0.52] * 5)

    gate = evaluate_edge_gate(model, baselines, config)

    assert gate.best_baseline == f"{BASELINE_PREFIX}last_direction"
    assert gate.edge_detected is False


def test_edge_gate_without_baselines_cannot_detect_an_edge():
    gate = evaluate_edge_gate(_result("model", 0.60), [], _gate_config())
    assert gate.edge_detected is False
    assert "no baselines" in gate.reason


def test_edge_gate_reason_states_the_null_result_plainly():
    config = _gate_config()
    baseline = _result(f"{BASELINE_PREFIX}majority", 0.50, fold_scores=[0.50] * 5)
    gate = evaluate_edge_gate(_result("weak", 0.50, fold_scores=[0.50] * 5), [baseline], config)

    assert "no stable predictive edge" in gate.reason


# --------------------------------------------------------------------------
# candidate construction
# --------------------------------------------------------------------------
def test_candidates_cover_every_configured_family():
    config = config_from_dict(
        small_config_payload(selection={"families": ["logistic", "hist_gradient_boosting"]})
    )
    families = {c.model_name for c in build_candidates(config)}
    assert families == {"logistic", "hist_gradient_boosting"}


def test_baseline_candidates_are_flagged_as_baselines():
    config = config_from_dict(small_config_payload())
    specs = baseline_candidates(config)

    assert specs
    assert all(spec.is_baseline for spec in specs)
    assert all(spec.name.startswith(BASELINE_PREFIX) for spec in specs)


def test_random_baseline_can_be_excluded():
    config = config_from_dict(small_config_payload(selection={"include_random_baseline": False}))
    names = {spec.name for spec in baseline_candidates(config)}
    assert f"{BASELINE_PREFIX}random" not in names


def test_no_candidates_is_an_error():
    config = config_from_dict(small_config_payload(selection={"families": []}))
    with pytest.raises(ValueError, match="no candidates"):
        build_candidates(config)


# --------------------------------------------------------------------------
# the central guarantee
# --------------------------------------------------------------------------
def test_all_candidates_use_identical_folds(dataset, offline_config):
    """Comparing candidates scored on different rows would be meaningless."""
    outcome = run_selection(dataset, offline_config)

    fold_signatures = set()
    for result in outcome.ranked + outcome.baselines:
        signature = tuple(
            (m["fold"], int(m["n"])) for m in sorted(result.fold_metrics, key=lambda m: m["fold"])
        )
        fold_signatures.add(signature)

    assert len(fold_signatures) == 1, "candidates were scored on different fold shapes"


def test_candidate_selection_never_reads_test_rows(dataset, offline_config):
    """The holdout must be untouched — asserted against row indices, not prose."""
    _, test_start = development_bounds(dataset, offline_config)
    outcome = run_selection(dataset, offline_config)

    for fold in outcome.folds:
        assert int(fold.train_idx.max()) < test_start
        assert int(fold.eval_idx.max()) < test_start

    assert outcome.decision["development"]["max_row_index_used"] < test_start
    assert outcome.decision["test_data_used_for_selection"] is False
    assert outcome.decision["held_out_test"]["opened_during_selection"] is False


def test_out_of_fold_predictions_stay_inside_development(dataset, offline_config):
    _, test_start = development_bounds(dataset, offline_config)
    outcome = run_selection(dataset, offline_config)

    test_first_date = dataset.index[test_start]
    for result in outcome.ranked + outcome.baselines:
        if len(result.oof_predictions):
            assert result.oof_predictions.index.max() < test_first_date


def test_baseline_cannot_be_selected_as_saved_model(dataset, offline_config):
    """Baselines are comparisons; they must never become the production model."""
    outcome = run_selection(dataset, offline_config)

    assert outcome.winner.spec.is_baseline is False
    assert not outcome.winner.name.startswith(BASELINE_PREFIX)
    assert not outcome.decision["selected_candidate"]["is_baseline"]
    assert all(not r.spec.is_baseline for r in outcome.ranked)


def test_selection_is_reproducible(dataset, offline_config):
    first = run_selection(dataset, offline_config)
    second = run_selection(dataset, offline_config)

    assert first.winner.name == second.winner.name
    assert first.winner.mean("balanced_accuracy") == pytest.approx(second.winner.mean("balanced_accuracy"))
    assert [r.name for r in first.ranked] == [r.name for r in second.ranked]


def test_decision_records_its_own_hash(dataset, offline_config):
    outcome = run_selection(dataset, offline_config)
    assert len(outcome.decision["decision_sha256"]) == 64
    assert outcome.decision["config_sha256"] == offline_config.sha256()


def test_development_bounds_leave_a_gap(dataset, offline_config):
    n_development, test_start = development_bounds(dataset, offline_config)
    assert test_start - n_development == offline_config.split.gap
    assert test_start < len(dataset)


def test_fold_metrics_frame_covers_every_candidate(dataset, offline_config):
    outcome = run_selection(dataset, offline_config)
    frame = outcome.fold_metrics_frame

    expected = {r.name for r in outcome.ranked + outcome.baselines}
    assert set(frame["candidate"].unique()) == expected
    assert np.isfinite(frame["balanced_accuracy"]).all()
