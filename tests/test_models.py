"""Model factory, complexity ordering and importance extraction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from stock_movement.models import (
    HGB,
    LOGISTIC,
    LSTM,
    MODEL_COMPLEXITY,
    RANDOM_FOREST,
    build_model,
    default_param_grid,
    extract_feature_importance,
    lstm_available,
    simplicity_key,
)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", [LOGISTIC, HGB, RANDOM_FOREST])
def test_every_tabular_family_builds_a_pipeline(family):
    model = build_model(family, {}, random_state=42)

    assert isinstance(model, Pipeline)
    assert "model" in model.named_steps


def test_logistic_and_random_forest_include_an_imputer():
    """Models without native NaN handling must impute inside the pipeline."""
    for family in (LOGISTIC, RANDOM_FOREST):
        assert "imputer" in build_model(family, {}).named_steps, family


def test_only_logistic_scales():
    """Trees are scale-invariant; scaling them would be noise, not preprocessing."""
    assert "scaler" in build_model(LOGISTIC, {}).named_steps
    assert "scaler" not in build_model(HGB, {}).named_steps
    assert "scaler" not in build_model(RANDOM_FOREST, {}).named_steps


def test_gradient_boosting_relies_on_native_nan_support():
    assert "imputer" not in build_model(HGB, {}).named_steps


def test_params_override_the_defaults():
    model = build_model(LOGISTIC, {"C": 0.005, "max_iter": 111})
    estimator = model.named_steps["model"]

    assert pytest.approx(0.005) == estimator.C
    assert estimator.max_iter == 111


def test_random_state_is_threaded_through():
    assert build_model(LOGISTIC, {}, random_state=7).named_steps["model"].random_state == 7
    assert build_model(HGB, {}, random_state=7).named_steps["model"].random_state == 7


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown model family"):
        build_model("crystal_ball", {})


def test_lstm_availability_is_reported_honestly():
    available = lstm_available()
    assert isinstance(available, bool)

    if not available:
        with pytest.raises(ImportError, match="TensorFlow"):
            build_model(LSTM, {})


# --------------------------------------------------------------------------
# complexity and grids
# --------------------------------------------------------------------------
def test_complexity_orders_families_from_simple_to_flexible():
    assert MODEL_COMPLEXITY[LOGISTIC] < MODEL_COMPLEXITY[HGB] < MODEL_COMPLEXITY[LSTM]


@pytest.mark.parametrize("family", [LOGISTIC, HGB, RANDOM_FOREST, LSTM])
def test_grids_are_small_and_non_empty(family):
    """Large grids on a few thousand noisy rows just overfit the folds."""
    grid = default_param_grid(family)

    assert 1 <= len(grid) <= 6, f"{family} grid has {len(grid)} candidates"
    assert all(isinstance(candidate, dict) for candidate in grid)


def test_unknown_family_grid_is_a_single_default():
    assert default_param_grid("unknown") == [{}]


@pytest.mark.parametrize("family", [LOGISTIC, HGB, RANDOM_FOREST])
def test_every_grid_candidate_is_constructible(family):
    for candidate in default_param_grid(family):
        build_model(family, candidate, random_state=42)


# --------------------------------------------------------------------------
# simplicity ordering (P1.9)
# --------------------------------------------------------------------------
def test_logistic_simplicity_is_driven_by_c():
    assert simplicity_key(LOGISTIC, {"C": 0.01}) < simplicity_key(LOGISTIC, {"C": 1.0})


def test_random_forest_simplicity_prefers_shallower_trees():
    assert simplicity_key(RANDOM_FOREST, {"max_depth": 3}) < simplicity_key(RANDOM_FOREST, {"max_depth": 10})


def test_lstm_simplicity_prefers_shorter_lookback_and_fewer_units():
    assert simplicity_key(LSTM, {"lookback": 20, "units": (8,)}) < simplicity_key(
        LSTM, {"lookback": 60, "units": (64, 32)}
    )


def test_unknown_family_simplicity_is_neutral():
    assert simplicity_key("unknown", {}) == (0.0,)


def test_simplicity_keys_are_totally_ordered_within_a_grid():
    """The key must never raise when comparing any two candidates in a grid."""
    for family in (LOGISTIC, HGB, RANDOM_FOREST):
        keys = [simplicity_key(family, candidate) for candidate in default_param_grid(family)]
        assert sorted(keys) == sorted(keys)  # comparison does not raise


# --------------------------------------------------------------------------
# feature importance
# --------------------------------------------------------------------------
def test_logistic_importance_returns_signed_coefficients(dataset):
    model = build_model(LOGISTIC, {}).fit(dataset.X, dataset.y)
    importance = extract_feature_importance(model, dataset.feature_names)

    assert importance is not None
    assert set(importance) == set(dataset.feature_names)
    assert any(v < 0 for v in importance.values()), "coefficients should carry a sign"


def test_random_forest_importance_is_non_negative(dataset):
    model = build_model(RANDOM_FOREST, {"n_estimators": 20}).fit(dataset.X, dataset.y)
    importance = extract_feature_importance(model, dataset.feature_names)

    assert importance is not None
    assert all(v >= 0 for v in importance.values())


def test_gradient_boosting_falls_back_to_permutation_importance(dataset):
    """HistGradientBoosting exposes neither coef_ nor feature_importances_.

    Without the permutation fallback it reported no importance at all.
    """
    model = build_model(HGB, {"max_iter": 20}).fit(dataset.X, dataset.y)

    assert not hasattr(model.named_steps["model"], "feature_importances_")
    assert extract_feature_importance(model, dataset.feature_names) is None

    importance = extract_feature_importance(model, dataset.feature_names, X=dataset.X, y=dataset.y)
    assert importance is not None
    assert set(importance) == set(dataset.feature_names)


def test_permutation_importance_is_deterministic_for_a_seed(dataset):
    model = build_model(HGB, {"max_iter": 15}).fit(dataset.X, dataset.y)

    first = extract_feature_importance(model, dataset.feature_names, X=dataset.X, y=dataset.y, random_state=1)
    second = extract_feature_importance(
        model, dataset.feature_names, X=dataset.X, y=dataset.y, random_state=1
    )

    assert first == second


def test_importance_of_an_estimator_without_the_notion_is_none():
    class Opaque:
        pass

    assert extract_feature_importance(Opaque(), ["a", "b"]) is None


def test_importance_accepts_a_bare_estimator(dataset):
    """Not everything arrives wrapped in a Pipeline."""
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(max_iter=200).fit(np.asarray(dataset.X), np.asarray(dataset.y))
    importance = extract_feature_importance(estimator, dataset.feature_names)

    assert importance is not None
    assert len(importance) == len(dataset.feature_names)


def test_importance_of_a_pipeline_without_a_model_step_is_none():
    from sklearn.impute import SimpleImputer

    pipeline = Pipeline(steps=[("imputer", SimpleImputer())])
    assert extract_feature_importance(pipeline, ["a"]) is None
