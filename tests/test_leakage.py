"""Leakage controls.

These decide whether any other number in the project is believable. Each targets a
specific documented failure mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from stock_movement.dataset import build_dataset
from stock_movement.features import build_features
from stock_movement.labels import drop_unlabeled_tail, make_labels
from stock_movement.models import build_model
from stock_movement.selection import development_bounds, run_selection
from stock_movement.split import temporal_holdout, walk_forward_folds
from stock_movement.validation import DataValidationError, assert_no_leakage_columns


def _matrix(ohlcv, config):
    features = build_features(ohlcv, config.features)
    labels = make_labels(ohlcv, config.labels)
    combined = drop_unlabeled_tail(features.join(labels), config.labels.horizon).dropna()
    return combined[list(features.columns)], combined["target"].astype(int), combined["future_return"]


def test_label_columns_are_rejected_from_the_feature_matrix(ohlcv, config):
    X, _, _ = _matrix(ohlcv, config)
    assert_no_leakage_columns(X)  # a clean matrix passes

    for forbidden in ("target", "future_return", "future_close"):
        with pytest.raises(DataValidationError, match="label/future columns"):
            assert_no_leakage_columns(X.assign(**{forbidden: 0.0}))


def test_dataset_contract_excludes_label_columns(dataset):
    assert "target" not in dataset.X.columns
    assert "future_return" not in dataset.X.columns
    assert not any(c.startswith("future_") for c in dataset.X.columns)


def test_scaler_is_fitted_on_training_rows_only(ohlcv, config):
    """The failure that leaves no trace in the code.

    Fit a StandardScaler on everything and the holdout's distribution silently
    informs training. The scaler's mean must equal the *training* mean.
    """
    X, y, _ = _matrix(ohlcv, config)
    holdout = temporal_holdout(len(X), config.split)

    pipeline: Pipeline = build_model("logistic", {}, random_state=42)
    pipeline.fit(X.iloc[holdout.train_idx], y.iloc[holdout.train_idx])

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    train_median = X.iloc[holdout.train_idx].median().to_numpy()
    imputed_train = imputer.transform(X.iloc[holdout.train_idx])

    np.testing.assert_allclose(imputer.statistics_, train_median, rtol=1e-9)
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean(axis=0), rtol=1e-9)

    full_sample_mean = imputer.transform(X).mean(axis=0)
    assert not np.allclose(scaler.mean_, full_sample_mean, rtol=1e-6), (
        "scaler statistics match the full sample — preprocessing leaked the holdout"
    )


def test_refitting_a_later_fold_does_not_change_an_earlier_fold(ohlcv, config):
    """Each fold must be independent: an expanding window cannot reach backwards."""
    X, y, _ = _matrix(ohlcv, config)
    folds = walk_forward_folds(len(X), n_splits=3, gap=1)

    def probabilities(fold):
        model = build_model("logistic", {}, random_state=42)
        model.fit(X.iloc[fold.train_idx], y.iloc[fold.train_idx])
        return model.predict_proba(X.iloc[fold.eval_idx])[:, 1]

    first_pass = probabilities(folds[0])
    probabilities(folds[-1])  # train a later fold in between
    second_pass = probabilities(folds[0])

    np.testing.assert_allclose(first_pass, second_pass, rtol=1e-12)


def test_a_deliberately_leaked_feature_is_detectably_too_good(ohlcv, config):
    """Sanity check on the whole evaluation apparatus.

    Hand the model tomorrow's return. If the pipeline does *not* then score
    near-perfectly, our metrics are not measuring what we think they are. A real
    feature scoring like this is a bug, not a discovery.
    """
    X, y, future_return = _matrix(ohlcv, config)
    holdout = temporal_holdout(len(X), config.split)

    leaked = X.assign(oracle=future_return)
    model = build_model("logistic", {}, random_state=42)
    model.fit(leaked.iloc[holdout.train_idx], y.iloc[holdout.train_idx])
    accuracy = model.score(leaked.iloc[holdout.test_idx], y.iloc[holdout.test_idx])

    assert accuracy > 0.95, (
        f"a leaked oracle feature only reached {accuracy:.3f} accuracy — "
        "the evaluation harness is not sensitive to leakage"
    )


def test_gap_removes_the_row_whose_label_overlaps_the_next_segment(ohlcv, config):
    """With horizon 1, the last training row's label *is* the first eval row."""
    X, _, _ = _matrix(ohlcv, config)
    holdout = temporal_holdout(len(X), config.split)

    last_train_date = X.index[holdout.train_idx[-1]]
    first_val_date = X.index[holdout.val_idx[0]]
    skipped = X.index[holdout.train_idx[-1] + 1]

    assert last_train_date < skipped < first_val_date


def test_no_evaluation_row_is_ever_also_a_training_row(ohlcv, config):
    X, _, _ = _matrix(ohlcv, config)

    for fold in walk_forward_folds(len(X), n_splits=3, gap=1):
        train_dates = set(X.index[fold.train_idx])
        eval_dates = set(X.index[fold.eval_idx])
        assert train_dates.isdisjoint(eval_dates)
        assert max(train_dates) < min(eval_dates)


def test_selection_cannot_see_the_holdout(dataset, offline_config):
    """P0.2's central guarantee, checked on row indices."""
    _, test_start = development_bounds(dataset, offline_config)
    outcome = run_selection(dataset, offline_config)

    for fold in outcome.folds:
        assert int(fold.eval_idx.max()) < test_start


def test_predictions_are_reproducible_under_a_fixed_seed(ohlcv, config):
    X, y, _ = _matrix(ohlcv, config)
    holdout = temporal_holdout(len(X), config.split)

    def run():
        model = build_model("logistic", {}, random_state=config.random_seed)
        model.fit(X.iloc[holdout.train_idx], y.iloc[holdout.train_idx])
        return model.predict_proba(X.iloc[holdout.test_idx])[:, 1]

    np.testing.assert_allclose(run(), run(), rtol=1e-12)


def test_random_baseline_is_reproducible(ohlcv, config):
    from stock_movement.baselines import RandomBaseline

    X, y, _ = _matrix(ohlcv, config)

    first = RandomBaseline(random_state=42).fit(X, y).predict(X)
    second = RandomBaseline(random_state=42).fit(X, y).predict(X)
    different = RandomBaseline(random_state=7).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_dataset_build_is_reproducible(offline_config):
    first = build_dataset(offline_config)
    second = build_dataset(offline_config)

    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    assert first.manifest["feature_names"] == second.manifest["feature_names"]
