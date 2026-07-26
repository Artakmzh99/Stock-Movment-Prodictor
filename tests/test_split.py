"""Splits must be chronological, non-overlapping, and gapped."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from stock_movement.config import SplitConfig
from stock_movement.split import (
    assert_fold_is_causal,
    assert_holdout_is_causal,
    temporal_holdout,
    walk_forward_folds,
)


def test_holdout_is_ordered_and_disjoint():
    holdout = temporal_holdout(1000, SplitConfig(gap=1))

    assert holdout.train_idx.max() < holdout.val_idx.min()
    assert holdout.val_idx.max() < holdout.test_idx.min()
    assert np.intersect1d(holdout.train_idx, holdout.val_idx).size == 0
    assert np.intersect1d(holdout.val_idx, holdout.test_idx).size == 0
    assert np.intersect1d(holdout.train_idx, holdout.test_idx).size == 0


@pytest.mark.parametrize("gap", [0, 1, 5])
def test_holdout_respects_the_configured_gap(gap):
    holdout = temporal_holdout(2000, SplitConfig(gap=gap))
    assert_holdout_is_causal(holdout, gap=gap)

    assert holdout.val_idx.min() - holdout.train_idx.max() - 1 == gap
    assert holdout.test_idx.min() - holdout.val_idx.max() - 1 == gap


def test_holdout_fractions_are_approximately_honoured():
    holdout = temporal_holdout(1000, SplitConfig(gap=1))
    assert len(holdout.train_idx) == 600
    assert len(holdout.val_idx) == 200
    assert len(holdout.test_idx) == pytest.approx(198, abs=3)  # two rows lost to gaps


def test_walk_forward_windows_expand_and_stay_ordered():
    folds = walk_forward_folds(n_samples=1000, n_splits=5, gap=1)

    assert len(folds) == 5
    previous_train_size = 0
    for fold in folds:
        assert_fold_is_causal(fold, gap=1)
        assert len(fold.train_idx) > previous_train_size, "training window must expand"
        previous_train_size = len(fold.train_idx)
        assert fold.train_idx.min() == 0, "expanding window always starts at the beginning"


def test_walk_forward_evaluation_windows_do_not_overlap():
    folds = walk_forward_folds(n_samples=1200, n_splits=4, gap=1)
    for earlier, later in itertools.pairwise(folds):
        assert np.intersect1d(earlier.eval_idx, later.eval_idx).size == 0
        assert earlier.eval_idx.max() < later.eval_idx.max()


def test_a_fold_whose_evaluation_precedes_training_is_rejected():
    from stock_movement.split import Fold

    bad = Fold(name="bad", train_idx=np.arange(50, 100), eval_idx=np.arange(0, 50))
    with pytest.raises(ValueError, match="evaluation starts at"):
        assert_fold_is_causal(bad)


def test_overlapping_fold_is_rejected():
    from stock_movement.split import Fold

    bad = Fold(name="overlap", train_idx=np.arange(0, 100), eval_idx=np.arange(90, 150))
    with pytest.raises(ValueError):
        assert_fold_is_causal(bad, gap=1)


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        SplitConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)


def test_impossible_split_requests_fail_loudly():
    with pytest.raises(ValueError):
        temporal_holdout(5, SplitConfig(gap=1))
    with pytest.raises(ValueError):
        walk_forward_folds(n_samples=10, n_splits=50, gap=1)


def test_splits_are_deterministic():
    """No randomness anywhere: identical inputs, identical indices."""
    first = walk_forward_folds(800, 4, gap=1)
    second = walk_forward_folds(800, 4, gap=1)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.eval_idx, b.eval_idx)


# --------------------------------------------------------------------------
# fold metadata and error branches
# --------------------------------------------------------------------------
def test_fold_dates_describe_the_window():
    import pandas as pd

    index = pd.bdate_range("2020-01-01", periods=500)
    fold = walk_forward_folds(500, n_splits=3, gap=1)[0]

    dates = fold.dates(index)

    assert dates["fold"] == "fold_1"
    assert dates["train_start"] == str(index[fold.train_idx[0]].date())
    assert dates["train_end"] == str(index[fold.train_idx[-1]].date())
    assert dates["eval_start"] == str(index[fold.eval_idx[0]].date())
    assert dates["n_train"] == len(fold.train_idx)
    assert dates["n_eval"] == len(fold.eval_idx)
    assert dates["train_end"] < dates["eval_start"]


def test_holdout_dates_describe_all_three_segments():
    import pandas as pd

    index = pd.bdate_range("2020-01-01", periods=1000)
    holdout = temporal_holdout(1000, SplitConfig(gap=1))

    dates = holdout.dates(index)

    assert set(dates) == {"train", "validation", "test"}
    assert dates["train"]["end"] < dates["validation"]["start"]
    assert dates["validation"]["end"] < dates["test"]["start"]
    for segment in dates.values():
        assert segment["n"] > 0


def test_last_fold_extends_to_the_end_of_the_data():
    """No development rows may be silently discarded by the final fold."""
    folds = walk_forward_folds(1000, n_splits=4, gap=1)
    assert folds[-1].eval_idx[-1] == 999


def test_zero_gap_is_allowed_but_leaves_no_buffer():
    folds = walk_forward_folds(600, n_splits=3, gap=0)
    for fold in folds:
        assert_fold_is_causal(fold, gap=0)
        assert fold.eval_idx.min() == fold.train_idx.max() + 1


def test_fold_violating_the_required_gap_is_rejected():
    from stock_movement.split import Fold

    tight = Fold(name="tight", train_idx=np.arange(0, 100), eval_idx=np.arange(100, 150))
    assert_fold_is_causal(tight, gap=0)

    with pytest.raises(ValueError, match="gap of 0 rows is smaller"):
        assert_fold_is_causal(tight, gap=1)


def test_holdout_violating_the_required_gap_is_rejected():
    from stock_movement.split import Holdout, assert_holdout_is_causal

    tight = Holdout(train_idx=np.arange(0, 60), val_idx=np.arange(60, 80), test_idx=np.arange(80, 100))
    assert_holdout_is_causal(tight, gap=0)

    with pytest.raises(ValueError, match="does not respect gap"):
        assert_holdout_is_causal(tight, gap=1)


def test_non_chronological_holdout_is_rejected():
    from stock_movement.split import Holdout, assert_holdout_is_causal

    backwards = Holdout(train_idx=np.arange(50, 100), val_idx=np.arange(0, 50), test_idx=np.arange(100, 150))
    with pytest.raises(ValueError, match="not chronological"):
        assert_holdout_is_causal(backwards)


def test_overlapping_holdout_segments_are_rejected():
    from stock_movement.split import Holdout, assert_holdout_is_causal

    overlapping = Holdout(train_idx=np.arange(0, 60), val_idx=np.arange(55, 80), test_idx=np.arange(80, 100))
    with pytest.raises(ValueError):
        assert_holdout_is_causal(overlapping)


def test_single_split_is_rejected():
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        walk_forward_folds(500, n_splits=0)


def test_min_train_size_skips_undersized_early_folds():
    folds = walk_forward_folds(1000, n_splits=5, gap=1, min_train_size=400)
    assert all(len(fold.train_idx) >= 400 for fold in folds)
    assert len(folds) < 5
