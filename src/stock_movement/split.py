"""Chronological data splitting.

Two schemes, both strictly ordered in time and both honouring a `gap`:

* ``temporal_holdout`` — one 60/20/20 train/validation/test cut;
* ``walk_forward_folds`` — expanding-window folds over the train+validation span.

**Why the gap matters.** With a horizon-1 label, the label of the last training
row is the return of the first validation row. Without a gap that row leaks: the
model is trained on information about a day it is then scored on. Dropping
`horizon` rows at every boundary removes the overlap. Those rows belong to no
split — they are discarded, not reassigned.

Random K-fold is not offered here at all. Shuffling puts the future in the
training set and the past in validation, which produces flattering, meaningless
scores — the single most common way this project can be got wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass(frozen=True)
class Fold:
    """One train/evaluation pair, expressed as positional indices."""

    name: str
    train_idx: np.ndarray
    eval_idx: np.ndarray

    def dates(self, index: pd.DatetimeIndex) -> dict[str, Any]:
        return {
            "fold": self.name,
            "train_start": str(index[self.train_idx[0]].date()),
            "train_end": str(index[self.train_idx[-1]].date()),
            "eval_start": str(index[self.eval_idx[0]].date()),
            "eval_end": str(index[self.eval_idx[-1]].date()),
            "n_train": len(self.train_idx),
            "n_eval": len(self.eval_idx),
        }


@dataclass(frozen=True)
class Holdout:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def dates(self, index: pd.DatetimeIndex) -> dict[str, dict[str, Any]]:
        def span(idx: np.ndarray) -> dict[str, Any]:
            return {
                "start": str(index[idx[0]].date()),
                "end": str(index[idx[-1]].date()),
                "n": len(idx),
            }

        return {"train": span(self.train_idx), "validation": span(self.val_idx), "test": span(self.test_idx)}


def temporal_holdout(n_samples: int, config: SplitConfig) -> Holdout:
    """Split ``[0, n)`` into train / validation / test in time order, with gaps."""
    gap = config.gap
    if n_samples < 3 * (gap + 1) + 3:
        raise ValueError(f"not enough samples ({n_samples}) to build a gapped 3-way split")

    n_train = int(np.floor(n_samples * config.train_fraction))
    n_val = int(np.floor(n_samples * config.validation_fraction))

    train_idx = np.arange(0, n_train)
    val_start = n_train + gap
    val_idx = np.arange(val_start, val_start + n_val)
    test_start = val_start + n_val + gap
    test_idx = np.arange(test_start, n_samples)

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"split produced an empty segment (train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}); need more data or smaller gap"
        )

    return Holdout(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def walk_forward_folds(
    n_samples: int,
    n_splits: int,
    gap: int = 1,
    min_train_size: int | None = None,
) -> list[Fold]:
    """Expanding-window folds: train on everything up to a point, evaluate the block after.

    Fold k trains on ``[0, cut_k)`` and evaluates ``[cut_k + gap, cut_k + gap + size)``.
    Training data only ever grows, mirroring how a model would actually be
    retrained as time passes.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    usable = n_samples - gap
    if usable < n_splits + 1:
        raise ValueError(f"cannot build {n_splits} folds from {n_samples} samples with gap={gap}")

    fold_size = usable // (n_splits + 1)
    if min_train_size is None:
        min_train_size = fold_size
    if fold_size < 1 or min_train_size < 1:
        raise ValueError("resulting folds would be empty; reduce n_splits or add more data")

    folds: list[Fold] = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        eval_start = train_end + gap
        eval_end = eval_start + fold_size if k < n_splits - 1 else n_samples

        if eval_start >= n_samples or eval_end <= eval_start:
            break
        if train_end < min_train_size:
            continue

        folds.append(
            Fold(
                name=f"fold_{k + 1}",
                train_idx=np.arange(0, train_end),
                eval_idx=np.arange(eval_start, min(eval_end, n_samples)),
            )
        )

    if not folds:
        raise ValueError("walk-forward produced no usable folds")
    return folds


def assert_fold_is_causal(fold: Fold, gap: int = 0) -> None:
    """Fail loudly if a fold's evaluation window is not strictly after its training window."""
    max_train, min_eval = fold.train_idx.max(), fold.eval_idx.min()
    if min_eval <= max_train:
        raise ValueError(f"{fold.name}: evaluation starts at {min_eval} but training ends at {max_train}")
    if min_eval - max_train - 1 < gap:
        raise ValueError(
            f"{fold.name}: gap of {min_eval - max_train - 1} rows is smaller than required {gap}"
        )
    if np.intersect1d(fold.train_idx, fold.eval_idx).size:
        raise ValueError(f"{fold.name}: train and evaluation indices overlap")


def assert_holdout_is_causal(holdout: Holdout, gap: int = 0) -> None:
    for earlier, later, label in (
        (holdout.train_idx, holdout.val_idx, "train/validation"),
        (holdout.val_idx, holdout.test_idx, "validation/test"),
    ):
        if later.min() <= earlier.max():
            raise ValueError(f"{label} boundary is not chronological")
        if later.min() - earlier.max() - 1 < gap:
            raise ValueError(f"{label} boundary does not respect gap={gap}")
        if np.intersect1d(earlier, later).size:
            raise ValueError(f"{label} segments overlap")
