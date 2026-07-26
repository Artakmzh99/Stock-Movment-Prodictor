"""Mandatory baselines.

A direction model is only interesting if it beats these. They are deliberately
trivial, and on daily equity data they are surprisingly hard to beat.

Each baseline exposes ``fit`` / ``predict_proba`` / ``predict`` so evaluation,
threshold selection and backtesting treat them identically to a real model.

**Hard rule versus probability.** Every baseline has an obvious *hard* rule and no
obvious probability. Thresholding a made-up probability at 0.5 would silently
change what the baseline asserts — a "majority" baseline whose training prior is
0.48 would flip to predicting *down* every day. So each baseline reports:

* ``predict`` — its actual rule, used for accuracy, F1, MCC and the confusion
  matrix;
* ``predict_proba`` — the training frequency of the outcome, used for log loss,
  Brier and ROC-AUC.

Emitting uniform random probabilities (the previous behaviour of the random
baseline) was worse than useless: it produced a log loss of ~1.06 against a
possible ~0.69, making the diagnostic look pathologically bad for reasons that had
nothing to do with the baseline's actual predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

#: Feature naming the "current direction" for each target definition.
DIRECTION_FEATURE = {
    "open_to_close": "open_to_close_return",
    "close_to_close": "return_1d",
}


class _Baseline(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Common plumbing. Subclasses implement ``_proba_up`` and ``_hard``."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any) -> _Baseline:
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.train_up_rate_ = float(y.mean()) if len(y) else 0.5
        self.n_features_in_ = X.shape[1]
        self._fit_extra(X, y)
        return self

    def _fit_extra(self, X: pd.DataFrame, y: np.ndarray) -> None:
        return None

    def _proba_up(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _hard(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_up = np.clip(np.asarray(self._proba_up(X), dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p_up, p_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._hard(X)).astype(int)


class MajorityClassBaseline(_Baseline):
    """Always predict whichever class was more common in training."""

    def _fit_extra(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.majority_class_ = int(self.train_up_rate_ >= 0.5)

    def _proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.train_up_rate_)

    def _hard(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=int)


class AlwaysUpBaseline(_Baseline):
    """Always predict "up" — the real opponent, given equities' upward drift."""

    def _proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.train_up_rate_)

    def _hard(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


class RandomBaseline(_Baseline):
    """Seeded coin flips weighted by the training prior. A diagnostic, not a rival.

    The probability is the *constant* training prior — the honestly calibrated
    answer for a process with no information. Only the hard prediction is random.
    """

    def _proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.train_up_rate_)

    def _hard(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        return rng.binomial(1, self.train_up_rate_, size=len(X)).astype(int)


class LastDirectionBaseline(_Baseline):
    """Follow the current session's direction: up today implies up next session.

    The hard prediction is pure sign-following. The probability is the training
    frequency of an up outcome *conditional on* the current direction, so the
    baseline is calibrated and its ROC-AUC is meaningful rather than degenerate.
    If direction carries no information, both conditional rates collapse to the
    prior and AUC lands at 0.50 — the honest answer.
    """

    def __init__(self, random_state: int = 42, feature: str = "open_to_close_return") -> None:
        super().__init__(random_state=random_state)
        self.feature = feature

    def _direction(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature not in X.columns:
            raise KeyError(
                f"{type(self).__name__} needs feature {self.feature!r}; available: {sorted(X.columns)[:8]}..."
            )
        return (X[self.feature].to_numpy() > 0).astype(int)

    def _fit_extra(self, X: pd.DataFrame, y: np.ndarray) -> None:
        direction = self._direction(X)
        up_mask = direction == 1
        self.rate_after_up_ = float(y[up_mask].mean()) if up_mask.any() else self.train_up_rate_
        self.rate_after_down_ = float(y[~up_mask].mean()) if (~up_mask).any() else self.train_up_rate_

    def _proba_up(self, X: pd.DataFrame) -> np.ndarray:
        direction = self._direction(X)
        return np.where(direction == 1, self.rate_after_up_, self.rate_after_down_)

    def _hard(self, X: pd.DataFrame) -> np.ndarray:
        return self._direction(X)


BASELINE_REGISTRY: dict[str, type[_Baseline]] = {
    "majority": MajorityClassBaseline,
    "always_up": AlwaysUpBaseline,
    "last_direction": LastDirectionBaseline,
    "random": RandomBaseline,
}

BASELINE_PREFIX = "baseline_"


def build_baselines(
    random_state: int = 42,
    target_definition: str = "open_to_close",
    include_random: bool = True,
) -> dict[str, _Baseline]:
    """Instantiate one fresh copy of every baseline, keyed by name."""
    direction_feature = DIRECTION_FEATURE.get(target_definition, "return_1d")

    baselines: dict[str, _Baseline] = {}
    for name, cls in BASELINE_REGISTRY.items():
        if name == "random" and not include_random:
            continue
        if cls is LastDirectionBaseline:
            baselines[name] = LastDirectionBaseline(random_state=random_state, feature=direction_feature)
        else:
            baselines[name] = cls(random_state=random_state)
    return baselines
