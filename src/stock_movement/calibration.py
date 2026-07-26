"""Probability calibration and threshold selection.

Two distinct questions, often conflated:

* *Discrimination* — can the model rank up-sessions above down-sessions? (ROC-AUC)
* *Calibration* — when it says 60%, does it happen 60% of the time? (Brier, ECE)

A model can discriminate well and be badly calibrated, or be perfectly calibrated
and useless. Position sizing needs calibration; ranking needs discrimination.

## Binning

Ten **fixed-width** bins over [0, 1], with every sample assigned exactly once::

    bin_id = min(int(p * 10), 9)

The previous implementation used sklearn's quantile strategy and then resized a
separately-computed histogram to match, which silently dropped or double-counted
samples and produced a meaningless ECE whenever predictions were concentrated —
which, for this problem, they always are. Fixed-width bins make the invariant
checkable: bin counts sum to the sample count, exactly.

The ``min(..., 9)`` is what puts p = 1.0 in the last bin instead of a tenth bin
that should not exist.

## Thresholds

Chosen on **validation data only**. Picking the threshold that looks best on the
final test set is a quiet way of fitting it, and the easiest way to fool yourself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
)

N_BINS = 10

CLASSIFICATION_OBJECTIVES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p)),
    "f1_macro": lambda y, p: float(f1_score(y, p, average="macro", zero_division=0)),
    "mcc": lambda y, p: float(matthews_corrcoef(y, p)) if len(np.unique(y)) == 2 else float("nan"),
}


@dataclass
class CalibrationBin:
    bin_id: int
    lower_edge: float
    upper_edge: float
    count: int
    mean_predicted: float
    observed_frequency: float
    absolute_gap: float
    squared_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bin_ids(proba_up: Any, n_bins: int = N_BINS) -> np.ndarray:
    """Fixed-width bin assignment. Every probability lands in exactly one bin."""
    proba = np.asarray(proba_up, dtype=float)
    if np.any((proba < 0.0) | (proba > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    # min(..., n_bins - 1) keeps p == 1.0 in the final bin.
    assigned: np.ndarray = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    return assigned


def calibration_bins(y_true: Any, proba_up: Any, n_bins: int = N_BINS) -> list[CalibrationBin]:
    """Per-bin calibration detail for every non-empty bin."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba_up, dtype=float)
    if len(y) != len(proba):
        raise ValueError(f"y_true has {len(y)} rows but proba_up has {len(proba)}")

    assignments = bin_ids(proba, n_bins)
    width = 1.0 / n_bins

    bins: list[CalibrationBin] = []
    for bin_id in range(n_bins):
        mask = assignments == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(proba[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(mean_predicted - observed)
        bins.append(
            CalibrationBin(
                bin_id=bin_id,
                lower_edge=bin_id * width,
                upper_edge=(bin_id + 1) * width,
                count=count,
                mean_predicted=mean_predicted,
                observed_frequency=observed,
                absolute_gap=gap,
                squared_gap=gap**2,
            )
        )
    return bins


def reliability_data(y_true: Any, proba_up: Any, n_bins: int = N_BINS) -> pd.DataFrame:
    """Reliability diagram data as a tidy frame."""
    bins = calibration_bins(y_true, proba_up, n_bins)
    if not bins:
        return pd.DataFrame(
            columns=["bin_id", "mean_predicted", "observed_frequency", "count", "absolute_gap"]
        )
    return pd.DataFrame([b.to_dict() for b in bins])[
        [
            "bin_id",
            "lower_edge",
            "upper_edge",
            "count",
            "mean_predicted",
            "observed_frequency",
            "absolute_gap",
        ]
    ]


def expected_calibration_error(y_true: Any, proba_up: Any, n_bins: int = N_BINS) -> float:
    """ECE = sum over bins of (bin_fraction * |mean_predicted - observed|)."""
    bins = calibration_bins(y_true, proba_up, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return float(sum((b.count / total) * b.absolute_gap for b in bins))


def maximum_calibration_error(y_true: Any, proba_up: Any, n_bins: int = N_BINS) -> float:
    """MCE = the largest absolute gap in any non-empty bin."""
    bins = calibration_bins(y_true, proba_up, n_bins)
    return float(max((b.absolute_gap for b in bins), default=float("nan")))


def calibration_report(y_true: Any, proba_up: Any, n_bins: int = N_BINS) -> dict[str, Any]:
    """Brier score, ECE, MCE and reference points.

    Reference: predicting the base rate every session gives Brier == p(1-p) (about
    0.2475 for a 55% up-rate). Anything above that is worse than a constant.
    """
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba_up, dtype=float)
    base_rate = float(y.mean()) if len(y) else float("nan")

    bins = calibration_bins(y, proba, n_bins)
    total_binned = sum(b.count for b in bins)
    if total_binned != len(y):  # pragma: no cover - invariant, not a runtime branch
        raise AssertionError(f"calibration bins hold {total_binned} of {len(y)} samples")

    return {
        "n": len(y),
        "n_bins": n_bins,
        "n_nonempty_bins": len(bins),
        "brier_score": float(brier_score_loss(y, np.clip(proba, 1e-9, 1 - 1e-9))) if len(y) else float("nan"),
        "brier_score_of_base_rate": float(base_rate * (1 - base_rate)),
        "expected_calibration_error": expected_calibration_error(y, proba, n_bins),
        "maximum_calibration_error": maximum_calibration_error(y, proba, n_bins),
        "mean_predicted_probability": float(proba.mean()) if len(proba) else float("nan"),
        "observed_up_rate": base_rate,
        "predicted_probability_std": float(proba.std(ddof=1)) if len(proba) > 1 else 0.0,
        "predicted_probability_min": float(proba.min()) if len(proba) else float("nan"),
        "predicted_probability_max": float(proba.max()) if len(proba) else float("nan"),
        "bins": [b.to_dict() for b in bins],
    }


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------
def _net_sharpe(
    proba_up: np.ndarray,
    future_return: np.ndarray,
    threshold: float,
    cost_rate: float,
    execution_mode: str,
    trading_days: int = 252,
) -> float:
    """Sharpe of a long/cash rule at this threshold, net of costs.

    Uses the same cost model as the real backtest, so a threshold tuned here is
    not tuned against arithmetic the backtest will not reproduce.
    """
    position = (proba_up >= threshold).astype(float)
    if execution_mode == "next_open":
        costs = np.abs(position) * cost_rate  # every active session is a round trip
    else:
        turnover = np.abs(np.diff(position, prepend=0.0))
        costs = turnover * cost_rate

    net = position * future_return - costs
    if len(net) < 2 or net.std(ddof=1) == 0:
        return float("nan")
    return float(net.mean() / net.std(ddof=1) * np.sqrt(trading_days))


def select_threshold(
    y_true: Any,
    proba_up: Any,
    candidates: Any,
    objective: str = "balanced_accuracy",
    future_return: Any | None = None,
    cost_rate: float = 0.0,
    execution_mode: str = "next_open",
    trading_days: int = 252,
) -> tuple[float, pd.DataFrame]:
    """Scan candidate thresholds on **validation** data and return the best.

    Also returns the full scan, so a report shows whether the optimum is a broad
    plateau (trustworthy) or a lone spike (almost certainly noise).
    """
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba_up, dtype=float)
    candidate_list = [float(c) for c in candidates]

    if not candidate_list:
        raise ValueError("no threshold candidates supplied")

    rows: list[dict[str, float]] = []
    for threshold in candidate_list:
        predictions = (proba >= threshold).astype(int)
        row: dict[str, float] = {
            "threshold": threshold,
            "predicted_up_rate": float(predictions.mean()),
        }
        for name, fn in CLASSIFICATION_OBJECTIVES.items():
            row[name] = fn(y, predictions)
        if future_return is not None:
            row["net_sharpe"] = _net_sharpe(
                proba,
                np.asarray(future_return, dtype=float),
                threshold,
                cost_rate,
                execution_mode,
                trading_days,
            )
        rows.append(row)

    scan = pd.DataFrame(rows)

    if objective not in scan.columns:
        available = [c for c in scan.columns if c != "threshold"]
        raise ValueError(f"unknown threshold objective {objective!r}; available: {available}")

    scores = scan[objective]
    if scores.isna().all():
        return 0.5, scan

    # Ties resolved toward the lower threshold for determinism.
    best = scan.loc[scores == scores.max(), "threshold"].min()
    return float(best), scan
