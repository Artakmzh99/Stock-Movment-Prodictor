"""Classification metrics and fold aggregation.

Balanced accuracy is the headline number. Plain accuracy is misleading here: if
54% of sessions are up, a model that always says "up" scores 54% and looks like it
learned something. Balanced accuracy averages per-class recall, so that model
scores exactly 0.500.

``classification_metrics`` takes an **explicit** ``y_pred``. Deriving hard labels
by thresholding probabilities inside the metric function silently overrode the
hard rule of every baseline, so the confusion matrix in a report did not match the
predictions saved alongside it. Callers now pass the predictions they actually
made; ``threshold`` is only a fallback for models whose rule *is* a threshold.

Every metric degrades gracefully: a window containing a single class yields NaN
for ROC-AUC rather than raising, and NaN propagates into the report instead of
being quietly replaced by a number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = "balanced_accuracy"

METRIC_ORDER: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "f1_macro",
    "precision_up",
    "recall_up",
    "mcc",
    "log_loss",
    "brier_score",
)

#: Metrics where lower is better, for ranking and aggregation.
LOWER_IS_BETTER = frozenset({"log_loss", "brier_score"})


def classification_metrics(
    y_true: Any,
    proba_up: Any,
    y_pred: Any | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full metric suite for one evaluation window.

    `y_pred` is the hard prediction actually made. When omitted it falls back to
    ``proba_up >= threshold``, which is correct for threshold-rule models only.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba_up, dtype=float), 1e-9, 1 - 1e-9)

    if y_pred is None:
        predictions = (proba >= threshold).astype(int)
    else:
        predictions = np.asarray(y_pred).astype(int)

    if len(predictions) != len(y_true_arr):
        raise ValueError(f"y_pred has {len(predictions)} rows but y_true has {len(y_true_arr)}")

    both_classes = len(np.unique(y_true_arr)) == 2

    metrics: dict[str, float] = {
        "n": float(len(y_true_arr)),
        "threshold": float(threshold),
        "positive_rate_true": float(y_true_arr.mean()),
        "positive_rate_pred": float(predictions.mean()),
        "accuracy": float(accuracy_score(y_true_arr, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, predictions)),
        "f1_macro": float(f1_score(y_true_arr, predictions, average="macro", zero_division=0)),
        "precision_up": float(precision_score(y_true_arr, predictions, pos_label=1, zero_division=0)),
        "recall_up": float(recall_score(y_true_arr, predictions, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, predictions)) if both_classes else float("nan"),
        "brier_score": float(brier_score_loss(y_true_arr, proba)),
        "log_loss": float(log_loss(y_true_arr, proba, labels=[0, 1])),
    }

    if both_classes:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true_arr, predictions, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
        }
    )
    return metrics


def confusion_frame(y_true: Any, y_pred: Any) -> pd.DataFrame:
    matrix = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index(["actual_down", "actual_up"], name="actual"),
        columns=pd.Index(["pred_down", "pred_up"], name="predicted"),
    )


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean, spread and a Student-t confidence interval across folds.

    The spread matters as much as the mean: a model averaging 0.53 balanced
    accuracy with a fold-to-fold std of 0.06 has demonstrated nothing.
    """
    if not fold_metrics:
        return {}

    from .statistics import t_confidence_interval

    frame = pd.DataFrame(fold_metrics)
    aggregate: dict[str, Any] = {"n_folds": len(fold_metrics)}

    for metric in METRIC_ORDER:
        if metric not in frame.columns:
            continue
        values = frame[metric].astype(float).dropna()
        if values.empty:
            continue
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate[f"{metric}_min"] = float(values.min())
        aggregate[f"{metric}_max"] = float(values.max())

        interval = t_confidence_interval(values.to_numpy())
        aggregate[f"{metric}_ci_low"] = interval.low
        aggregate[f"{metric}_ci_high"] = interval.high

    return aggregate


def metrics_to_frame(named_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Tidy comparison table: one row per model/baseline."""
    rows = [{"model": name, **metrics} for name, metrics in named_metrics.items()]
    frame = pd.DataFrame(rows).set_index("model")
    ordered = [c for c in ("n", "threshold", *METRIC_ORDER) if c in frame.columns]
    remaining = [c for c in frame.columns if c not in ordered]
    return frame[ordered + remaining]


def summarize_comparison(frame: pd.DataFrame, metric: str = PRIMARY_METRIC) -> str:
    """One-line verdict, phrased so that a null result reads as a null result.

    The comparison that matters is *the model* against *the best baseline* — not
    who won overall. Reporting "best score: 0.5157" when that score belongs to a
    random-number generator would be technically true and thoroughly misleading.
    """
    if metric not in frame.columns:
        return f"metric {metric!r} not present in comparison"

    from .baselines import BASELINE_PREFIX

    baseline_names = [n for n in frame.index if str(n).startswith(BASELINE_PREFIX)]
    model_names = [n for n in frame.index if not str(n).startswith(BASELINE_PREFIX)]

    if not model_names:
        return f"no model in comparison; best {metric} = {frame[metric].max():.4f}"

    model_name = frame.loc[model_names, metric].idxmax()
    model_score = float(frame.loc[model_name, metric])

    if not baseline_names:
        return f"{model_name}: {metric} = {model_score:.4f} (no baselines to compare against)"

    best_baseline = frame.loc[baseline_names, metric].idxmax()
    baseline_score = float(frame.loc[best_baseline, metric])
    margin = model_score - baseline_score

    verdict = "BEATS" if margin > 0 else "DOES NOT BEAT"
    return (
        f"{model_name} {metric} = {model_score:.4f} vs best baseline "
        f"{best_baseline} = {baseline_score:.4f} | margin {margin:+.4f} — "
        f"the model {verdict} the baselines on the final test set"
    )
