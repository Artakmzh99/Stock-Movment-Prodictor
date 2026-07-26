"""Figures for the run report.

Two conventions exist to prevent honest-looking lies: equity curves are plotted on
a shared log scale so strategies are actually comparable, and every
accuracy-style chart carries a reference line at the level a trivial baseline
reaches. A bar chart of balanced accuracy with a y-axis starting at 0.49 can make
noise look like a discovery; the coin-flip line makes that impossible to miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .calibration import reliability_data

FIGSIZE = (9, 5.5)
DPI = 130
COIN_FLIP = 0.5


def _finish(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(y: pd.Series, path: Path, title: str = "Target class balance") -> Path:
    """Class balance overall and per year — the target itself is non-stationary."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 3]})

    counts = y.value_counts().sort_index()
    labels = ["down (0)", "up (1)"]
    ax_left.bar(labels[: len(counts)], counts.to_numpy(), color=["indianred", "seagreen"][: len(counts)])
    ax_left.set_title(f"Overall (up rate {y.mean():.3f})")
    ax_left.grid(alpha=0.3, axis="y")

    index = pd.DatetimeIndex(y.index)
    yearly = y.groupby(index.year).mean()
    ax_right.bar(yearly.index.astype(str), yearly.to_numpy(), color="steelblue")
    ax_right.axhline(COIN_FLIP, color="crimson", ls="--", lw=1.5, label="0.50")
    ax_right.set_title("Up-rate per year")
    ax_right.set_ylim(0, 1)
    ax_right.tick_params(axis="x", rotation=90, labelsize=8)
    ax_right.legend(fontsize=8)
    ax_right.grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    return _finish(fig, path)


def plot_confusion_matrix(confusion: pd.DataFrame, path: Path, title: str = "Confusion matrix") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    values = confusion.to_numpy()
    ax.imshow(values, cmap="Blues")

    ax.set_xticks(range(len(confusion.columns)), list(confusion.columns))
    ax.set_yticks(range(len(confusion.index)), list(confusion.index))
    midpoint = values.max() / 2 if values.max() else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:,}",
                ha="center",
                va="center",
                color="white" if values[i, j] > midpoint else "black",
                fontsize=13,
            )
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    return _finish(fig, path)


def plot_roc_curve(y_true: Any, proba_up: Any, path: Path, title: str = "ROC curve") -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    y = np.asarray(y_true).astype(int)

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, proba_up)
        ax.plot(fpr, tpr, lw=2, label=f"model (AUC = {roc_auc_score(y, proba_up):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC = 0.500)")

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_precision_recall_curve(
    y_true: Any, proba_up: Any, path: Path, title: str = "Precision-recall curve"
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    y = np.asarray(y_true).astype(int)

    if len(np.unique(y)) == 2:
        precision, recall, _ = precision_recall_curve(y, proba_up)
        ax.plot(recall, precision, lw=2, label="model")
    base_rate = float(y.mean())
    # The right reference for PR is the positive base rate, not 0.5.
    ax.axhline(base_rate, color="crimson", ls="--", lw=1.5, label=f"always-up ({base_rate:.3f})")

    ax.set_xlabel("recall (up)")
    ax.set_ylabel("precision (up)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_calibration_curve(y_true: Any, proba_up: Any, path: Path, title: str = "Calibration") -> Path:
    reliability = reliability_data(y_true, proba_up)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    if len(reliability):
        sizes = 30 + 220 * reliability["count"] / reliability["count"].max()
        ax.plot(
            reliability["mean_predicted"], reliability["observed_frequency"], "-", lw=1.5, color="steelblue"
        )
        ax.scatter(
            reliability["mean_predicted"],
            reliability["observed_frequency"],
            s=sizes,
            color="steelblue",
            zorder=3,
            label="model (marker size = bin count)",
        )
    ax.axhline(float(np.mean(y_true)), color="grey", ls=":", lw=1, label="observed base rate")

    ax.set_xlabel("predicted P(up)")
    ax.set_ylabel("observed frequency of up-sessions")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_probability_distribution(
    proba_up: Any, path: Path, title: str = "Predicted probability distribution"
) -> Path:
    """A narrow spike means the model barely differentiates between sessions."""
    proba = np.asarray(proba_up, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.hist(proba, bins=40, color="steelblue", alpha=0.85)
    ax.set_xlim(0, 1)  # full axis: shows how narrow the range of opinion really is
    ax.axvline(0.5, color="black", lw=1, ls="--", label="0.50")
    ax.set_xlabel("predicted P(up)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.annotate(
        f"span {proba.min():.3f}-{proba.max():.3f}   std {proba.std(ddof=1):.4f}"
        if len(proba) > 1
        else "single prediction",
        xy=(0.5, 0.93),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "grey", "alpha": 0.85},
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_equity_curves(
    curves: dict[str, pd.Series], path: Path, title: str = "Equity curves (net of costs)"
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, equity in curves.items():
        if name.startswith("buy_and_hold"):
            # Buy-and-hold often coincides exactly with another line; dashing it
            # keeps both visible instead of one silently painting over the other.
            style: dict[str, Any] = {"lw": 2.6, "ls": "--", "color": "black", "alpha": 0.75, "zorder": 4}
        elif name.startswith("baseline") or name.startswith("always_long"):
            style = {"lw": 1.3, "alpha": 0.85}
        else:
            style = {"lw": 2.2, "zorder": 3}
        ax.plot(equity.index, equity.to_numpy(), label=name, **style)

    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 unit (log scale)")
    ax.set_xlabel("date")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, path)


def plot_drawdown(drawdowns: dict[str, pd.Series], path: Path, title: str = "Drawdown") -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, series in drawdowns.items():
        ax.plot(series.index, series.to_numpy() * 100, label=name, lw=1.5)
    ax.set_ylabel("drawdown (%)")
    ax.set_xlabel("date")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_fold_balanced_accuracy(
    fold_metrics: pd.DataFrame,
    path: Path,
    winner: str | None = None,
    metric: str = "balanced_accuracy",
    title: str = "Walk-forward balanced accuracy by fold",
) -> Path:
    """Fold-by-fold scores for every candidate, with the coin-flip reference."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if {"candidate", "fold", metric}.issubset(fold_metrics.columns):
        for candidate, block in fold_metrics.groupby("candidate"):
            is_winner = candidate == winner
            is_baseline = str(candidate).startswith("baseline_")
            ax.plot(
                range(1, len(block) + 1),
                block[metric].to_numpy(),
                marker="o" if is_winner else ".",
                lw=2.4 if is_winner else 1.0,
                alpha=1.0 if is_winner else 0.55,
                ls="--" if is_baseline else "-",
                label=f"{candidate}{' (selected)' if is_winner else ''}",
                zorder=3 if is_winner else 2,
            )

    ax.axhline(COIN_FLIP, color="crimson", ls="--", lw=2, label="coin flip (0.50)", zorder=1)
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_feature_importance(importance: dict[str, float], path: Path, top_n: int = 20) -> Path:
    series = pd.Series(importance).sort_values(key=np.abs, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(series))))
    colors = ["seagreen" if v >= 0 else "indianred" for v in series]
    ax.barh(list(series.index), series.to_numpy(), color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"Feature importance (top {len(series)})")
    ax.grid(alpha=0.3, axis="x")
    return _finish(fig, path)


def plot_threshold_scan(scan: pd.DataFrame, chosen: float, path: Path, objective: str) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for column in ("balanced_accuracy", "f1_macro", "mcc"):
        if column in scan.columns:
            ax.plot(scan["threshold"], scan[column], label=column, lw=1.6)
    ax.axvline(chosen, color="crimson", ls="--", lw=1.5, label=f"chosen = {chosen:.2f}")
    ax.set_xlabel("threshold on P(up)")
    ax.set_ylabel("score (validation data)")
    ax.set_title(f"Threshold scan on validation — objective: {objective}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, path)
