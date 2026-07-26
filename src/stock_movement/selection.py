"""Development-only candidate selection.

## The problem this solves

Previously each config ran one model and immediately scored the final test. Running
logistic, gradient boosting, LSTM and a benchmark-feature variant therefore exposed
the *same* final test four times, and then the best number was reported. That is
multiple comparisons against a holdout, and it inflates the result whether or not
anyone intended it. The holdout stops being a holdout the second time you look.

## The protocol

Every model decision — family, hyperparameters, threshold — happens inside the
development window (the first 80% of rows). All candidates are scored on
**identical** walk-forward folds, ranked deterministically, and exactly one winner
is written to ``selection_decision.json``. Only then may ``final_test`` open the
holdout, once.

``rows_used`` in the decision records the development row count, and
``max_row_index_used`` proves no candidate saw a row at or beyond the test
boundary — a claim that is asserted in code rather than promised in a README.

## The edge gate

A winner is always recorded (something has to be saved), but ``edge_detected`` is
true only when all four conditions hold:

1. balanced accuracy at least `min_balanced_accuracy_margin` above the best baseline;
2. beats the best baseline in at least `min_fold_wins` of the folds;
3. mean ROC-AUC above 0.50;
4. mean MCC above 0.

Condition 2 is the one that matters most. A model can clear an average margin by
winning one fold enormously and losing the rest, which is noise wearing a
convincing costume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from .baselines import BASELINE_PREFIX, build_baselines
from .config import Config
from .dataset import Dataset
from .evaluation import PRIMARY_METRIC, aggregate_fold_metrics, classification_metrics
from .models import (
    MODEL_COMPLEXITY,
    build_model,
    default_param_grid,
    lstm_available,
    simplicity_key,
)
from .provenance import sha256_canonical_json
from .split import Fold, assert_fold_is_causal, walk_forward_folds


@dataclass(frozen=True)
class CandidateSpec:
    """One fully-specified thing that could become the production model."""

    name: str
    model_name: str
    params: dict[str, Any]
    complexity_rank: int
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "params": _jsonable(self.params),
            "complexity_rank": self.complexity_rank,
            "is_baseline": self.is_baseline,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _param_label(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (list, tuple)):
            value = "x".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return ",".join(parts)


def build_candidates(config: Config) -> list[CandidateSpec]:
    """Enumerate every model candidate. Baselines are added separately."""
    families = list(config.selection.families)
    if config.selection.include_lstm and "lstm" not in families:
        families.append("lstm")

    candidates: list[CandidateSpec] = []
    for family in families:
        if family == "lstm" and not lstm_available():
            # Absent TensorFlow is a fact about the environment; skip loudly in the
            # decision record rather than failing the whole selection.
            continue
        for params in default_param_grid(family):
            candidates.append(
                CandidateSpec(
                    name=f"{family}[{_param_label(params)}]",
                    model_name=family,
                    params=dict(params),
                    complexity_rank=MODEL_COMPLEXITY.get(family, 99),
                )
            )

    if not candidates:
        raise ValueError(
            f"no candidates built from selection.families={families}. "
            "At least one model family must be available."
        )
    return candidates


def baseline_candidates(config: Config) -> list[CandidateSpec]:
    baselines = build_baselines(
        random_state=config.random_seed,
        target_definition=config.labels.target_definition,
        include_random=config.selection.include_random_baseline,
    )
    return [
        CandidateSpec(
            name=f"{BASELINE_PREFIX}{name}",
            model_name=f"{BASELINE_PREFIX}{name}",
            params={},
            complexity_rank=0,
            is_baseline=True,
        )
        for name in baselines
    ]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@dataclass
class CandidateResult:
    spec: CandidateSpec
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    oof_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def name(self) -> str:
        return self.spec.name

    def fold_scores(self, metric: str = PRIMARY_METRIC) -> np.ndarray:
        return np.array([float(m.get(metric, np.nan)) for m in self.fold_metrics])

    def mean(self, metric: str) -> float:
        return float(self.aggregate.get(f"{metric}_mean", np.nan))

    def std(self, metric: str) -> float:
        return float(self.aggregate.get(f"{metric}_std", np.nan))

    def summary_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "candidate": self.spec.name,
            "family": self.spec.model_name,
            "complexity_rank": self.spec.complexity_rank,
            "is_baseline": self.spec.is_baseline,
            "params": json.dumps(_jsonable(self.spec.params), sort_keys=True),
        }
        for metric in ("balanced_accuracy", "roc_auc", "log_loss", "mcc", "accuracy", "brier_score"):
            row[f"{metric}_mean"] = self.mean(metric)
            row[f"{metric}_std"] = self.std(metric)
        row["balanced_accuracy_ci_low"] = self.aggregate.get("balanced_accuracy_ci_low")
        row["balanced_accuracy_ci_high"] = self.aggregate.get("balanced_accuracy_ci_high")
        row["n_folds"] = self.aggregate.get("n_folds")
        return row


def _estimator_for(spec: CandidateSpec, config: Config) -> Any:
    if spec.is_baseline:
        name = spec.model_name.removeprefix(BASELINE_PREFIX)
        baselines = build_baselines(
            random_state=config.random_seed,
            target_definition=config.labels.target_definition,
            include_random=True,
        )
        return clone(baselines[name])
    return build_model(spec.model_name, spec.params, random_state=config.random_seed)


def _proba_up(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(estimator.predict_proba(X))
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()


def evaluate_candidate(
    spec: CandidateSpec,
    dataset: Dataset,
    folds: list[Fold],
    config: Config,
) -> CandidateResult:
    """Score one candidate across the shared folds.

    A *fresh* estimator is built per fold. Reusing a fitted object would carry a
    later fold's parameters into an earlier one.
    """
    fold_metrics: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for fold in folds:
        assert_fold_is_causal(fold, gap=config.split.gap)

        X_train = dataset.X.iloc[fold.train_idx]
        y_train = dataset.y.iloc[fold.train_idx]
        X_eval = dataset.X.iloc[fold.eval_idx]
        y_eval = dataset.y.iloc[fold.eval_idx]

        estimator = _estimator_for(spec, config)
        estimator.fit(X_train, y_train)

        proba = _proba_up(estimator, X_eval)
        # The candidate's own hard rule — not a 0.5 cut of a probability it may
        # never have meant as a decision boundary.
        predictions = np.asarray(estimator.predict(X_eval)).astype(int)

        scores = classification_metrics(
            y_eval, proba, y_pred=predictions, threshold=config.threshold.classification_value
        )
        metrics: dict[str, Any] = dict(scores)
        metrics["fold"] = fold.name
        fold_metrics.append(metrics)

        frames.append(
            pd.DataFrame(
                {
                    "candidate": spec.name,
                    "fold": fold.name,
                    "proba_up": proba,
                    "prediction": predictions,
                    "target": y_eval.to_numpy(),
                    "future_return": dataset.future_return.iloc[fold.eval_idx].to_numpy(),
                },
                index=X_eval.index,
            )
        )

    return CandidateResult(
        spec=spec,
        fold_metrics=fold_metrics,
        aggregate=aggregate_fold_metrics(fold_metrics),
        oof_predictions=pd.concat(frames) if frames else pd.DataFrame(),
    )


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def ranking_key(result: CandidateResult) -> tuple[Any, ...]:
    """Deterministic total order. Lower tuple sorts first, i.e. ranks better.

    1. higher mean balanced accuracy
    2. lower balanced-accuracy std (stability across regimes)
    3. lower mean log loss (better probabilities)
    4. lower family complexity
    5. simpler hyperparameters within the family
    6. candidate name, so the order is total and never depends on dict iteration
    """
    balanced = result.mean(PRIMARY_METRIC)
    stability = result.std(PRIMARY_METRIC)
    log_loss_mean = result.mean("log_loss")

    return (
        -(balanced if np.isfinite(balanced) else -np.inf),
        stability if np.isfinite(stability) else np.inf,
        log_loss_mean if np.isfinite(log_loss_mean) else np.inf,
        result.spec.complexity_rank,
        simplicity_key(result.spec.model_name, result.spec.params),
        result.spec.name,
    )


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    return sorted(results, key=ranking_key)


# --------------------------------------------------------------------------
# edge gate
# --------------------------------------------------------------------------
@dataclass
class EdgeGate:
    edge_detected: bool
    checks: dict[str, Any]
    best_baseline: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_edge_gate(
    winner: CandidateResult,
    baselines: list[CandidateResult],
    config: Config,
) -> EdgeGate:
    """Apply the four documented conditions. All must hold."""
    settings = config.selection

    if not baselines:
        return EdgeGate(
            edge_detected=False,
            checks={},
            best_baseline=None,
            reason="no baselines evaluated, so no edge can be established",
        )

    best = max(baselines, key=lambda r: r.mean(PRIMARY_METRIC))
    baseline_scores = best.fold_scores(PRIMARY_METRIC)
    winner_scores = winner.fold_scores(PRIMARY_METRIC)

    comparable = min(len(winner_scores), len(baseline_scores))
    fold_wins = int(np.sum(winner_scores[:comparable] > baseline_scores[:comparable]))

    margin = winner.mean(PRIMARY_METRIC) - best.mean(PRIMARY_METRIC)
    roc_auc = winner.mean("roc_auc")
    mcc = winner.mean("mcc")

    checks: dict[str, dict[str, Any]] = {
        "margin_over_best_baseline": {
            "value": margin,
            "required": settings.min_balanced_accuracy_margin,
            "passed": bool(np.isfinite(margin) and margin >= settings.min_balanced_accuracy_margin),
        },
        "fold_wins": {
            "value": fold_wins,
            "required": settings.min_fold_wins,
            "of_folds": comparable,
            "passed": fold_wins >= settings.min_fold_wins,
        },
        "roc_auc": {
            "value": roc_auc,
            "required": f"> {settings.min_roc_auc}",
            "passed": bool(np.isfinite(roc_auc) and roc_auc > settings.min_roc_auc),
        },
        "mcc": {
            "value": mcc,
            "required": f"> {settings.min_mcc}",
            "passed": bool(np.isfinite(mcc) and mcc > settings.min_mcc),
        },
    }

    failed = [name for name, check in checks.items() if not bool(check["passed"])]
    if failed:
        reason = (
            f"no stable predictive edge: {', '.join(failed)} did not pass. "
            f"Winner {winner.name} scored {winner.mean(PRIMARY_METRIC):.4f} balanced accuracy "
            f"against best baseline {best.name} at {best.mean(PRIMARY_METRIC):.4f} "
            f"(margin {margin:+.4f}, {fold_wins}/{comparable} fold wins)."
        )
    else:
        reason = (
            f"edge detected: {winner.name} beat {best.name} by {margin:+.4f} balanced "
            f"accuracy in {fold_wins}/{comparable} folds, with ROC-AUC {roc_auc:.4f} "
            f"and MCC {mcc:.4f}."
        )

    return EdgeGate(
        edge_detected=not failed,
        checks=checks,
        best_baseline=best.name,
        reason=reason,
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class SelectionOutcome:
    winner: CandidateResult
    ranked: list[CandidateResult]
    baselines: list[CandidateResult]
    edge: EdgeGate
    folds: list[Fold]
    development_end_index: int
    decision: dict[str, Any]

    @property
    def summary_frame(self) -> pd.DataFrame:
        rows = [r.summary_row() for r in self.ranked + self.baselines]
        frame = pd.DataFrame(rows)
        return frame.sort_values(
            ["is_baseline", "balanced_accuracy_mean"], ascending=[True, False]
        ).reset_index(drop=True)

    @property
    def fold_metrics_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.ranked + self.baselines:
            for metrics in result.fold_metrics:
                rows.append({"candidate": result.name, **metrics})
        return pd.DataFrame(rows)

    @property
    def oof_predictions(self) -> pd.DataFrame:
        frames = [r.oof_predictions for r in self.ranked + self.baselines if len(r.oof_predictions)]
        return pd.concat(frames) if frames else pd.DataFrame()


def development_bounds(dataset: Dataset, config: Config) -> tuple[int, int]:
    """Return ``(development_end_exclusive, test_start)``.

    The test window is the final `test_fraction` of rows. `gap` rows between the
    two belong to neither, so a horizon-1 label cannot straddle the boundary.
    """
    n = len(dataset)
    n_development = int(np.floor(n * config.split.development_fraction))
    test_start = n_development + config.split.gap

    if test_start >= n:
        raise ValueError(
            f"no rows left for the final test (n={n}, development={n_development}, gap={config.split.gap})"
        )
    return n_development, test_start


def run_selection(dataset: Dataset, config: Config) -> SelectionOutcome:
    """Evaluate every candidate on shared development folds and pick one winner."""
    n_development, test_start = development_bounds(dataset, config)

    # One shared fold set. Every candidate is scored on exactly these indices.
    folds = walk_forward_folds(
        n_samples=n_development,
        n_splits=config.split.walk_forward_splits,
        gap=config.split.gap,
    )

    max_row_used = int(max(int(f.eval_idx.max()) for f in folds))
    if max_row_used >= test_start:
        raise AssertionError(
            f"selection folds reach row {max_row_used} but the test set starts at "
            f"{test_start}. Selection must never touch the holdout."
        )

    model_results = [evaluate_candidate(spec, dataset, folds, config) for spec in build_candidates(config)]
    baseline_results = [
        evaluate_candidate(spec, dataset, folds, config) for spec in baseline_candidates(config)
    ]

    ranked = rank_candidates(model_results)
    winner = ranked[0]
    edge = evaluate_edge_gate(winner, baseline_results, config)

    fold_dates = [f.dates(dataset.index) for f in folds]
    decision: dict[str, Any] = {
        "selected_candidate": winner.spec.to_dict(),
        "selection_metric": PRIMARY_METRIC,
        "selection_score": winner.mean(PRIMARY_METRIC),
        "selection_score_std": winner.std(PRIMARY_METRIC),
        "ranking": [
            {
                "rank": i + 1,
                "candidate": r.name,
                "balanced_accuracy_mean": r.mean(PRIMARY_METRIC),
                "balanced_accuracy_std": r.std(PRIMARY_METRIC),
                "log_loss_mean": r.mean("log_loss"),
                "complexity_rank": r.spec.complexity_rank,
            }
            for i, r in enumerate(ranked)
        ],
        "baselines": [
            {
                "candidate": r.name,
                "balanced_accuracy_mean": r.mean(PRIMARY_METRIC),
                "balanced_accuracy_std": r.std(PRIMARY_METRIC),
            }
            for r in baseline_results
        ],
        "edge_gate": edge.to_dict(),
        "edge_detected": edge.edge_detected,
        # Proof of the central claim, recorded rather than asserted in prose.
        "test_data_used_for_selection": False,
        "development": {
            "n_rows": n_development,
            "first_date": str(dataset.index[0].date()),
            "last_date": str(dataset.index[n_development - 1].date()),
            "max_row_index_used": max_row_used,
        },
        "held_out_test": {
            "test_start_index": test_start,
            "first_date": str(dataset.index[test_start].date()),
            "last_date": str(dataset.index[-1].date()),
            "n_rows": int(len(dataset) - test_start),
            "opened_during_selection": False,
        },
        "gap_rows": config.split.gap,
        "folds": fold_dates,
        "n_candidates_evaluated": len(model_results),
        "n_baselines_evaluated": len(baseline_results),
        "lstm_available": lstm_available(),
        "config_sha256": config.sha256(),
    }
    decision["decision_sha256"] = sha256_canonical_json(decision)

    return SelectionOutcome(
        winner=winner,
        ranked=ranked,
        baselines=baseline_results,
        edge=edge,
        folds=folds,
        development_end_index=n_development,
        decision=decision,
    )
