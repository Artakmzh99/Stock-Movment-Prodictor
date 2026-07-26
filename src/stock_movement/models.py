"""Model factory, complexity ranking, and deterministic tie-breaking.

Every model is an sklearn ``Pipeline``. That is not cosmetic: putting the imputer
and scaler *inside* the pipeline is what guarantees they are re-fitted on each
fold's training rows only. A scaler fitted once on the whole dataset would leak
the final test set's mean and variance into training — invisible in the code,
fatal to the result.

Two orderings are defined here and used by ``selection``:

**Complexity** ranks families. When two candidates are statistically
indistinguishable the simpler family wins, because a tie on ~3,000 noisy rows is
not evidence for the more flexible model.

**Simplicity within a family** breaks remaining ties deterministically — smaller
`C`, shallower trees, larger leaves, more regularisation. Without this, "best"
depends on dict iteration order, and a rerun can silently pick a different model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC = "logistic"
HGB = "hist_gradient_boosting"
RANDOM_FOREST = "random_forest"
LSTM = "lstm"

#: Lower is simpler. Ties in performance are resolved toward the simpler family.
MODEL_COMPLEXITY: dict[str, int] = {LOGISTIC: 1, HGB: 2, RANDOM_FOREST: 2, LSTM: 3}


def _logistic(params: dict[str, Any], random_state: int) -> Pipeline:
    # `penalty` is deliberately omitted: sklearn 1.8 deprecated it in favour of
    # `l1_ratio`, and the default is already L2.
    defaults: dict[str, Any] = {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
    }
    defaults.update(params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state, **defaults)),
        ]
    )


def _hist_gradient_boosting(params: dict[str, Any], random_state: int) -> Pipeline:
    defaults: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
    defaults.update(params)
    # Trees need neither imputation (native NaN support) nor scaling; the pipeline
    # wrapper keeps the interface uniform across families.
    return Pipeline(steps=[("model", HistGradientBoostingClassifier(random_state=random_state, **defaults))])


def _random_forest(params: dict[str, Any], random_state: int) -> Pipeline:
    defaults: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 6,
        "min_samples_leaf": 30,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    }
    defaults.update(params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=random_state, **defaults)),
        ]
    )


MODEL_REGISTRY = {
    LOGISTIC: _logistic,
    HGB: _hist_gradient_boosting,
    RANDOM_FOREST: _random_forest,
}


def build_model(family: str, params: dict[str, Any] | None = None, random_state: int = 42) -> Any:
    """Construct a fresh, unfitted estimator for `family`."""
    params = dict(params or {})

    if family == LSTM:
        from .lstm import build_lstm_estimator  # imported lazily; needs tensorflow

        return build_lstm_estimator(params, random_state=random_state)

    if family not in MODEL_REGISTRY:
        raise ValueError(f"unknown model family {family!r}; available: {[*MODEL_REGISTRY, LSTM]}")
    return MODEL_REGISTRY[family](params, random_state)


def lstm_available() -> bool:
    """Whether the optional LSTM dependency is importable."""
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# candidate grids
# --------------------------------------------------------------------------
def default_param_grid(family: str) -> list[dict[str, Any]]:
    """Deliberately small grids.

    With a few thousand noisy rows, a large search finds the configuration that
    best fits *this* history, not the one that generalises. Every extra candidate
    also raises the chance that one looks good by luck.
    """
    if family == LOGISTIC:
        return [
            {"C": 0.001, "class_weight": None},
            {"C": 0.01, "class_weight": None},
            {"C": 0.01, "class_weight": "balanced"},
            {"C": 0.1, "class_weight": "balanced"},
            {"C": 1.0, "class_weight": "balanced"},
        ]
    if family == HGB:
        return [
            {"learning_rate": 0.03, "max_leaf_nodes": 7, "min_samples_leaf": 60, "max_iter": 150},
            {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 40, "max_iter": 200},
            {"learning_rate": 0.10, "max_leaf_nodes": 7, "min_samples_leaf": 60, "max_iter": 150},
        ]
    if family == RANDOM_FOREST:
        return [{"max_depth": 3}, {"max_depth": 6}]
    if family == LSTM:
        # Sequence models are slow; one deliberate configuration, not a grid.
        return [{"lookback": 20, "units": (32, 16), "dropout": 0.2}]
    return [{}]


# --------------------------------------------------------------------------
# deterministic tie-breaking
# --------------------------------------------------------------------------
def simplicity_key(family: str, params: dict[str, Any]) -> tuple[float, ...]:
    """Sort key where *smaller means simpler*, used only to break exact ties."""
    if family == LOGISTIC:
        return (
            float(params.get("C", 1.0)),  # smaller C == stronger regularisation
            0.0 if params.get("class_weight") is None else 1.0,
        )
    if family == HGB:
        return (
            float(params.get("max_leaf_nodes", 15)),
            float(params.get("max_iter", 200)),
            -float(params.get("min_samples_leaf", 40)),  # larger leaves are simpler
            -float(params.get("l2_regularization", 1.0)),  # more regularisation is simpler
            float(params.get("learning_rate", 0.05)),
        )
    if family == RANDOM_FOREST:
        return (
            float(params.get("max_depth", 6)),
            float(params.get("n_estimators", 400)),
            -float(params.get("min_samples_leaf", 30)),
        )
    if family == LSTM:
        units = params.get("units", (32, 16))
        return (float(params.get("lookback", 20)), float(sum(units)), float(len(units)))
    return (0.0,)


def extract_feature_importance(
    estimator: Any,
    feature_names: list[str],
    X: Any = None,
    y: Any = None,
    random_state: int = 42,
) -> dict[str, float] | None:
    """Per-feature importance, by the best method the estimator supports.

    1. ``coef_`` — signed logistic coefficients.
    2. ``feature_importances_`` — impurity-based gains (random forests).
    3. permutation importance — needed for ``HistGradientBoostingClassifier``,
       which exposes *neither* of the above. Without this fallback the gradient
       boosting model silently reported no importance at all.

    Returns None when no method applies and no data is supplied. Inventing a
    number would be worse than omitting it.
    """
    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
    if model is None:
        return None

    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_).ravel()
        return {name: float(v) for name, v in zip(feature_names, coefficients, strict=False)}

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).ravel()
        return {name: float(v) for name, v in zip(feature_names, values, strict=False)}

    if X is None or y is None:
        return None

    # Permutation importance is purely descriptive: it is computed after the model
    # is frozen and feeds into no decision, so it cannot influence selection.
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            estimator, X, y, n_repeats=5, random_state=random_state, scoring="balanced_accuracy"
        )
    except Exception:
        return None
    return {
        name: float(v)
        for name, v in zip(feature_names, np.asarray(result.importances_mean).ravel(), strict=False)
    }
