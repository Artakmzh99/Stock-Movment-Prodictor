"""Optional LSTM classifier (Phase 7).

This is the last thing to try, not the first. The reference project it derives
from predicts the *price* with an LSTM and reports MAE — which looks impressive
because prices are autocorrelated, and says nothing about direction. Here the
same architecture family is repurposed for classification:

| Reference (regression) | Here (classification)     |
|------------------------|---------------------------|
| Dense(1), linear       | Dense(1), sigmoid         |
| MSE loss               | binary crossentropy       |
| target = price         | target = 0/1 direction    |
| MAE                    | AUC / balanced accuracy   |
| 70/30 split            | walk-forward, gapped      |
| scaler on all data     | scaler per training fold  |

Two details that are easy to get wrong and are handled explicitly:

**Sequence context across a fold boundary.** Predicting row `i` needs the
`lookback` rows ending at `i`. For the first rows of an evaluation block those
rows sit in the training block. That is legitimate — they are strictly in the
past — so ``fit`` retains the tail of its training window and prepends it at
predict time. It never uses a row at or after the row being predicted.

**shuffle=False.** Keras shuffles by default. On sequential data that does not
leak by itself, but combined with a validation split taken from the *end* of the
array it silently mixes regimes; both are disabled here.

TensorFlow is an optional dependency: ``uv sync --all-extras``.

**Where the dependency is checked.** ``_import_tensorflow`` is the single place
importability is determined, and both ``tensorflow_available`` (used by
``selection`` to skip the family) and ``require_tensorflow`` (used at runtime)
route through it, so they cannot disagree. ``build_lstm_estimator`` checks
eagerly, because a factory should not hand back a model that is certain to fail;
``LSTMClassifier.__init__`` does not, because scikit-learn requires it to store
parameters only so that ``clone`` keeps working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TENSORFLOW_MISSING_MESSAGE = (
    "the LSTM model needs TensorFlow, which is an optional dependency.\n"
    '  uv sync --all-extras   (or: pip install -e ".[lstm]")'
)


def _import_tensorflow() -> tuple[Any | None, BaseException | None]:
    """Attempt the import once, and report the outcome rather than deciding it.

    This is the **single** place TensorFlow's importability is determined.
    ``tensorflow_available`` and ``require_tensorflow`` both route through it, so
    the "can we?" answer and the "then do it" answer cannot diverge.

    They previously did diverge: the availability check caught ``Exception`` while
    the runtime check caught only ``ImportError``. TensorFlow really does fail at
    import with ``RuntimeError`` or ``OSError`` on protobuf and NumPy ABI
    mismatches, so in those environments the family was correctly skipped but a
    direct ``fit`` leaked a raw error instead of the actionable message below.

    Catching ``BaseException`` is deliberate and not over-broad here: for our
    purposes "TensorFlow is unusable" is the same answer however it fails, and a
    partially-initialised TF module is worse than none.
    """
    try:
        import tensorflow as tf
    except BaseException as exc:
        return None, exc
    return tf, None


def tensorflow_available() -> bool:
    """Whether TensorFlow can actually be imported in this environment."""
    module, _ = _import_tensorflow()
    return module is not None


def require_tensorflow() -> Any:
    """Return the TensorFlow module, or raise a clear, actionable ImportError."""
    module, error = _import_tensorflow()
    if module is None:
        raise ImportError(TENSORFLOW_MISSING_MESSAGE) from error
    return module


def make_sequences(values: np.ndarray, lookback: int) -> np.ndarray:
    """Turn a 2-D array into overlapping windows of shape (n, lookback, n_features).

    Window `i` ends at row `i`, so it contains rows ``[i - lookback + 1, i]`` and
    never row ``i + 1``. The first ``lookback - 1`` rows produce no window.
    """
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    n_rows = len(values)
    if n_rows < lookback:
        raise ValueError(f"need at least {lookback} rows to build one sequence, got {n_rows}")

    windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
    # sliding_window_view yields (n, n_features, lookback); models want time first.
    ordered: np.ndarray = np.ascontiguousarray(windows.transpose(0, 2, 1))
    return ordered


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Keras LSTM wrapped in the sklearn interface the rest of the project expects."""

    def __init__(
        self,
        lookback: int = 20,
        units: tuple[int, ...] = (32, 16),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 8,
        validation_fraction: float = 0.15,
        class_weight_balanced: bool = True,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.class_weight_balanced = class_weight_balanced
        self.random_state = random_state
        self.verbose = verbose

    # -- internals --------------------------------------------------------
    def _build_network(self, n_features: int) -> Any:
        tf = require_tensorflow()
        tf.keras.utils.set_random_seed(self.random_state)

        layers = [tf.keras.layers.Input(shape=(self.lookback, n_features))]
        for i, size in enumerate(self.units):
            layers.append(tf.keras.layers.LSTM(size, return_sequences=i < len(self.units) - 1))
            layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def _preprocess(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if fit:
            self._imputer = SimpleImputer(strategy="median").fit(values)
            imputed = self._imputer.transform(values)
            self._scaler = StandardScaler().fit(imputed)
            scaled: np.ndarray = self._scaler.transform(imputed)
            return scaled
        transformed: np.ndarray = self._scaler.transform(self._imputer.transform(values))
        return transformed

    # -- sklearn API ------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any) -> LSTMClassifier:
        tf = require_tensorflow()
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        scaled = self._preprocess(X, fit=True)
        sequences = make_sequences(scaled, self.lookback)
        targets = y[self.lookback - 1 :]

        # Retained so evaluation blocks can be given genuine past context.
        self._context_ = (
            scaled[-(self.lookback - 1) :] if self.lookback > 1 else np.empty((0, scaled.shape[1]))
        )
        self._train_prior_ = float(y.mean())

        class_weight = None
        if self.class_weight_balanced:
            counts = np.bincount(targets, minlength=2).astype(float)
            counts[counts == 0] = 1.0
            class_weight = {i: len(targets) / (2.0 * counts[i]) for i in (0, 1)}

        self.model_ = self._build_network(scaled.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if self.validation_fraction > 0 else "loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]

        # The validation slice is the tail of the training window — the most
        # recent data, which is the only sensible choice for a time series.
        self.history_ = self.model_.fit(
            sequences,
            targets,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_fraction,
            shuffle=False,  # never shuffle sequential data
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scaled = self._preprocess(X, fit=False)
        n_rows = len(scaled)

        padded = np.vstack([self._context_, scaled]) if len(self._context_) else scaled
        available = len(padded) - self.lookback + 1

        if available <= 0:
            p_up = np.full(n_rows, self._train_prior_)
        else:
            sequences = make_sequences(padded, self.lookback)
            predicted = self.model_.predict(sequences, verbose=0).ravel()
            p_up = np.full(n_rows, self._train_prior_)
            p_up[n_rows - len(predicted) :] = predicted[-n_rows:] if len(predicted) >= n_rows else predicted

        p_up = np.clip(p_up, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p_up, p_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_lstm_estimator(params: dict[str, Any], random_state: int = 42) -> LSTMClassifier:
    """Factory entry point for the LSTM family.

    This checks for TensorFlow **eagerly**, unlike ``LSTMClassifier.__init__``.
    The two boundaries have different jobs:

    * ``build_lstm_estimator`` exists only to hand back a model that is about to be
      fitted, so returning an object guaranteed to fail later is strictly worse
      than refusing now — the traceback would point at ``fit`` rather than at the
      missing dependency.
    * ``LSTMClassifier.__init__`` must stay side-effect free and store parameters
      only. That is scikit-learn's estimator contract, and ``sklearn.base.clone``
      (used per fold in ``selection``) depends on it. Importing TensorFlow there
      would break cloning and ``get_params``.

    So ``build_model("lstm")`` fails fast, while constructing the class directly
    still works and defers the error to ``fit``.
    """
    require_tensorflow()
    defaults: dict[str, Any] = {"lookback": 20, "random_state": random_state}
    defaults.update(params or {})
    if "units" in defaults and isinstance(defaults["units"], list):
        defaults["units"] = tuple(defaults["units"])
    return LSTMClassifier(**defaults)
