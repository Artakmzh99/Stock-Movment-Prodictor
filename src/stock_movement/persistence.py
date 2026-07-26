"""Model persistence.

A run that reports a metric but saves no usable model has not produced a model —
it has produced a number. So saving is part of the run, and **a failed save fails
the run**. The previous behaviour swallowed the exception and returned None, which
meant a run could look completely successful and leave nothing to predict with.

Two storage layouts:

* **sklearn** — the whole fitted ``Pipeline`` in ``model.joblib``. The pipeline
  carries its own imputer and scaler, so reloading restores the exact transform
  chain that was fitted, not a reconstruction of it.
* **Keras/LSTM** — ``model.keras`` for the network, plus ``imputer.joblib`` and
  ``scaler.joblib`` separately, because those are sklearn objects living on the
  wrapper rather than inside the Keras graph. Saving the network alone would
  silently lose the scaling and produce garbage predictions on reload.

``model_metadata.json`` records the exact ordered feature list. Inference reindexes
to it; a permuted column order otherwise yields confident nonsense rather than an
error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import FEATURE_VERSION, Config
from .provenance import package_versions, sha256_file, utc_now_iso

MODEL_DIR = "model"
SKLEARN_MODEL_FILE = "model.joblib"
KERAS_MODEL_FILE = "model.keras"
IMPUTER_FILE = "imputer.joblib"
SCALER_FILE = "scaler.joblib"
METADATA_FILE = "model_metadata.json"


class ModelPersistenceError(RuntimeError):
    """Saving or loading a model failed. Never swallowed."""


@dataclass
class ModelMetadata:
    """Everything needed to use a saved model correctly, or refuse to."""

    run_id: str
    model_format: str  # "sklearn" | "keras"
    family: str
    candidate_name: str
    params: dict[str, Any]
    feature_names: list[str]  # exact order required at inference time
    feature_version: str
    target_definition: str
    horizon: int
    execution_mode: str
    classification_threshold: float
    trading_threshold: float
    ticker: str
    training_first_date: str
    training_last_date: str
    n_training_rows: int
    #: Development-only gate outcome (see selection.evaluate_edge_gate).
    edge_detected: bool
    #: Whether the *final test* confirmed it. A development edge that the holdout
    #: contradicts is the normal case, and inference must say so rather than let
    #: `edge_detected` imply a working model.
    final_test_beat_baselines: bool | None
    final_test_balanced_accuracy: float | None
    final_test_best_baseline_balanced_accuracy: float | None
    config_sha256: str
    data_sha256: str | None
    git_commit: str | None
    package_versions: dict[str, str]
    saved_at_utc: str
    artifact_files: dict[str, str]  # filename -> sha256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_dir(run_path: Path) -> Path:
    target = run_path / MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def is_keras_estimator(estimator: Any) -> bool:
    return hasattr(estimator, "model_") and estimator.__class__.__name__ == "LSTMClassifier"


def save_model(
    estimator: Any,
    run_path: Path,
    metadata_fields: dict[str, Any],
) -> ModelMetadata:
    """Persist a fitted estimator plus its metadata. Raises on any failure."""
    directory = _model_dir(run_path)
    artifact_files: dict[str, str] = {}

    try:
        import joblib

        if is_keras_estimator(estimator):
            model_format = "keras"
            estimator.model_.save(directory / KERAS_MODEL_FILE)
            # The scaler and imputer live on the wrapper, not in the Keras graph.
            joblib.dump(estimator._imputer, directory / IMPUTER_FILE)
            joblib.dump(estimator._scaler, directory / SCALER_FILE)
            joblib.dump(
                {
                    "lookback": estimator.lookback,
                    "context": estimator._context_,
                    "train_prior": estimator._train_prior_,
                    "units": estimator.units,
                    "dropout": estimator.dropout,
                },
                directory / "lstm_state.joblib",
            )
            for name in (KERAS_MODEL_FILE, IMPUTER_FILE, SCALER_FILE, "lstm_state.joblib"):
                artifact_files[name] = sha256_file(directory / name)
        else:
            model_format = "sklearn"
            joblib.dump(estimator, directory / SKLEARN_MODEL_FILE)
            artifact_files[SKLEARN_MODEL_FILE] = sha256_file(directory / SKLEARN_MODEL_FILE)

    except Exception as exc:
        raise ModelPersistenceError(
            f"failed to save the selected model to {directory}: {exc}. "
            "A run that cannot persist its model has not produced a usable result."
        ) from exc

    metadata = ModelMetadata(
        model_format=model_format,
        feature_version=FEATURE_VERSION,
        package_versions=package_versions(),
        saved_at_utc=utc_now_iso(),
        artifact_files=artifact_files,
        **metadata_fields,
    )

    metadata_path = directory / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2, default=str))

    # Verify the round trip immediately: a file that cannot be reloaded is not saved.
    try:
        reloaded = load_model(run_path)
    except Exception as exc:
        raise ModelPersistenceError(
            f"model was written to {directory} but could not be reloaded: {exc}"
        ) from exc
    if reloaded is None:  # pragma: no cover - defensive
        raise ModelPersistenceError(f"model reload from {directory} returned nothing")

    return metadata


def load_model_metadata(run_path: Path) -> ModelMetadata:
    path = Path(run_path) / MODEL_DIR / METADATA_FILE
    if not path.exists():
        raise ModelPersistenceError(
            f"no model metadata at {path}. This run has no saved model — it may predate "
            "model persistence, or `final-test` was never run."
        )
    payload = json.loads(path.read_text())
    known = set(ModelMetadata.__dataclass_fields__)
    return ModelMetadata(**{k: v for k, v in payload.items() if k in known})


def load_model(run_path: Path) -> Any:
    """Reload a saved estimator. Raises rather than returning a silent None."""
    run_path = Path(run_path)
    metadata = load_model_metadata(run_path)
    directory = run_path / MODEL_DIR

    try:
        import joblib

        if metadata.model_format == "sklearn":
            return joblib.load(directory / SKLEARN_MODEL_FILE)

        if metadata.model_format == "keras":
            import tensorflow as tf

            from .lstm import LSTMClassifier

            state = joblib.load(directory / "lstm_state.joblib")
            estimator = LSTMClassifier(
                lookback=state["lookback"], units=state["units"], dropout=state["dropout"]
            )
            estimator.model_ = tf.keras.models.load_model(directory / KERAS_MODEL_FILE)
            estimator._imputer = joblib.load(directory / IMPUTER_FILE)
            estimator._scaler = joblib.load(directory / SCALER_FILE)
            estimator._context_ = state["context"]
            estimator._train_prior_ = state["train_prior"]
            estimator.classes_ = __import__("numpy").array([0, 1])
            return estimator

        raise ModelPersistenceError(f"unknown model_format {metadata.model_format!r}")

    except ModelPersistenceError:
        raise
    except Exception as exc:
        raise ModelPersistenceError(f"failed to load the model from {directory}: {exc}") from exc


def verify_model_artifacts(run_path: Path) -> dict[str, bool]:
    """Check every saved artifact still matches its recorded digest."""
    metadata = load_model_metadata(run_path)
    directory = Path(run_path) / MODEL_DIR

    results: dict[str, bool] = {}
    for name, expected in metadata.artifact_files.items():
        path = directory / name
        if not path.exists():
            results[name] = False
            continue
        # Keras saves a directory in some versions; digest only regular files.
        results[name] = sha256_file(path) == expected if path.is_file() else True
    return results


def build_metadata_fields(
    config: Config,
    run_id: str,
    candidate_name: str,
    family: str,
    params: dict[str, Any],
    feature_names: list[str],
    training_first_date: str,
    training_last_date: str,
    n_training_rows: int,
    edge_detected: bool,
    data_sha256: str | None,
    git_commit: str | None,
    final_test_beat_baselines: bool | None = None,
    final_test_balanced_accuracy: float | None = None,
    final_test_best_baseline_balanced_accuracy: float | None = None,
) -> dict[str, Any]:
    return {
        "final_test_beat_baselines": final_test_beat_baselines,
        "final_test_balanced_accuracy": final_test_balanced_accuracy,
        "final_test_best_baseline_balanced_accuracy": final_test_best_baseline_balanced_accuracy,
        "run_id": run_id,
        "family": family,
        "candidate_name": candidate_name,
        "params": params,
        "feature_names": list(feature_names),
        "target_definition": config.labels.target_definition,
        "horizon": config.labels.horizon,
        "execution_mode": config.backtest.execution_mode,
        "classification_threshold": config.threshold.classification_value,
        "trading_threshold": config.threshold.trading_value,
        "ticker": config.data.ticker,
        "training_first_date": training_first_date,
        "training_last_date": training_last_date,
        "n_training_rows": n_training_rows,
        "edge_detected": edge_detected,
        "config_sha256": config.sha256(),
        "data_sha256": data_sha256,
        "git_commit": git_commit,
    }
