"""Model persistence (P0.4).

The behaviour under test is that a save which cannot be reloaded is a *failed run*,
not a warning. The previous implementation caught every exception and returned
None, so a run could report metrics and leave nothing usable behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stock_movement.config import FEATURE_VERSION, Config
from stock_movement.models import build_model
from stock_movement.persistence import (
    METADATA_FILE,
    MODEL_DIR,
    SKLEARN_MODEL_FILE,
    ModelPersistenceError,
    build_metadata_fields,
    load_model,
    load_model_metadata,
    save_model,
    verify_model_artifacts,
)


def _fields(config: Config, dataset: Any, **overrides: Any) -> dict[str, Any]:
    payload = build_metadata_fields(
        config=config,
        run_id="test-run",
        candidate_name="logistic[C=1.0]",
        family="logistic",
        params={"C": 1.0},
        feature_names=dataset.feature_names,
        training_first_date=str(dataset.index[0].date()),
        training_last_date=str(dataset.index[-1].date()),
        n_training_rows=len(dataset),
        edge_detected=False,
        data_sha256="abc123",
        git_commit="deadbeef",
        final_test_beat_baselines=False,
        final_test_balanced_accuracy=0.49,
        final_test_best_baseline_balanced_accuracy=0.52,
    )
    payload.update(overrides)
    return payload


def _fit(dataset: Any, family: str = "logistic", params: dict[str, Any] | None = None) -> Any:
    model = build_model(family, params or {}, random_state=42)
    model.fit(dataset.X, dataset.y)
    return model


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------
def test_logistic_save_reload_preserves_probabilities(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    expected = model.predict_proba(dataset.X)[:, 1]

    save_model(model, tmp_path, _fields(offline_config, dataset))
    reloaded = load_model(tmp_path)

    np.testing.assert_allclose(reloaded.predict_proba(dataset.X)[:, 1], expected, rtol=1e-12)


def test_hgb_save_reload_preserves_probabilities(tmp_path, offline_config, dataset):
    model = _fit(dataset, "hist_gradient_boosting", {"max_iter": 30})
    expected = model.predict_proba(dataset.X)[:, 1]

    save_model(model, tmp_path, _fields(offline_config, dataset, family="hist_gradient_boosting"))
    reloaded = load_model(tmp_path)

    np.testing.assert_allclose(reloaded.predict_proba(dataset.X)[:, 1], expected, rtol=1e-12)


def test_saved_pipeline_carries_its_own_preprocessing(tmp_path, offline_config, dataset):
    """Reloading must restore the fitted scaler, not a fresh one."""
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    reloaded = load_model(tmp_path)
    original_scaler = model.named_steps["scaler"]
    reloaded_scaler = reloaded.named_steps["scaler"]

    np.testing.assert_allclose(reloaded_scaler.mean_, original_scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose(reloaded_scaler.scale_, original_scaler.scale_, rtol=1e-12)


# --------------------------------------------------------------------------
# failures must fail the run
# --------------------------------------------------------------------------
def test_model_save_failure_fails_run(tmp_path, offline_config, dataset, monkeypatch):
    """An unpicklable estimator must raise, not be silently skipped."""
    model = _fit(dataset, "logistic")

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("joblib.dump", explode)

    with pytest.raises(ModelPersistenceError, match="failed to save"):
        save_model(model, tmp_path, _fields(offline_config, dataset))


def test_save_verifies_the_round_trip(tmp_path, offline_config, dataset, monkeypatch):
    """A file written but not reloadable counts as a save failure."""
    model = _fit(dataset, "logistic")

    real_load = __import__("joblib").load
    calls = {"n": 0}

    def flaky_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("corrupt archive")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr("joblib.load", flaky_load)

    with pytest.raises(ModelPersistenceError, match="could not be reloaded"):
        save_model(model, tmp_path, _fields(offline_config, dataset))


def test_loading_from_a_run_without_a_model_raises(tmp_path):
    with pytest.raises(ModelPersistenceError, match="no model metadata"):
        load_model(tmp_path)


def test_unknown_model_format_is_rejected(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    metadata_path = tmp_path / MODEL_DIR / METADATA_FILE
    payload = __import__("json").loads(metadata_path.read_text())
    payload["model_format"] = "telepathy"
    metadata_path.write_text(__import__("json").dumps(payload))

    with pytest.raises(ModelPersistenceError, match="unknown model_format"):
        load_model(tmp_path)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------
def test_model_metadata_matches_feature_manifest(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    metadata = load_model_metadata(tmp_path)

    assert metadata.feature_names == dataset.feature_names
    assert metadata.feature_version == FEATURE_VERSION
    assert metadata.n_training_rows == len(dataset)
    assert metadata.target_definition == offline_config.labels.target_definition
    assert metadata.execution_mode == offline_config.backtest.execution_mode
    assert metadata.config_sha256 == offline_config.sha256()


def test_metadata_records_feature_order_not_just_the_set(tmp_path, offline_config, dataset):
    """Order matters: a permuted matrix would produce confident nonsense."""
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    metadata = load_model_metadata(tmp_path)
    assert metadata.feature_names == list(dataset.X.columns)


def test_metadata_records_both_thresholds(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    metadata = load_model_metadata(tmp_path)
    assert metadata.classification_threshold == offline_config.threshold.classification_value
    assert metadata.trading_threshold == offline_config.threshold.trading_value


def test_metadata_records_the_final_test_outcome(tmp_path, offline_config, dataset):
    """A development edge the holdout contradicted must be recorded, not hidden."""
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    metadata = load_model_metadata(tmp_path)
    assert metadata.final_test_beat_baselines is False
    assert metadata.final_test_balanced_accuracy == pytest.approx(0.49)
    assert metadata.final_test_best_baseline_balanced_accuracy == pytest.approx(0.52)


def test_artifact_digests_are_recorded_and_verifiable(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    metadata = save_model(model, tmp_path, _fields(offline_config, dataset))

    assert SKLEARN_MODEL_FILE in metadata.artifact_files
    assert all(verify_model_artifacts(tmp_path).values())


def test_tampering_with_a_saved_model_is_detected(tmp_path, offline_config, dataset):
    model = _fit(dataset, "logistic")
    save_model(model, tmp_path, _fields(offline_config, dataset))

    target: Path = tmp_path / MODEL_DIR / SKLEARN_MODEL_FILE
    target.write_bytes(target.read_bytes() + b"tampered")

    assert verify_model_artifacts(tmp_path)[SKLEARN_MODEL_FILE] is False
