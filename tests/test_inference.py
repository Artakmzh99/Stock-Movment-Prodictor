"""Prediction from a saved model (P0.5).

Inference must either produce a correct answer or refuse. A confident number built
on a permuted feature matrix, a stale feature version, or an incomplete session is
worse than an error.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stock_movement.inference import (
    InferenceError,
    build_inference_features,
    load_feature_manifest,
    load_run_config,
    predict,
)
from stock_movement.pipeline import run_all

#: After the NYSE close on Friday 2026-01-02, the last completed session.
AFTER_CLOSE = datetime(2026, 1, 2, 22, 0, tzinfo=UTC)


@pytest.fixture
def completed_run(offline_config):
    _, final = run_all(offline_config, run_id="inference-run")
    return final


def test_predict_loads_saved_model(offline_config, completed_run):
    prediction = predict("inference-run", offline_config, as_of=None, now_utc=AFTER_CLOSE)

    assert 0.0 <= prediction.probability_up <= 1.0
    assert prediction.model_run_id == "inference-run"
    assert prediction.ticker == "TEST"
    assert prediction.model_candidate


def test_predict_output_has_required_fields(offline_config, completed_run):
    prediction = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)
    payload = json.loads(prediction.to_json())

    for field in (
        "signal_date",
        "expected_execution_date",
        "ticker",
        "probability_up",
        "predicted_class",
        "trading_signal",
        "classification_threshold",
        "trading_threshold",
        "model_run_id",
        "model_candidate",
        "feature_version",
    ):
        assert field in payload, field

    assert payload["predicted_class"] in (0, 1)
    assert payload["trading_signal"] in ("long", "cash")


def test_trading_signal_follows_the_trading_threshold(offline_config, completed_run):
    prediction = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)

    expected = "long" if prediction.probability_up >= prediction.trading_threshold else "cash"
    assert prediction.trading_signal == expected
    # Classification and trading thresholds are distinct by design.
    assert prediction.classification_threshold == 0.50
    assert prediction.trading_threshold == 0.55


def test_next_open_execution_dates_the_following_session(offline_config, completed_run):
    prediction = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)

    assert prediction.execution_mode == "next_open"
    assert prediction.expected_execution_date is not None
    assert prediction.expected_execution_date > prediction.signal_date


def test_predict_uses_manifest_feature_order(offline_config, completed_run):
    """Feature order is enforced, not assumed."""
    manifest = load_feature_manifest(completed_run.run.path)
    run_config = load_run_config(completed_run.run.path)

    features, _ = build_inference_features(run_config, manifest, now_utc=AFTER_CLOSE)

    assert list(features.columns) == list(manifest["feature_names"])


def test_predict_rejects_missing_feature(offline_config, completed_run):
    """If the code can no longer build a required feature, refuse to guess."""
    manifest = load_feature_manifest(completed_run.run.path)
    run_config = load_run_config(completed_run.run.path)
    broken = {**manifest, "feature_names": [*manifest["feature_names"], "a_feature_that_never_existed"]}

    with pytest.raises(InferenceError, match="cannot produce"):
        build_inference_features(run_config, broken, now_utc=AFTER_CLOSE)


def test_predict_rejects_incomplete_signal_date(offline_config, completed_run):
    """A weekend is not a session and has no feature row."""
    with pytest.raises(InferenceError, match="not a completed session"):
        predict("inference-run", offline_config, as_of=date(2026, 1, 3), now_utc=AFTER_CLOSE)


def test_predict_rejects_a_future_date(offline_config, completed_run):
    with pytest.raises(InferenceError, match="not a completed session"):
        predict("inference-run", offline_config, as_of=date(2030, 6, 3), now_utc=AFTER_CLOSE)


def test_predict_accepts_an_explicit_historical_session(offline_config, completed_run):
    manifest = load_feature_manifest(completed_run.run.path)
    run_config = load_run_config(completed_run.run.path)
    features, _ = build_inference_features(run_config, manifest, now_utc=AFTER_CLOSE)

    session = features.index[-10]
    prediction = predict("inference-run", offline_config, as_of=session.date(), now_utc=AFTER_CLOSE)

    assert prediction.signal_date == str(session.date())


def test_predict_is_reproducible(offline_config, completed_run):
    first = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)
    second = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)

    assert first.probability_up == second.probability_up
    assert first.signal_date == second.signal_date
    assert first.trading_signal == second.trading_signal


def test_predict_matches_the_saved_test_prediction_for_a_test_session(offline_config, completed_run):
    """Inference must reproduce what the final test recorded for the same row.

    The final test refits on development only, and `predict` loads that same fitted
    model, so a shared date must give an identical probability.
    """
    predictions = pd.read_parquet(completed_run.run.path / "final_test_predictions.parquet")
    session = predictions.index[5]

    live = predict("inference-run", offline_config, as_of=session.date(), now_utc=AFTER_CLOSE)

    assert live.probability_up == pytest.approx(float(predictions.loc[session, "proba_up"]), abs=1e-6)


def test_predict_warns_when_the_final_test_contradicted_the_edge(offline_config, completed_run):
    """A development edge the holdout rejected must be surfaced, not implied away."""
    metadata = json.loads((completed_run.run.path / "model" / "model_metadata.json").read_text())
    prediction = predict("inference-run", offline_config, now_utc=AFTER_CLOSE)

    assert prediction.final_test_confirmed_edge == metadata["final_test_beat_baselines"]

    if metadata["edge_detected"] and metadata["final_test_beat_baselines"] is False:
        assert any("CONTRADICTED" in w for w in prediction.warnings)
    if not metadata["edge_detected"]:
        assert any("edge gate" in w for w in prediction.warnings)


def test_predict_refuses_a_run_without_a_saved_model(offline_config):
    from stock_movement.persistence import ModelPersistenceError
    from stock_movement.pipeline import run_selection_stage

    selection = run_selection_stage(offline_config, run_id="selection-only")

    with pytest.raises(ModelPersistenceError, match="no model metadata"):
        predict("selection-only", offline_config, now_utc=AFTER_CLOSE)
    assert not (selection.run.path / "model").exists()


def test_predict_refuses_a_missing_run(offline_config):
    with pytest.raises(FileNotFoundError):
        predict("no-such-run", offline_config, now_utc=AFTER_CLOSE)


def test_predict_refuses_a_stale_feature_version(offline_config, completed_run, monkeypatch):
    """A model trained on different feature semantics must not be reused."""
    metadata_path = completed_run.run.path / "model" / "model_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["feature_version"] = "v0-ancient"
    metadata_path.write_text(json.dumps(payload))

    with pytest.raises(InferenceError, match="feature_version"):
        predict("inference-run", offline_config, now_utc=AFTER_CLOSE)


def test_predict_refuses_inconsistent_artifacts(offline_config, completed_run):
    """A manifest and model that disagree on features indicate a corrupted run."""
    manifest_path = completed_run.run.path / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_names"] = manifest["feature_names"][:-1]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(InferenceError, match="disagree on the feature list"):
        predict("inference-run", offline_config, now_utc=AFTER_CLOSE)


def test_missing_resolved_config_is_an_error(tmp_path):
    with pytest.raises(InferenceError, match=r"no resolved_config\.json"):
        load_run_config(tmp_path)


def test_missing_feature_manifest_is_an_error(tmp_path):
    with pytest.raises(InferenceError, match=r"no feature_manifest\.json"):
        load_feature_manifest(tmp_path)


def test_inference_features_have_no_label_columns(offline_config, completed_run):
    """No label exists at prediction time; the newest session is the row we predict from."""
    manifest = load_feature_manifest(completed_run.run.path)
    run_config = load_run_config(completed_run.run.path)

    features, _ = build_inference_features(run_config, manifest, now_utc=AFTER_CLOSE)

    assert "target" not in features.columns
    assert "future_return" not in features.columns
    assert not features.isna().to_numpy().any()
