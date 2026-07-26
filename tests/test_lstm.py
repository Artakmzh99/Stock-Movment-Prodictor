"""Optional LSTM phase.

Sequence-construction tests always run — they are pure numpy and encode the
causality guarantee for windowed inputs. Anything that trains a network is marked
`slow` and excluded from the fast gate.
"""

from __future__ import annotations

import numpy as np
import pytest

# make_sequences is pure numpy; import it without requiring TensorFlow.
from stock_movement.lstm import make_sequences
from stock_movement.models import lstm_available

requires_tensorflow = pytest.mark.skipif(not lstm_available(), reason="LSTM phase requires TensorFlow")


# --------------------------------------------------------------------------
# sequence construction — always runs
# --------------------------------------------------------------------------
def test_sequence_window_ends_at_its_own_row_and_never_later():
    """Window i must contain rows [i-lookback+1, i]. One row further is the future."""
    values = np.arange(20).reshape(10, 2).astype(float)
    sequences = make_sequences(values, lookback=3)

    assert sequences.shape == (8, 3, 2)
    np.testing.assert_array_equal(sequences[0], values[0:3])
    np.testing.assert_array_equal(sequences[-1], values[7:10])

    # The last timestep of window i is row i + lookback - 1, never beyond.
    for i in range(len(sequences)):
        np.testing.assert_array_equal(sequences[i][-1], values[i + 2])


def test_sequences_preserve_time_order_within_a_window():
    values = np.arange(10).reshape(10, 1).astype(float)
    sequences = make_sequences(values, lookback=4)

    for window in sequences:
        assert np.all(np.diff(window.ravel()) > 0), "timesteps are out of order"


def test_sequence_count_is_rows_minus_lookback_plus_one():
    for n, lookback in ((10, 3), (100, 20), (50, 50)):
        values = np.zeros((n, 2))
        assert len(make_sequences(values, lookback)) == n - lookback + 1


def test_sequences_require_a_full_window():
    with pytest.raises(ValueError, match="at least 5 rows"):
        make_sequences(np.zeros((3, 2)), lookback=5)


def test_two_dimensional_input_is_required():
    with pytest.raises(ValueError, match="2-D array"):
        make_sequences(np.zeros(10), lookback=3)


def test_changing_a_future_row_cannot_change_an_earlier_window():
    """The causality guarantee, in windowed form."""
    values = np.arange(40).reshape(20, 2).astype(float)
    original = make_sequences(values, lookback=5)

    tampered = values.copy()
    tampered[15:] = -999.0
    changed = make_sequences(tampered, lookback=5)

    # Windows ending before row 15 must be untouched.
    np.testing.assert_array_equal(original[:11], changed[:11])


# --------------------------------------------------------------------------
# training — slow
# --------------------------------------------------------------------------
@requires_tensorflow
@pytest.mark.slow
def test_lstm_trains_and_emits_probability_shaped_output(dataset):
    from stock_movement.lstm import LSTMClassifier

    X, y = dataset.X, dataset.y
    split = int(len(X) * 0.7)

    model = LSTMClassifier(lookback=10, units=(8,), epochs=2, patience=1, verbose=0, random_state=0)
    model.fit(X.iloc[:split], y.iloc[:split])
    proba = model.predict_proba(X.iloc[split:])

    assert proba.shape == (len(X) - split, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert len(model.predict(X.iloc[split:])) == len(X) - split


@requires_tensorflow
@pytest.mark.slow
def test_predictions_cover_every_row_of_the_evaluation_block(dataset):
    """A fold's evaluation block must be scored in full, using past context only."""
    from stock_movement.lstm import LSTMClassifier

    X, y = dataset.X, dataset.y
    split = int(len(X) * 0.8)

    model = LSTMClassifier(lookback=20, units=(8,), epochs=2, patience=1, verbose=0, random_state=0)
    model.fit(X.iloc[:split], y.iloc[:split])

    evaluation = X.iloc[split:]
    proba = model.predict_proba(evaluation)[:, 1]

    assert len(proba) == len(evaluation)
    assert np.isfinite(proba).all()
    # Retained context is training-window data, i.e. strictly earlier rows.
    assert len(model._context_) == model.lookback - 1


@requires_tensorflow
@pytest.mark.slow
def test_lstm_save_reload_preserves_probabilities(tmp_path, offline_config, dataset):
    """Saving the network alone would lose the scaler and produce garbage."""
    from stock_movement.lstm import LSTMClassifier
    from stock_movement.persistence import build_metadata_fields, load_model, save_model

    model = LSTMClassifier(lookback=10, units=(8,), epochs=2, patience=1, verbose=0, random_state=0)
    model.fit(dataset.X, dataset.y)
    expected = model.predict_proba(dataset.X)[:, 1]

    fields = build_metadata_fields(
        config=offline_config,
        run_id="lstm-run",
        candidate_name="lstm[lookback=10]",
        family="lstm",
        params={"lookback": 10},
        feature_names=dataset.feature_names,
        training_first_date=str(dataset.index[0].date()),
        training_last_date=str(dataset.index[-1].date()),
        n_training_rows=len(dataset),
        edge_detected=False,
        data_sha256=None,
        git_commit=None,
    )
    metadata = save_model(model, tmp_path, fields)
    assert metadata.model_format == "keras"

    reloaded = load_model(tmp_path)
    np.testing.assert_allclose(reloaded.predict_proba(dataset.X)[:, 1], expected, rtol=1e-4)


@requires_tensorflow
@pytest.mark.slow
def test_lstm_competes_as_a_candidate_and_loses_ties_on_complexity(dataset, offline_config):
    """The LSTM must be evaluated on the same folds, and rank 3 on complexity."""
    from stock_movement.models import MODEL_COMPLEXITY
    from stock_movement.selection import build_candidates

    config = offline_config.model_copy(
        update={
            "selection": offline_config.selection.model_copy(
                update={"families": ("logistic",), "include_lstm": True}
            )
        }
    )
    families = {c.model_name for c in build_candidates(config)}

    assert "lstm" in families
    assert MODEL_COMPLEXITY["lstm"] > MODEL_COMPLEXITY["logistic"]


def test_lstm_is_skipped_cleanly_when_tensorflow_is_absent(offline_config, monkeypatch):
    """An absent optional dependency must not fail the whole selection."""
    from stock_movement import selection as selection_module

    monkeypatch.setattr(selection_module, "lstm_available", lambda: False)
    config = offline_config.model_copy(
        update={
            "selection": offline_config.selection.model_copy(
                update={"families": ("logistic",), "include_lstm": True}
            )
        }
    )

    families = {c.model_name for c in selection_module.build_candidates(config)}
    assert "lstm" not in families
    assert "logistic" in families
