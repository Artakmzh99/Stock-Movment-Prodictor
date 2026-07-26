"""Baseline definitions (P1.1).

Each baseline must report its *own* hard rule and a calibrated probability, and the
two must stay consistent with each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_movement.baselines import (
    BASELINE_PREFIX,
    AlwaysUpBaseline,
    LastDirectionBaseline,
    MajorityClassBaseline,
    RandomBaseline,
    build_baselines,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame(
        {
            "open_to_close_return": [0.01, -0.01, 0.02, -0.02, 0.01, 0.01, -0.01, 0.03, -0.03, 0.01],
            "return_1d": [0.01, -0.01, 0.02, -0.02, 0.01, 0.01, -0.01, 0.03, -0.03, 0.01],
            "other": np.arange(10, dtype=float),
        },
        index=index,
    )


@pytest.fixture
def y() -> np.ndarray:
    #  7 up, 3 down -> prior 0.7
    return np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])


# --------------------------------------------------------------------------
# majority
# --------------------------------------------------------------------------
def test_majority_predicts_the_training_majority(frame, y):
    model = MajorityClassBaseline().fit(frame, y)

    np.testing.assert_array_equal(model.predict(frame), np.ones(10, dtype=int))
    np.testing.assert_allclose(model.predict_proba(frame)[:, 1], np.full(10, 0.7))


def test_majority_follows_a_down_skewed_prior(frame):
    y = np.array([0] * 8 + [1] * 2)  # prior 0.2
    model = MajorityClassBaseline().fit(frame, y)

    np.testing.assert_array_equal(model.predict(frame), np.zeros(10, dtype=int))
    np.testing.assert_allclose(model.predict_proba(frame)[:, 1], np.full(10, 0.2))


def test_majority_hard_rule_is_not_a_threshold_on_its_probability(frame):
    """With a prior of 0.2, thresholding at 0.5 happens to agree — but with 0.45 it
    would not, and the hard rule must win either way."""
    y = np.array([0] * 6 + [1] * 4)  # prior 0.4
    model = MajorityClassBaseline().fit(frame, y)

    assert model.predict_proba(frame)[0, 1] == pytest.approx(0.4)
    assert model.predict(frame)[0] == 0
    assert model.majority_class_ == 0


# --------------------------------------------------------------------------
# always up
# --------------------------------------------------------------------------
def test_always_up_predicts_one_regardless_of_prior(frame):
    """Even with a down-skewed prior, "always up" must still predict up."""
    y = np.array([0] * 9 + [1])  # prior 0.1
    model = AlwaysUpBaseline().fit(frame, y)

    np.testing.assert_array_equal(model.predict(frame), np.ones(10, dtype=int))
    np.testing.assert_allclose(model.predict_proba(frame)[:, 1], np.full(10, 0.1))


def test_always_up_probability_keeps_log_loss_finite(frame, y):
    """Emitting a hard 1.0 would make log loss infinite on the first down session."""
    from sklearn.metrics import log_loss

    model = AlwaysUpBaseline().fit(frame, y)
    proba = model.predict_proba(frame)[:, 1]

    assert np.isfinite(log_loss(y, proba, labels=[0, 1]))
    assert proba.max() < 1.0


# --------------------------------------------------------------------------
# random
# --------------------------------------------------------------------------
def test_random_probability_equals_training_prior(frame, y):
    """Regression test: it used to emit uniform noise, wrecking its log loss."""
    model = RandomBaseline(random_state=42).fit(frame, y)
    proba = model.predict_proba(frame)[:, 1]

    np.testing.assert_allclose(proba, np.full(10, 0.7))
    assert proba.std() == pytest.approx(0.0)


def test_random_probability_beats_uniform_noise_on_log_loss(frame, y):
    from sklearn.metrics import log_loss

    prior_based = log_loss(
        y, RandomBaseline(random_state=42).fit(frame, y).predict_proba(frame)[:, 1], labels=[0, 1]
    )
    uniform = log_loss(y, np.random.default_rng(0).uniform(0.01, 0.99, 10), labels=[0, 1])

    assert prior_based < uniform


def test_random_hard_predictions_are_seeded_and_bernoulli(frame, y):
    first = RandomBaseline(random_state=42).fit(frame, y).predict(frame)
    second = RandomBaseline(random_state=42).fit(frame, y).predict(frame)
    other = RandomBaseline(random_state=7).fit(frame, y).predict(frame)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert set(np.unique(first)) <= {0, 1}


def test_random_hard_rate_approaches_the_prior():
    index = pd.bdate_range("2020-01-01", periods=5000)
    big = pd.DataFrame({"open_to_close_return": np.zeros(5000)}, index=index)
    y = np.concatenate([np.ones(3000), np.zeros(2000)])  # prior 0.6

    predictions = RandomBaseline(random_state=42).fit(big, y).predict(big)

    assert predictions.mean() == pytest.approx(0.6, abs=0.02)


# --------------------------------------------------------------------------
# last direction
# --------------------------------------------------------------------------
def test_last_direction_follows_the_current_session_sign(frame, y):
    model = LastDirectionBaseline(feature="open_to_close_return").fit(frame, y)
    predictions = model.predict(frame)

    expected = (frame["open_to_close_return"] > 0).astype(int).to_numpy()
    np.testing.assert_array_equal(predictions, expected)


def test_last_direction_probability_is_conditional_on_direction(frame, y):
    model = LastDirectionBaseline(feature="open_to_close_return").fit(frame, y)

    direction = (frame["open_to_close_return"] > 0).to_numpy()
    expected_up = y[direction].mean()
    expected_down = y[~direction].mean()

    proba = model.predict_proba(frame)[:, 1]
    np.testing.assert_allclose(proba[direction], expected_up)
    np.testing.assert_allclose(proba[~direction], expected_down)


def test_last_direction_collapses_to_the_prior_when_uninformative():
    """If direction carries no information, both conditional rates equal the prior,
    so ROC-AUC lands at 0.50 — the honest answer, not a degenerate one."""
    index = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"open_to_close_return": rng.normal(0, 0.01, 400)}, index=index)
    y = rng.binomial(1, 0.55, 400)

    model = LastDirectionBaseline(feature="open_to_close_return").fit(frame, y)

    assert model.rate_after_up_ == pytest.approx(model.rate_after_down_, abs=0.12)


def test_last_direction_requires_its_feature(frame, y):
    model = LastDirectionBaseline(feature="not_a_column")
    with pytest.raises(KeyError, match="needs feature"):
        model.fit(frame, y)


def test_last_direction_uses_the_target_appropriate_feature():
    """open_to_close targets key off the intraday move, not the close-to-close one."""
    intraday = build_baselines(target_definition="open_to_close")["last_direction"]
    overnight = build_baselines(target_definition="close_to_close")["last_direction"]

    assert intraday.feature == "open_to_close_return"
    assert overnight.feature == "return_1d"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_build_baselines_returns_all_four():
    baselines = build_baselines()
    assert set(baselines) == {"majority", "always_up", "last_direction", "random"}


def test_random_baseline_can_be_excluded():
    assert "random" not in build_baselines(include_random=False)


def test_every_baseline_emits_consistent_shapes(frame, y):
    for name, model in build_baselines().items():
        fitted = model.fit(frame, y)
        proba = fitted.predict_proba(frame)
        hard = fitted.predict(frame)

        assert proba.shape == (10, 2), name
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
        assert hard.shape == (10,), name
        assert set(np.unique(hard)) <= {0, 1}, name


def test_baseline_probability_and_label_metrics_are_consistent(frame, y):
    """Confusion counts must come from predict(), and log loss from predict_proba()."""
    from stock_movement.evaluation import classification_metrics

    for name, model in build_baselines().items():
        fitted = model.fit(frame, y)
        proba = fitted.predict_proba(frame)[:, 1]
        hard = fitted.predict(frame)

        metrics = classification_metrics(y, proba, y_pred=hard)
        counted = sum(
            metrics[k] for k in ("true_positives", "true_negatives", "false_positives", "false_negatives")
        )

        assert counted == len(y), name
        assert metrics["positive_rate_pred"] == pytest.approx(hard.mean()), name
        assert np.isfinite(metrics["log_loss"]), name


def test_baseline_prefix_is_used_consistently():
    assert BASELINE_PREFIX == "baseline_"
