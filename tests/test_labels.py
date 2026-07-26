"""Label correctness. An off-by-one here invalidates every number downstream."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from stock_movement.config import LabelConfig
from stock_movement.labels import (
    drop_unlabeled_tail,
    future_return_close_to_close,
    future_return_open_to_close,
    label_summary,
    make_labels,
)

CLOSE_TO_CLOSE = LabelConfig(target_definition="close_to_close", horizon=1)
OPEN_TO_CLOSE = LabelConfig(target_definition="open_to_close", horizon=1)


def _frame(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(closes))
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(c, o) * 1.01 for c, o in zip(closes, opens, strict=True)],
            "Low": [min(c, o) * 0.99 for c, o in zip(closes, opens, strict=True)],
            "Close": closes,
            "Volume": [1e6] * len(closes),
        },
        index=index,
    )


# --------------------------------------------------------------------------
# the default (executable) target
# --------------------------------------------------------------------------
def test_open_to_close_target_uses_the_next_session_open_and_close():
    """The default target: Close(t+1) / Open(t+1) - 1.

    Row 0's label describes session 1 entirely: enter at its open, exit at its
    close. Nothing about session 0's own prices enters the label.
    """
    frame = _frame(closes=[100.0, 110.0, 90.0], opens=[99.0, 100.0, 100.0])
    labels = make_labels(frame, OPEN_TO_CLOSE)

    assert labels["future_return"].iloc[0] == pytest.approx(110.0 / 100.0 - 1.0)
    assert labels["target"].iloc[0] == 1
    assert labels["future_return"].iloc[1] == pytest.approx(90.0 / 100.0 - 1.0)
    assert labels["target"].iloc[1] == 0
    assert np.isnan(labels["target"].iloc[2])


def test_open_to_close_ignores_the_overnight_gap():
    """A huge overnight gap that fully reverses intraday must not be labelled up."""
    # Session 1 gaps up to 150 at the open but closes at 120: intraday is a loss.
    frame = _frame(closes=[100.0, 120.0], opens=[99.0, 150.0])
    labels = make_labels(frame, OPEN_TO_CLOSE)

    assert labels["future_return"].iloc[0] == pytest.approx(120.0 / 150.0 - 1.0)
    assert labels["target"].iloc[0] == 0

    # Close-to-close would call the same session a gain — the two targets differ.
    close_labels = make_labels(frame, CLOSE_TO_CLOSE)
    assert close_labels["target"].iloc[0] == 1


def test_open_to_close_helper_matches_the_formula():
    frame = _frame(closes=[100.0, 110.0, 120.0], opens=[99.0, 100.0, 100.0])
    future = future_return_open_to_close(frame["Open"], frame["Close"], horizon=1)
    assert future.iloc[0] == pytest.approx(110.0 / 100.0 - 1.0)


# --------------------------------------------------------------------------
# the research (close-to-close) target
# --------------------------------------------------------------------------
def test_the_worked_example_from_the_specification():
    """Close: 100, 101, 99  ->  target: 1, 0, NaN (close-to-close)."""
    labels = make_labels(_frame([100.0, 101.0, 99.0]), CLOSE_TO_CLOSE)

    assert labels["target"].iloc[0] == 1
    assert labels["target"].iloc[1] == 0
    assert np.isnan(labels["target"].iloc[2])

    assert labels["future_return"].iloc[0] == pytest.approx(0.01)
    assert labels["future_return"].iloc[1] == pytest.approx(99.0 / 101.0 - 1.0)
    assert np.isnan(labels["future_return"].iloc[2])


def test_label_describes_the_next_session_not_the_current_one():
    close = pd.Series([10.0, 20.0, 5.0])
    future = future_return_close_to_close(close, horizon=1)

    assert future.iloc[0] == pytest.approx(1.0)  # 10 -> 20 belongs to row 0
    assert future.iloc[1] == pytest.approx(-0.75)  # 20 -> 5 belongs to row 1


def test_horizon_greater_than_one_looks_further_ahead():
    """Multi-day horizons are a close-to-close concept only."""
    config = LabelConfig(target_definition="close_to_close", horizon=2)
    labels = make_labels(_frame([100.0, 90.0, 130.0, 100.0]), config)

    assert labels["future_return"].iloc[0] == pytest.approx(0.30)
    assert labels["target"].iloc[0] == 1
    assert labels["target"].iloc[2:].isna().all()


# --------------------------------------------------------------------------
# shared behaviour
# --------------------------------------------------------------------------
@pytest.mark.parametrize("config", [CLOSE_TO_CLOSE, OPEN_TO_CLOSE])
def test_flat_session_is_labelled_down(config):
    """`future_return > 0` is strict: an unchanged price is not "up"."""
    labels = make_labels(_frame([100.0, 100.0, 100.0], opens=[100.0, 100.0, 100.0]), config)

    assert labels["target"].iloc[0] == 0
    assert labels["target"].iloc[1] == 0


@pytest.mark.parametrize("config", [CLOSE_TO_CLOSE, OPEN_TO_CLOSE])
def test_unlabeled_tail_is_removed(config):
    labels = make_labels(_frame([100.0, 101.0, 99.0]), config)
    trimmed = drop_unlabeled_tail(labels, horizon=1)

    assert len(trimmed) == 2
    assert not trimmed["target"].isna().any()


def test_dropping_more_rows_than_exist_yields_an_empty_frame():
    labels = make_labels(_frame([100.0, 101.0]), CLOSE_TO_CLOSE)
    assert drop_unlabeled_tail(labels, horizon=5).empty


def test_label_summary_counts_are_consistent():
    labels = make_labels(_frame([100.0, 101.0, 99.0, 105.0]), CLOSE_TO_CLOSE)
    summary = label_summary(labels)

    assert summary["n_up"] + summary["n_down"] == summary["n_labeled"]
    assert 0.0 <= summary["up_rate"] <= 1.0


def test_zero_horizon_is_rejected():
    with pytest.raises(ValidationError):
        LabelConfig(horizon=0)


def test_unknown_target_definition_is_rejected_by_the_schema():
    """The literal type makes an invalid target unrepresentable, not just unhandled."""
    with pytest.raises(ValidationError, match="open_to_close"):
        LabelConfig(target_definition="crystal_ball")  # type: ignore[arg-type]


def test_open_to_close_rejects_a_multi_session_horizon():
    with pytest.raises(ValidationError, match="single-session target"):
        LabelConfig(target_definition="open_to_close", horizon=2)


def test_negative_horizon_helpers_are_rejected():
    close = pd.Series([100.0, 101.0])
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        future_return_close_to_close(close, horizon=0)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        future_return_open_to_close(close, close, horizon=0)
