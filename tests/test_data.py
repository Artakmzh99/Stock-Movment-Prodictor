"""Data schema, normalisation, duplicate policy and hard OHLC validation (P1.7)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stock_movement.data import get_ohlcv, normalize_ohlcv
from stock_movement.provenance import ChecksumError, sha256_file
from stock_movement.validation import (
    DataValidationError,
    assert_chronological,
    assert_finite,
    resolve_duplicate_index,
    validate_ohlcv,
)
from tests.conftest import make_ohlcv

AFTER_CLOSE = datetime(2026, 3, 5, 22, 0, tzinfo=UTC)


def test_valid_frame_passes_and_reports(ohlcv):
    report = validate_ohlcv(ohlcv, min_rows=100)

    assert report.n_rows == len(ohlcv)
    assert report.first_date == str(ohlcv.index[0].date())
    assert all(pct == 0.0 for pct in report.missing_pct.values())


# --------------------------------------------------------------------------
# P1.7 impossible OHLC bars must hard-fail
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("column", "factor", "expected"),
    [
        ("High", 0.5, "High < Low"),  # High pushed below Low
        ("Low", 1.5, "Low > Open"),  # Low pushed above the rest
    ],
)
def test_impossible_bar_geometry_fails(ohlcv, column, factor, expected):
    broken = ohlcv.copy()
    broken.iloc[50, broken.columns.get_loc(column)] *= factor

    with pytest.raises(DataValidationError, match="impossible OHLC bars"):
        validate_ohlcv(broken, min_rows=100)


def test_high_below_open_fails(ohlcv):
    broken = ohlcv.copy()
    row = 60
    broken.iloc[row, broken.columns.get_loc("Open")] = broken["High"].iloc[row] * 1.10
    with pytest.raises(DataValidationError, match="High < Open"):
        validate_ohlcv(broken, min_rows=100)


def test_high_below_close_fails(ohlcv):
    broken = ohlcv.copy()
    row = 61
    broken.iloc[row, broken.columns.get_loc("Close")] = broken["High"].iloc[row] * 1.10
    with pytest.raises(DataValidationError, match="High < Close"):
        validate_ohlcv(broken, min_rows=100)


def test_low_above_open_fails(ohlcv):
    broken = ohlcv.copy()
    row = 62
    broken.iloc[row, broken.columns.get_loc("Open")] = broken["Low"].iloc[row] * 0.90
    with pytest.raises(DataValidationError, match="Low > Open"):
        validate_ohlcv(broken, min_rows=100)


def test_low_above_close_fails(ohlcv):
    broken = ohlcv.copy()
    row = 63
    broken.iloc[row, broken.columns.get_loc("Close")] = broken["Low"].iloc[row] * 0.90
    with pytest.raises(DataValidationError, match="Low > Close"):
        validate_ohlcv(broken, min_rows=100)


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close"])
def test_non_positive_price_is_rejected(ohlcv, column):
    broken = ohlcv.copy()
    broken.iloc[5, broken.columns.get_loc(column)] = 0.0
    with pytest.raises(DataValidationError, match="non-positive prices"):
        validate_ohlcv(broken, min_rows=100)


def test_negative_volume_is_rejected(ohlcv):
    broken = ohlcv.copy()
    broken.iloc[3, broken.columns.get_loc("Volume")] = -1.0
    with pytest.raises(DataValidationError, match="Volume contains negative"):
        validate_ohlcv(broken, min_rows=100)


def test_missing_price_leg_fails_for_an_executable_target(ohlcv):
    """open_to_close needs a complete bar; a missing Open makes the label undefined."""
    broken = ohlcv.copy()
    broken.iloc[7, broken.columns.get_loc("Open")] = None

    with pytest.raises(DataValidationError, match="executable"):
        validate_ohlcv(broken, min_rows=100, require_full_ohlc=True)

    # Tolerated when the target does not need the Open.
    report = validate_ohlcv(broken, min_rows=100, require_full_ohlc=False)
    assert report.missing_pct["Open"] > 0


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_duplicate_dates_are_rejected_by_validation(ohlcv):
    duplicated = pd.concat([ohlcv, ohlcv.iloc[[10]]]).sort_index()
    with pytest.raises(DataValidationError, match="duplicate dates"):
        validate_ohlcv(duplicated, min_rows=100)


def test_unsorted_index_is_rejected(ohlcv):
    with pytest.raises(DataValidationError, match="ascending date order"):
        validate_ohlcv(ohlcv.iloc[::-1], min_rows=100)


def test_missing_column_is_rejected(ohlcv):
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_ohlcv(ohlcv.drop(columns=["Volume"]), min_rows=100)


def test_non_datetime_index_is_rejected(ohlcv):
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        validate_ohlcv(ohlcv.reset_index(drop=True), min_rows=100)


def test_too_few_rows_is_rejected(ohlcv):
    with pytest.raises(DataValidationError, match="need at least"):
        validate_ohlcv(ohlcv.head(20), min_rows=100)


def test_large_moves_are_recorded_not_removed(ohlcv):
    """Crashes and split artefacts are real. Flag them; never silently delete."""
    spiked = ohlcv.copy()
    row = 50
    for column in ("Open", "High", "Low", "Close"):
        spiked.iloc[row, spiked.columns.get_loc(column)] *= 2.0

    report = validate_ohlcv(spiked, min_rows=100)

    assert report.n_large_moves >= 1
    assert any("|return|" in w for w in report.warnings)
    assert report.n_rows == len(spiked), "validation must not drop rows"


def test_zero_volume_days_are_recorded(ohlcv):
    frame = ohlcv.copy()
    frame.iloc[10, frame.columns.get_loc("Volume")] = 0.0
    report = validate_ohlcv(frame, min_rows=100)

    assert report.n_zero_volume_days == 1
    assert any("zero volume" in w for w in report.warnings)


# --------------------------------------------------------------------------
# P1.7 duplicate policy
# --------------------------------------------------------------------------
def test_identical_duplicates_are_deduplicated_and_counted(ohlcv):
    duplicated = pd.concat([ohlcv, ohlcv.iloc[[10]]]).sort_index()

    resolved, removed = resolve_duplicate_index(duplicated, deduplicate_identical=True)

    assert removed == 1
    assert not resolved.index.has_duplicates
    assert len(resolved) == len(ohlcv)


def test_conflicting_duplicates_fail(ohlcv):
    """Two different prices for one session cannot be reconciled automatically."""
    conflicting = ohlcv.iloc[[10]].copy()
    conflicting.iloc[0, conflicting.columns.get_loc("Close")] *= 1.05
    frame = pd.concat([ohlcv, conflicting]).sort_index()

    with pytest.raises(DataValidationError, match="conflicting duplicate rows"):
        resolve_duplicate_index(frame)


def test_duplicates_rejected_when_deduplication_is_disabled(ohlcv):
    duplicated = pd.concat([ohlcv, ohlcv.iloc[[10]]]).sort_index()
    with pytest.raises(DataValidationError, match="deduplicate_identical_rows is false"):
        resolve_duplicate_index(duplicated, deduplicate_identical=False)


def test_duplicate_policy_is_recorded(ohlcv):
    report = validate_ohlcv(ohlcv, min_rows=100, n_identical_duplicates_removed=3)

    assert report.n_identical_duplicates_removed == 3
    assert "conflicting rows fail" in report.duplicate_policy
    assert any("identical duplicate" in w for w in report.warnings)


def test_frames_without_duplicates_pass_through_untouched(ohlcv):
    resolved, removed = resolve_duplicate_index(ohlcv)
    assert removed == 0
    assert resolved is ohlcv


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def test_normalize_flattens_a_multiindex_and_sorts():
    frame = make_ohlcv(n=30)
    multi = frame.copy()
    multi.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])

    result = normalize_ohlcv(multi.iloc[::-1], "AAPL")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.is_monotonic_increasing
    assert not isinstance(result.columns, pd.MultiIndex)


def test_normalize_handles_ticker_first_multiindex():
    frame = make_ohlcv(n=30)
    multi = frame.copy()
    multi.columns = pd.MultiIndex.from_product([["AAPL"], frame.columns])

    result = normalize_ohlcv(multi, "AAPL")
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalize_strips_timezone_and_deduplicates():
    frame = make_ohlcv(n=30)
    tz_aware = frame.copy()
    tz_aware.index = tz_aware.index.tz_localize("America/New_York")

    result = normalize_ohlcv(pd.concat([tz_aware, tz_aware.iloc[[5]]]), "AAPL")

    assert result.index.tz is None
    assert not result.index.has_duplicates
    assert result.index.is_monotonic_increasing


def test_normalize_raises_when_a_price_column_is_absent():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_ohlcv(make_ohlcv(n=30).drop(columns=["High"]), "AAPL")


# --------------------------------------------------------------------------
# P1.3 cache integrity
# --------------------------------------------------------------------------
def test_cache_records_a_digest_and_is_reused(offline_config):
    first = get_ohlcv(offline_config, now_utc=AFTER_CLOSE)
    assert first.metadata["cache_hit"] is False
    assert len(first.metadata["raw_sha256"]) == 64

    second = get_ohlcv(offline_config, now_utc=AFTER_CLOSE)
    assert second.metadata["cache_hit"] is True
    assert second.metadata["raw_sha256"] == first.metadata["raw_sha256"]
    pd.testing.assert_frame_equal(first.prices, second.prices)


def test_modified_raw_cache_fails_checksum(offline_config):
    """Hand-editing a cached Parquet must surface as an error, not a new result."""
    from stock_movement.data import cache_paths

    get_ohlcv(offline_config, now_utc=AFTER_CLOSE)
    parquet_path, _ = cache_paths(offline_config)

    original = sha256_file(parquet_path)
    parquet_path.write_bytes(parquet_path.read_bytes() + b"corruption")
    assert sha256_file(parquet_path) != original

    with pytest.raises(ChecksumError, match="checksum mismatch"):
        get_ohlcv(offline_config, now_utc=AFTER_CLOSE)


def test_force_refresh_bypasses_the_cache(offline_config):
    get_ohlcv(offline_config, now_utc=AFTER_CLOSE)
    refreshed = get_ohlcv(offline_config, force_refresh=True, now_utc=AFTER_CLOSE)
    assert refreshed.metadata["cache_hit"] is False


def test_metadata_records_the_partial_bar_decision(offline_config):
    bundle = get_ohlcv(offline_config, now_utc=AFTER_CLOSE)
    decision = bundle.metadata["partial_bar_decision"]

    assert "drop_last_row" in decision
    assert "reason" in decision
    # This config has an explicit end_date, so the final row is complete and kept.
    assert decision["drop_last_row"] is False
    assert "explicit end_date" in decision["reason"]


def test_historical_end_date_keeps_the_last_row(offline_config, monkeypatch):
    """Regression for P1.4: the last row must survive an explicit end_date."""
    frame = make_ohlcv(n=1400, seed=11)
    monkeypatch.setattr(
        "stock_movement.data.download_ohlcv",
        lambda ticker, start, end, interval="1d", auto_adjust=True: frame.copy(),
    )
    assert offline_config.data.end_date == date(2026, 1, 2)

    bundle = get_ohlcv(offline_config, force_refresh=True, now_utc=AFTER_CLOSE)
    assert bundle.prices.index[-1] == frame.index[-1]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_assert_chronological_catches_disorder(ohlcv):
    assert_chronological(ohlcv.index)

    with pytest.raises(DataValidationError, match="ascending chronological"):
        assert_chronological(ohlcv.index[::-1])

    with pytest.raises(DataValidationError, match="duplicate"):
        assert_chronological(ohlcv.index.append(ohlcv.index[[0]]).sort_values())

    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        assert_chronological(pd.Index([1, 2, 3]))


def test_assert_finite_rejects_nan_and_inf():
    import numpy as np

    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert_finite(frame)

    with pytest.raises(DataValidationError, match="NaN in columns"):
        assert_finite(frame.assign(a=[1.0, np.nan]))

    with pytest.raises(DataValidationError, match="infinite values"):
        assert_finite(frame.assign(b=[1.0, np.inf]))


# --------------------------------------------------------------------------
# OHLC tolerance: float noise vs real violations
# --------------------------------------------------------------------------
def test_one_ulp_ohlc_violation_is_tolerated(ohlcv):
    """Regression test for a real SPY case.

    Split/dividend adjustment multiplies every price leg by a factor. When a
    session closes exactly at its high, the adjusted High and Close can differ by
    one unit in the last place — observed on SPY 2013-03-04 as `High < Close` by
    1.17e-16 relative. That is float64 epsilon, not bad data, and rejecting it
    would discard a sound series.
    """
    import numpy as np

    frame = ohlcv.copy()
    row = 100
    close = float(frame["Close"].iloc[row])

    # A flat bar that closed exactly at its high, then nudge High down one ULP.
    # Open and Low are pinned too, so only the High-vs-Close relation is tested.
    for column in ("Open", "Low", "Close"):
        frame.iloc[row, frame.columns.get_loc(column)] = close
    frame.iloc[row, frame.columns.get_loc("High")] = np.nextafter(close, 0.0)

    report = validate_ohlcv(frame, min_rows=100)  # must not raise
    assert report.n_rows == len(frame)


def test_economically_meaningful_ohlc_violation_still_fails(ohlcv):
    """The tolerance must not swallow a real broken bar.

    One basis point is four orders of magnitude above the tolerance and is the
    scale at which a genuinely corrupt bar shows up.
    """
    frame = ohlcv.copy()
    row = 100
    close = float(frame["Close"].iloc[row])
    for column in ("Open", "Low", "Close"):
        frame.iloc[row, frame.columns.get_loc(column)] = close
    frame.iloc[row, frame.columns.get_loc("High")] = close * (1 - 1e-4)

    with pytest.raises(DataValidationError, match="High < Close"):
        validate_ohlcv(frame, min_rows=100)


def test_tolerance_scales_with_price_level(ohlcv):
    """A fixed absolute tolerance would be too loose on cheap stocks and too
    tight on expensive ones, so the slack is relative to the close."""
    from stock_movement.validation import OHLC_RELATIVE_TOLERANCE

    frame = ohlcv.copy()
    row = 100
    close = float(frame["Close"].iloc[row])
    for column in ("Open", "Low", "Close"):
        frame.iloc[row, frame.columns.get_loc(column)] = close

    # Just inside the tolerance band.
    frame.iloc[row, frame.columns.get_loc("High")] = close * (1 - OHLC_RELATIVE_TOLERANCE / 10)
    validate_ohlcv(frame, min_rows=100)

    # Comfortably outside it.
    frame.iloc[row, frame.columns.get_loc("High")] = close * (1 - OHLC_RELATIVE_TOLERANCE * 100)
    with pytest.raises(DataValidationError, match="High < Close"):
        validate_ohlcv(frame, min_rows=100)
