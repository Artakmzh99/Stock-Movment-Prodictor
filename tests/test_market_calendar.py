"""Partial-session handling (P1.4), with the clock mocked.

Unconditionally dropping the last row was wrong in three of four situations. Each
is pinned here at a specific instant so the behaviour cannot silently regress.

Reference facts for the NYSE (XNYS):
* 2026-03-05 is a Thursday session, closing 21:00 UTC (16:00 America/New_York).
* 2026-03-07 is a Saturday; the prior session is Friday 2026-03-06.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stock_movement.market_calendar import (
    decide_partial_bar,
    last_completed_session,
    next_session,
)

THURSDAY = pd.Timestamp("2026-03-05")
FRIDAY = pd.Timestamp("2026-03-06")
SATURDAY = pd.Timestamp("2026-03-07")


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_historical_end_date_keeps_last_row():
    """With an explicit end_date the vendor returned finished sessions."""
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 17), explicit_end_date=date(2026, 3, 5))

    assert decision.drop_last_row is False
    assert "explicit end_date" in decision.reason


def test_during_market_hours_drops_current_partial_session():
    """17:00 UTC == 12:00 New York: the session is open and the bar is forming."""
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 17))

    assert decision.drop_last_row is True
    assert "still forming" in decision.reason
    assert decision.session_close_utc is not None


def test_after_close_keeps_current_session():
    """22:00 UTC == 17:00 New York, an hour after the bell."""
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 22))

    assert decision.drop_last_row is False
    assert "complete" in decision.reason


def test_exactly_at_the_close_keeps_the_session():
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 21))
    assert decision.drop_last_row is False


def test_one_minute_before_the_close_drops_the_session():
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 20, 59))
    assert decision.drop_last_row is True


def test_weekend_keeps_last_completed_session():
    """Running on Saturday, Friday's bar is finished."""
    decision = decide_partial_bar(FRIDAY, now_utc=_utc(2026, 3, 7, 12))

    assert decision.drop_last_row is False
    assert "complete" in decision.reason


def test_non_session_row_is_kept_and_recorded():
    decision = decide_partial_bar(SATURDAY, now_utc=_utc(2026, 3, 9, 12))

    assert decision.drop_last_row is False
    assert "not a trading session" in decision.reason


def test_disabled_handling_never_drops():
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 17), enabled=False)

    assert decision.drop_last_row is False
    assert "disabled" in decision.reason


def test_partial_bar_decision_records_full_context():
    decision = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 17))
    payload = decision.to_dict()

    for key in (
        "last_row_date",
        "drop_last_row",
        "reason",
        "exchange",
        "exchange_timezone",
        "session_close_utc",
        "evaluated_at_utc",
    ):
        assert key in payload

    assert payload["last_row_date"] == "2026-03-05"
    assert payload["exchange"] == "XNYS"
    assert payload["exchange_timezone"] == "America/New_York"


def test_naive_now_is_treated_as_utc():
    aware = decide_partial_bar(THURSDAY, now_utc=_utc(2026, 3, 5, 17))
    naive = decide_partial_bar(THURSDAY, now_utc=datetime(2026, 3, 5, 17))
    assert aware.drop_last_row == naive.drop_last_row


def test_unknown_exchange_is_rejected():
    with pytest.raises(ValueError, match="unknown exchange calendar"):
        decide_partial_bar(THURSDAY, exchange="NOT_AN_EXCHANGE", now_utc=_utc(2026, 3, 5, 22))


def test_last_completed_session_on_a_weekend_is_friday():
    assert last_completed_session(now_utc=_utc(2026, 3, 7, 12)) == FRIDAY


def test_last_completed_session_during_a_session_is_the_previous_day():
    """Mid-session on Thursday, the last *completed* session is Wednesday."""
    result = last_completed_session(now_utc=_utc(2026, 3, 5, 17))
    assert result == pd.Timestamp("2026-03-04")


def test_last_completed_session_after_close_is_today():
    assert last_completed_session(now_utc=_utc(2026, 3, 5, 22)) == THURSDAY


def test_next_session_skips_the_weekend():
    assert next_session(FRIDAY) == pd.Timestamp("2026-03-09")


def test_next_session_of_a_thursday_is_friday():
    assert next_session(THURSDAY) == FRIDAY
