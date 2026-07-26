"""Exchange-calendar-aware handling of the final (possibly incomplete) daily bar.

The previous behaviour was to unconditionally drop the last row. That is wrong
often enough to matter:

* with an explicit historical ``end_date``, the last row is a finished session and
  dropping it silently discards real data;
* on a weekend, the last row is Friday — already complete;
* after the closing bell, today's bar is complete;
* only *during* a session is the last bar genuinely partial.

So the decision depends on the wall clock relative to the exchange's session
close, which is what a maintained calendar is for (holidays, half-days, and the
1:00 pm closes around Thanksgiving and Christmas Eve).

Every decision is recorded in metadata with its reason, so a reader can tell
whether a row was dropped and why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

DEFAULT_EXCHANGE = "XNYS"


@dataclass(frozen=True)
class PartialBarDecision:
    """Why the final bar was kept or dropped."""

    last_row_date: str
    drop_last_row: bool
    reason: str
    exchange: str
    exchange_timezone: str
    session_close_utc: str | None
    evaluated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_calendar(exchange: str) -> Any:
    import exchange_calendars as xcals

    try:
        return xcals.get_calendar(exchange)
    except Exception as exc:  # pragma: no cover - depends on the library's registry
        raise ValueError(
            f"unknown exchange calendar {exchange!r}; see exchange_calendars.get_calendar_names()"
        ) from exc


def decide_partial_bar(
    last_row_date: pd.Timestamp | date,
    exchange: str = DEFAULT_EXCHANGE,
    now_utc: datetime | None = None,
    explicit_end_date: date | None = None,
    enabled: bool = True,
) -> PartialBarDecision:
    """Decide whether the final daily bar is an incomplete session.

    `now_utc` is injectable so the behaviour is testable without waiting for the
    market to open.
    """
    now_utc = now_utc or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)

    session = pd.Timestamp(last_row_date).normalize()
    calendar = _get_calendar(exchange)
    exchange_tz = str(calendar.tz)

    def decision(drop: bool, reason: str, close: pd.Timestamp | None = None) -> PartialBarDecision:
        return PartialBarDecision(
            last_row_date=str(session.date()),
            drop_last_row=drop,
            reason=reason,
            exchange=exchange,
            exchange_timezone=exchange_tz,
            session_close_utc=close.isoformat() if close is not None else None,
            evaluated_at_utc=now_utc.isoformat(),
        )

    if not enabled:
        return decision(False, "partial-bar handling disabled by config (drop_last_incomplete=false)")

    # An explicit historical end date means the vendor returned finished sessions.
    if explicit_end_date is not None:
        return decision(
            False,
            f"explicit end_date {explicit_end_date} requested; final row is a completed session",
        )

    if not calendar.is_session(session):
        # Not a trading session at all (bad vendor row, or a tz artefact). Keep it
        # and record the oddity rather than silently discarding data.
        return decision(False, f"{session.date()} is not a trading session on {exchange}; row kept")

    session_close = calendar.session_close(session)
    if now_utc < session_close.to_pydatetime():
        return decision(
            True,
            f"session {session.date()} closes at {session_close.isoformat()} "
            f"which is after now ({now_utc.isoformat()}); bar is still forming",
            session_close,
        )

    return decision(
        False,
        f"session {session.date()} closed at {session_close.isoformat()}, "
        f"at or before now ({now_utc.isoformat()}); bar is complete",
        session_close,
    )


def last_completed_session(exchange: str = DEFAULT_EXCHANGE, now_utc: datetime | None = None) -> pd.Timestamp:
    """The most recent session whose close is at or before `now_utc`."""
    now_utc = now_utc or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)

    calendar = _get_calendar(exchange)
    candidate = pd.Timestamp(now_utc).tz_convert("UTC").normalize().tz_localize(None)

    for _ in range(30):  # a month of slack covers any holiday cluster
        if calendar.is_session(candidate) and now_utc >= calendar.session_close(candidate).to_pydatetime():
            return candidate
        candidate -= pd.Timedelta(days=1)

    raise RuntimeError(f"no completed {exchange} session found in the 30 days before {now_utc}")


def next_session(session: pd.Timestamp | date, exchange: str = DEFAULT_EXCHANGE) -> pd.Timestamp | None:
    """The trading session following `session`, used to date an expected execution."""
    calendar = _get_calendar(session_exchange := exchange)
    current = pd.Timestamp(session).normalize()
    try:
        return pd.Timestamp(calendar.next_session(current)).normalize()
    except Exception:
        # Beyond the calendar's known bounds; the caller reports null rather than guessing.
        del session_exchange
        return None
