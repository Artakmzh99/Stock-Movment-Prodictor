"""Data validation.

Two categories, deliberately separated:

**Hard failures** — anything that makes the data *impossible* rather than merely
surprising: an unsorted or duplicated index, missing columns, non-positive prices,
negative volume, or an OHLC bar that violates its own definition (a High below the
Open cannot happen; the open price is by definition a traded price within the
day's range). These indicate a broken feed or a broken assumption, and silently
proceeding would corrupt every downstream number.

**Recorded observations** — anomalies that are real market events: a 30% crash, a
zero-volume day, a long calendar gap. These are reported and kept. Automatically
"cleaning" outliers out of market data removes exactly the days a risk model needs
to see.

Duplicate policy is explicit: identical duplicate rows may be collapsed and the
action recorded; *conflicting* duplicates for the same date always fail, because
there is no defensible way to choose between two different prices for one day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

#: Relative slack allowed on OHLC ordering checks. Absorbs float64 rounding in
#: adjusted prices (~1e-16) while still catching real violations (~1e-4 or worse).
OHLC_RELATIVE_TOLERANCE = 1e-8
EPSILON = 1e-12


class DataValidationError(ValueError):
    """The data violates a contract the rest of the pipeline assumes."""


@dataclass
class ValidationReport:
    n_rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    missing_pct: dict[str, float] = field(default_factory=dict)
    n_large_moves: int = 0
    large_move_dates: list[str] = field(default_factory=list)
    n_zero_volume_days: int = 0
    max_calendar_gap_days: int = 0
    n_identical_duplicates_removed: int = 0
    duplicate_policy: str = "identical rows deduplicated; conflicting rows fail"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_duplicate_index(df: pd.DataFrame, deduplicate_identical: bool = True) -> tuple[pd.DataFrame, int]:
    """Collapse identical duplicate rows; fail on conflicting ones.

    Returns the frame and the number of rows removed.
    """
    if not df.index.has_duplicates:
        return df, 0

    duplicated_dates = df.index[df.index.duplicated(keep=False)].unique()
    conflicting: list[str] = []
    for stamp in duplicated_dates:
        block = df.loc[[stamp]]
        # nunique per column across the duplicated rows: >1 anywhere means the rows
        # genuinely disagree. This is a frame-wide comparison, not the
        # "is this one series constant" check PD101 targets.
        if (block.nunique(dropna=False) > 1).any():  # noqa: PD101
            conflicting.append(str(pd.Timestamp(stamp).date()))

    if conflicting:
        raise DataValidationError(
            f"conflicting duplicate rows for {len(conflicting)} date(s): {conflicting[:5]}. "
            "Two different prices for one session cannot be reconciled automatically — "
            "re-download with --force-refresh or inspect the source data."
        )

    if not deduplicate_identical:
        raise DataValidationError(
            f"{len(duplicated_dates)} duplicated date(s) present and data.deduplicate_identical_rows is false"
        )

    before = len(df)
    deduplicated = df[~df.index.duplicated(keep="first")]
    return deduplicated, before - len(deduplicated)


def validate_ohlcv(
    df: pd.DataFrame,
    min_rows: int = 1000,
    large_move_threshold: float = 0.25,
    require_full_ohlc: bool = True,
    n_identical_duplicates_removed: int = 0,
) -> ValidationReport:
    """Validate an OHLCV frame. Raises on contract violations, returns a report."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(f"index must be a DatetimeIndex, got {type(df.index).__name__}")

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:5]
        raise DataValidationError(f"index contains duplicate dates, e.g. {list(dupes)}")

    if not df.index.is_monotonic_increasing:
        raise DataValidationError("index must be sorted in ascending date order")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"missing required columns: {sorted(missing_cols)}")

    if len(df) < min_rows:
        raise DataValidationError(
            f"only {len(df)} rows available, need at least {min_rows}. "
            "Widen the date range or lower data.min_rows."
        )

    # -- prices must be positive ------------------------------------------
    for col in PRICE_COLUMNS:
        values = df[col]
        if (values.dropna() <= 0).any():
            bad = df.index[values <= 0][:5]
            raise DataValidationError(
                f"column {col} has non-positive prices at {[str(d.date()) for d in bad]}"
            )

    if (df["Volume"].dropna() < 0).any():
        bad = df.index[df["Volume"] < 0][:5]
        raise DataValidationError(f"Volume contains negative values at {[str(d.date()) for d in bad]}")

    # -- an executable target needs every price leg -----------------------
    if require_full_ohlc:
        for col in PRICE_COLUMNS:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                raise DataValidationError(
                    f"column {col} has {n_missing} missing value(s). An executable "
                    "open-to-close target requires a complete OHLC bar for every row."
                )

    # -- OHLC internal consistency: hard failures -------------------------
    # A High below any other price in the same bar is definitionally impossible.
    #
    # The comparison carries a relative tolerance because split/dividend
    # adjustment multiplies every leg by a factor, and when a session closes
    # exactly at its high the two adjusted values can differ by one unit in the
    # last place. Observed on SPY: `High < Close` by 1.17e-16 relative — pure
    # float64 epsilon (~2.2e-16), not bad data. Rejecting that would throw away a
    # sound series.
    #
    # OHLC_RELATIVE_TOLERANCE sits eight orders of magnitude above float noise and
    # four below anything economically meaningful: a genuinely broken bar is off
    # by basis points (~1e-4) or more, so real violations still fail.
    scale = df["Close"].abs().clip(lower=EPSILON)
    tolerance = scale * OHLC_RELATIVE_TOLERANCE

    checks: tuple[tuple[str, pd.Series], ...] = (
        ("High < Low", df["Low"] - df["High"] > tolerance),
        ("High < Open", df["Open"] - df["High"] > tolerance),
        ("High < Close", df["Close"] - df["High"] > tolerance),
        ("Low > Open", df["Low"] - df["Open"] > tolerance),
        ("Low > Close", df["Low"] - df["Close"] > tolerance),
    )
    violations: list[str] = []
    for label, mask in checks:
        mask = mask.fillna(False)
        if mask.any():
            dates = [str(d.date()) for d in df.index[mask][:3]]
            violations.append(f"{label} on {int(mask.sum())} row(s), e.g. {dates}")
    if violations:
        raise DataValidationError(
            "impossible OHLC bars detected (a bar cannot trade outside its own range): "
            + "; ".join(violations)
        )

    report = ValidationReport(
        n_rows=len(df),
        first_date=str(df.index[0].date()),
        last_date=str(df.index[-1].date()),
        missing_pct={c: float(df[c].isna().mean() * 100) for c in REQUIRED_COLUMNS},
        n_identical_duplicates_removed=int(n_identical_duplicates_removed),
    )
    if n_identical_duplicates_removed:
        report.warnings.append(f"{n_identical_duplicates_removed} identical duplicate row(s) collapsed")

    # -- recorded observations --------------------------------------------
    returns = df["Close"].pct_change()
    large = returns[returns.abs() > large_move_threshold]
    report.n_large_moves = len(large)
    report.large_move_dates = [str(d.date()) for d in large.index[:20]]
    if report.n_large_moves:
        report.warnings.append(
            f"{report.n_large_moves} day(s) with |return| > {large_move_threshold:.0%} "
            "— recorded, not removed (splits and crashes are real events)"
        )

    report.n_zero_volume_days = int((df["Volume"] == 0).sum())
    if report.n_zero_volume_days:
        report.warnings.append(f"{report.n_zero_volume_days} day(s) with zero volume")

    if len(df) > 1:
        gaps = np.diff(df.index.values).astype("timedelta64[D]").astype(int)
        report.max_calendar_gap_days = int(gaps.max())
        if report.max_calendar_gap_days > 10:
            report.warnings.append(
                f"largest calendar gap between bars is {report.max_calendar_gap_days} days"
            )

    for col, pct in report.missing_pct.items():
        if pct > 0:
            report.warnings.append(f"{col} is {pct:.2f}% missing")

    return report


def assert_no_leakage_columns(X: pd.DataFrame) -> None:
    """Guard against label columns reaching the feature matrix."""
    forbidden_exact = {"target", "future_return", "future_close", "future_open"}
    hits = [c for c in X.columns if c in forbidden_exact or c.startswith("future_")]
    if hits:
        raise DataValidationError(f"label/future columns present in X: {sorted(hits)}")


def assert_chronological(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError("expected a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise DataValidationError("index is not in ascending chronological order")
    if index.has_duplicates:
        raise DataValidationError("index contains duplicate timestamps")


def assert_finite(X: pd.DataFrame, context: str = "feature matrix") -> None:
    """Inference must refuse to guess: no NaN, no inf."""
    numeric = X.select_dtypes(include=[np.number])
    if numeric.isna().to_numpy().any():
        columns = sorted(numeric.columns[numeric.isna().any()].tolist())
        raise DataValidationError(f"{context} contains NaN in columns: {columns}")
    if np.isinf(numeric.to_numpy()).any():
        infinite = np.isinf(numeric.to_numpy()).any(axis=0)
        columns = sorted(numeric.columns[infinite].tolist())
        raise DataValidationError(f"{context} contains infinite values in columns: {columns}")
