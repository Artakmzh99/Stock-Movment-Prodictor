"""Assemble the modelling dataset: download -> features -> labels -> clean frame.

The output obeys a contract that is *enforced* here rather than trusted:

* index is a sorted, unique, tz-naive DatetimeIndex of trading dates;
* ``X`` holds only feature columns, in manifest order, with no NaN or inf;
* ``target`` and ``future_return`` are returned separately and never inside ``X``;
* rows are dropped only after every rolling feature *and* the label exist, so
  features and labels cannot drift out of alignment.

Benchmark alignment uses an inner join on shared trading sessions. Forward-filling
would invent a benchmark price for a day it did not trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .config import Config
from .data import get_ohlcv
from .features import build_features, feature_manifest
from .labels import drop_unlabeled_tail, label_summary, make_labels
from .validation import (
    DataValidationError,
    assert_chronological,
    assert_finite,
    assert_no_leakage_columns,
)


@dataclass
class BenchmarkAlignment:
    """Audit trail for the benchmark inner join."""

    benchmark_ticker: str
    n_target_rows_before: int
    n_benchmark_rows_before: int
    n_rows_after: int
    n_removed: int
    removed_fraction: float
    removed_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def align_benchmark(
    close: pd.Series,
    benchmark_close: pd.Series,
    benchmark_ticker: str,
    max_join_loss: float = 0.02,
) -> tuple[pd.Series, pd.Series, BenchmarkAlignment]:
    """Restrict both series to sessions where *both* actually traded.

    Returns the trimmed asset close, the aligned benchmark close, and an audit
    record. Fails when the join would discard more than `max_join_loss` of the
    target's sessions — at that point the two instruments do not share a calendar
    and pairing them is a modelling error, not a data-cleaning detail.
    """
    shared = close.index.intersection(benchmark_close.index)
    removed = close.index.difference(shared)
    removed_fraction = len(removed) / len(close.index) if len(close.index) else 0.0

    alignment = BenchmarkAlignment(
        benchmark_ticker=benchmark_ticker,
        n_target_rows_before=len(close),
        n_benchmark_rows_before=len(benchmark_close),
        n_rows_after=len(shared),
        n_removed=len(removed),
        removed_fraction=float(removed_fraction),
        removed_dates=[str(d.date()) for d in removed[:50]],
    )

    if removed_fraction > max_join_loss:
        raise DataValidationError(
            f"benchmark join with {benchmark_ticker} would remove {len(removed)} of "
            f"{len(close.index)} target sessions ({removed_fraction:.2%}), above the "
            f"{max_join_loss:.2%} limit. The two instruments do not share a trading "
            "calendar closely enough to be paired."
        )

    return close.loc[shared], benchmark_close.loc[shared], alignment


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    future_return: pd.Series
    prices: pd.DataFrame
    manifest: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        idx = self.X.index
        assert isinstance(idx, pd.DatetimeIndex)
        return idx

    def __len__(self) -> int:
        return len(self.X)

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.X),
            "n_features": self.X.shape[1],
            "first_date": str(self.index[0].date()),
            "last_date": str(self.index[-1].date()),
            "up_rate": float(self.y.mean()),
            "target_definition": self.manifest.get("target_definition"),
        }


def build_dataset(
    config: Config,
    force_refresh: bool = False,
    now_utc: datetime | None = None,
) -> Dataset:
    bundle = get_ohlcv(config, force_refresh=force_refresh, now_utc=now_utc)
    prices = bundle.prices
    metadata: dict[str, Any] = {"target": bundle.metadata}

    benchmark_close: pd.Series | None = None
    if config.features.use_benchmark_features:
        assert config.data.benchmark_ticker is not None  # guaranteed by config validation
        bench_bundle = get_ohlcv(
            config,
            ticker=config.data.benchmark_ticker,
            force_refresh=force_refresh,
            now_utc=now_utc,
        )
        metadata["benchmark"] = bench_bundle.metadata

        asset_close, benchmark_close, alignment = align_benchmark(
            prices["Close"].astype(float),
            bench_bundle.prices["Close"].astype(float),
            config.data.benchmark_ticker,
            max_join_loss=config.features.max_benchmark_join_loss,
        )
        prices = prices.loc[asset_close.index]
        metadata["benchmark_alignment"] = alignment.to_dict()

    features = build_features(prices, config.features, benchmark_close=benchmark_close)
    labels = make_labels(prices, config.labels)

    combined = features.join(labels, how="inner")
    combined = drop_unlabeled_tail(combined, config.labels.horizon)

    n_before = len(combined)
    combined = combined.dropna()
    n_dropped = n_before - len(combined)

    if combined.empty:
        raise DataValidationError(
            "no rows survived NaN removal — the date range is too short for the configured rolling windows"
        )

    if len(combined) < config.data.min_rows:
        raise DataValidationError(
            f"only {len(combined)} rows remain after feature warm-up and label "
            f"construction, below data.min_rows={config.data.min_rows}. "
            f"({n_dropped} rows consumed by rolling windows; longest window is "
            f"{config.features.max_window}.) Widen start_date or lower min_rows."
        )

    feature_columns = list(features.columns)
    X = combined[feature_columns]
    y = combined["target"].astype(int)
    future_return = combined["future_return"].astype(float)

    assert_chronological(X.index)
    assert_no_leakage_columns(X)
    assert_finite(X, context="dataset feature matrix")

    manifest = feature_manifest(X, config.features)
    manifest["rows_dropped_for_nan"] = int(n_dropped)
    manifest["target_definition"] = config.labels.target_definition
    manifest["horizon"] = config.labels.horizon
    manifest["labels"] = label_summary(combined[["future_return", "target"]])

    return Dataset(
        X=X,
        y=y,
        future_return=future_return,
        prices=prices.loc[X.index[0] :],
        manifest=manifest,
        metadata=metadata,
    )
