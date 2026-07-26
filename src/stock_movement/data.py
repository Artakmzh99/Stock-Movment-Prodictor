"""Data ingestion: download OHLCV, normalise it, cache it with a verified digest.

The cache key covers every parameter that can change the returned bytes, so a
cache hit is the same dataset a fresh download would produce. The cached file's
SHA-256 is recorded on write and verified on read: a Parquet file edited by hand
(or corrupted) fails loudly instead of quietly changing a result.

The final bar is dropped only when the exchange calendar says the session is still
open — see ``market_calendar``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .market_calendar import PartialBarDecision, decide_partial_bar
from .provenance import sha256_canonical_json, sha256_file, utc_now_iso, verify_file
from .validation import (
    REQUIRED_COLUMNS,
    ValidationReport,
    resolve_duplicate_index,
    validate_ohlcv,
)


@dataclass
class OhlcvBundle:
    """Validated prices plus everything needed to audit where they came from."""

    prices: pd.DataFrame
    report: ValidationReport
    partial_bar: PartialBarDecision
    metadata: dict[str, Any]


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance may return a MultiIndex keyed (field, ticker) or (ticker, field)."""
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c) for c in df.columns]
        return df

    ticker_level = None
    for i, level in enumerate(df.columns.levels):
        if ticker in set(level):
            ticker_level = i
            break

    if ticker_level is None:
        for i in range(df.columns.nlevels):
            # Locating the constant level of a MultiIndex, not testing a data series.
            if df.columns.get_level_values(i).nunique() == 1:  # noqa: PD101
                df.columns = df.columns.droplevel(i)
                break
    else:
        df = pd.DataFrame(df.xs(ticker, axis=1, level=ticker_level))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(p) for p in c if p) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def normalize_ohlcv(df: pd.DataFrame, ticker: str, deduplicate_identical: bool = True) -> pd.DataFrame:
    """Flatten columns and coerce the index to a sorted unique tz-naive DatetimeIndex."""
    df = _flatten_columns(df.copy(), ticker)

    df.index = pd.to_datetime(df.index, utc=False)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    df = df.sort_index()

    rename = {c: c.title() for c in df.columns if c.title() in [*REQUIRED_COLUMNS, "Adj Close"]}
    df = df.rename(columns=rename)

    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if missing := set(REQUIRED_COLUMNS) - set(keep):
        raise ValueError(f"downloaded data for {ticker} is missing columns: {sorted(missing)}")

    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df, _ = resolve_duplicate_index(df, deduplicate_identical=deduplicate_identical)
    return df.dropna(subset=["Close"])


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch raw OHLCV from Yahoo Finance. `auto_adjust` is always explicit."""
    import yfinance as yf

    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        actions=False,
        threads=False,
        group_by="column",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker} between {start} and {end}. "
            "Check the ticker symbol and network connectivity."
        )
    return pd.DataFrame(raw)


def _cache_key(config: Config, ticker: str) -> str:
    dc = config.data
    payload = {
        "ticker": ticker,
        "start": str(dc.start_date),
        "end": str(dc.end_date) if dc.end_date else None,
        "interval": dc.interval,
        "auto_adjust": dc.auto_adjust,
    }
    return f"{ticker}_{dc.interval}_{sha256_canonical_json(payload)[:12]}"


def cache_paths(config: Config, ticker: str | None = None) -> tuple[Path, Path]:
    ticker = ticker or config.data.ticker
    key = _cache_key(config, ticker)
    cache_dir = config.path("data_raw")
    return cache_dir / f"{key}.parquet", cache_dir / f"{key}.meta.json"


def get_ohlcv(
    config: Config,
    ticker: str | None = None,
    force_refresh: bool = False,
    now_utc: datetime | None = None,
) -> OhlcvBundle:
    """Return validated OHLCV with provenance, using the verified Parquet cache."""
    dc = config.data
    ticker = ticker or dc.ticker
    parquet_path, meta_path = cache_paths(config, ticker)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    cache_hit = parquet_path.exists() and meta_path.exists() and not force_refresh

    if cache_hit:
        metadata: dict[str, Any] = json.loads(meta_path.read_text())
        recorded = metadata.get("raw_sha256")
        if recorded:
            verify_file(parquet_path, recorded)
        normalized = pd.DataFrame(pd.read_parquet(parquet_path))
        metadata["cache_hit"] = True
    else:
        raw = download_ohlcv(
            ticker=ticker,
            start=str(dc.start_date),
            end=str(dc.end_date) if dc.end_date else None,
            interval=dc.interval,
            auto_adjust=dc.auto_adjust,
        )
        normalized = normalize_ohlcv(raw, ticker, deduplicate_identical=dc.deduplicate_identical_rows)
        normalized.to_parquet(parquet_path)
        metadata = {
            "ticker": ticker,
            "start_date": str(dc.start_date),
            "end_date": str(dc.end_date) if dc.end_date else None,
            "interval": dc.interval,
            "auto_adjust": dc.auto_adjust,
            "downloaded_at_utc": utc_now_iso(),
            "n_rows_downloaded": len(normalized),
            "raw_sha256": sha256_file(parquet_path),
            "raw_parquet_path": str(parquet_path.relative_to(config.path("data_raw").parent.parent)),
            "cache_hit": False,
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

    # -- duplicates (cache may predate the current policy) ----------------
    prices, n_deduplicated = resolve_duplicate_index(
        normalized, deduplicate_identical=dc.deduplicate_identical_rows
    )

    # -- partial final bar -------------------------------------------------
    decision = decide_partial_bar(
        last_row_date=prices.index[-1],
        exchange=dc.exchange,
        now_utc=now_utc,
        explicit_end_date=dc.end_date,
        enabled=dc.drop_last_incomplete,
    )
    if decision.drop_last_row and len(prices) > 1:
        prices = prices.iloc[:-1]

    # `min_rows` is the requirement *after* feature warm-up and labelling, so the
    # raw series must be that much longer again. Checking the same number at both
    # stages would either pass too early here or fail twice with the same message.
    warm_up = config.features.max_window + config.labels.horizon + 1
    report = validate_ohlcv(
        prices,
        min_rows=dc.min_rows + warm_up,
        require_full_ohlc=config.labels.target_definition == "open_to_close",
        n_identical_duplicates_removed=n_deduplicated,
    )

    metadata = dict(metadata)
    metadata.update(
        {
            "partial_bar_decision": decision.to_dict(),
            "validation": report.to_dict(),
            "rows_used": len(prices),
            "first_date": str(prices.index[0].date()),
            "last_date": str(prices.index[-1].date()),
        }
    )

    return OhlcvBundle(prices=prices, report=report, partial_bar=decision, metadata=metadata)
