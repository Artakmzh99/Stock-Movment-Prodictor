"""Feature engineering.

Every function here is pure and *causal*: the value at row `t` may depend only on
rows `<= t`. Concretely that means:

* no ``.shift(-n)`` anywhere in this module;
* no ``rolling(..., center=True)``;
* no statistic computed over the full history (means, stds, min/max) — those are
  fitted inside the model Pipeline, per training fold, in ``models.py``.

Features are scale-free (ratios and returns) rather than raw prices, so that a
model trained on a $30 stock still means something when the stock trades at $200.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureConfig

EPSILON = 1e-12


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf on a zero denominator."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


# --------------------------------------------------------------------------
# Group A: lagged returns
# --------------------------------------------------------------------------
def lagged_returns(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Trailing simple return over each window, ending at t."""
    out = {f"return_{w}d": close.pct_change(w) for w in windows}
    return pd.DataFrame(out, index=close.index)


# --------------------------------------------------------------------------
# Group B: momentum
# --------------------------------------------------------------------------
def skip_momentum(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Momentum from t-w to t-1, deliberately skipping the most recent day.

    Plain w-day momentum would be perfectly collinear with ``return_{w}d``, which
    hurts logistic regression and tells us nothing new. Skipping the last day is
    the standard fix and separates medium-term trend from short-term reversal.
    """
    prev = close.shift(1)
    out = {f"momentum_{w}d": prev / close.shift(w) - 1.0 for w in windows}
    return pd.DataFrame(out, index=close.index)


def positive_day_counts(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Fraction of the last w days (inclusive of t) that closed up."""
    daily_up = (close.pct_change(1) > 0).astype(float)
    daily_up[close.pct_change(1).isna()] = np.nan
    out = {f"positive_days_last_{w}": daily_up.rolling(w, min_periods=w).sum() / w for w in windows}
    return pd.DataFrame(out, index=close.index)


# --------------------------------------------------------------------------
# Group C: moving-average ratios
# --------------------------------------------------------------------------
def sma_ratios(close: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Close-to-SMA and SMA-to-SMA ratios (never the raw SMA level)."""
    smas = {w: close.rolling(w, min_periods=w).mean() for w in windows}
    out: dict[str, pd.Series] = {}
    for w in windows:
        out[f"close_to_sma_{w}"] = _safe_divide(close, smas[w]) - 1.0
    ordered = sorted(windows)
    for short, long in itertools.pairwise(ordered):
        out[f"sma_{short}_to_sma_{long}"] = _safe_divide(smas[short], smas[long]) - 1.0
    return pd.DataFrame(out, index=close.index)


# --------------------------------------------------------------------------
# Group D: volatility
# --------------------------------------------------------------------------
def volatility_features(daily_return: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Trailing standard deviation of daily returns, plus a short/long vol ratio."""
    out = {f"volatility_{w}d": daily_return.rolling(w, min_periods=w).std(ddof=1) for w in windows}
    frame = pd.DataFrame(out, index=daily_return.index)
    ordered = sorted(windows)
    if len(ordered) >= 2:
        short, long = ordered[0], ordered[-1]
        frame[f"vol_ratio_{short}d_{long}d"] = (
            _safe_divide(frame[f"volatility_{short}d"], frame[f"volatility_{long}d"]) - 1.0
        )
    return frame


# --------------------------------------------------------------------------
# Group E: intraday range and gaps
# --------------------------------------------------------------------------
def range_and_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shape-of-the-bar features. All observable at the close of day t."""
    high, low, close, open_ = df["High"], df["Low"], df["Close"], df["Open"]
    span = high - low

    out = pd.DataFrame(index=df.index)
    out["intraday_range"] = _safe_divide(span, close)
    out["overnight_gap"] = _safe_divide(open_, close.shift(1)) - 1.0
    out["open_to_close_return"] = _safe_divide(close, open_) - 1.0
    # Where High == Low the bar has no range; 0.5 is the neutral "middle" encoding.
    position = _safe_divide(close - low, span)
    out["close_position_in_range"] = position.where(span > EPSILON, 0.5)
    return out


# --------------------------------------------------------------------------
# Group F: volume
# --------------------------------------------------------------------------
def volume_features(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """Relative volume only — absolute share counts are not comparable over time."""
    volume = volume.astype(float)
    rolling = volume.rolling(window, min_periods=window)
    mean, std = rolling.mean(), rolling.std(ddof=1)

    out = pd.DataFrame(index=volume.index)
    out["volume_change_1d"] = _safe_divide(volume, volume.shift(1)) - 1.0
    out[f"volume_to_sma_{window}"] = _safe_divide(volume, mean) - 1.0
    out[f"volume_zscore_{window}"] = _safe_divide(volume - mean, std)
    return out


# --------------------------------------------------------------------------
# Group G: benchmark-relative (optional)
# --------------------------------------------------------------------------
def benchmark_features(
    close: pd.Series,
    benchmark_close: pd.Series,
    windows: list[int],
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Return-relative and co-movement features versus a benchmark (e.g. SPY).

    The benchmark must already be aligned to exactly `close.index` — see
    ``dataset.align_benchmark``, which uses an inner join on shared trading dates.

    Forward-filling was the previous approach and it is wrong: it fabricates a
    benchmark price for a date the benchmark did not trade, which then produces a
    fictitious 0% benchmark return and a distorted beta. Restricting to genuinely
    shared sessions loses a handful of rows and invents nothing.
    """
    if not benchmark_close.index.equals(close.index):
        raise ValueError(
            "benchmark series must be pre-aligned to the asset index "
            "(use dataset.align_benchmark, which inner-joins shared sessions)"
        )
    bench = benchmark_close
    asset_ret = close.pct_change(1)
    bench_ret = bench.pct_change(1)

    out = pd.DataFrame(index=close.index)
    out["benchmark_return_1d"] = bench_ret
    for w in windows:
        out[f"relative_return_{w}d"] = close.pct_change(w) - bench.pct_change(w)

    cov = asset_ret.rolling(rolling_window, min_periods=rolling_window).cov(bench_ret)
    var = bench_ret.rolling(rolling_window, min_periods=rolling_window).var(ddof=1)
    out[f"rolling_beta_{rolling_window}"] = _safe_divide(cov, var)
    out[f"rolling_correlation_{rolling_window}"] = asset_ret.rolling(
        rolling_window, min_periods=rolling_window
    ).corr(bench_ret)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix from an OHLCV frame.

    Returns features only — no target, no future_return. NaNs from incomplete
    rolling windows are left in place; they are dropped in ``dataset.py`` after
    the label is attached, so features and labels stay aligned.
    """
    close = df["Close"].astype(float)
    daily_return = close.pct_change(1)

    blocks = [
        lagged_returns(close, list(config.return_windows)),
        # Momentum windows are configured independently of volatility windows.
        # Deriving one from the other meant that retuning volatility silently
        # changed which momentum features existed.
        skip_momentum(close, list(config.momentum_windows)),
        positive_day_counts(close, list(config.positive_day_windows)),
        sma_ratios(close, list(config.sma_windows)),
        volatility_features(daily_return, list(config.volatility_windows)),
        range_and_gap_features(df),
        volume_features(df["Volume"], config.volume_window),
    ]

    if config.use_benchmark_features:
        if benchmark_close is None:
            raise ValueError("features.use_benchmark_features is true but no benchmark series was provided")
        blocks.append(
            benchmark_features(
                close,
                benchmark_close,
                list(config.benchmark_windows),
                config.benchmark_rolling_window,
            )
        )

    features = pd.concat(blocks, axis=1)

    duplicates = features.columns[features.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"duplicate feature names produced: {sorted(set(duplicates))}")

    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = df.index.name
    return features


def feature_manifest(features: pd.DataFrame, config: FeatureConfig) -> dict[str, Any]:
    """Serialisable description of the feature matrix, saved with every run.

    ``feature_names`` is ordered and authoritative: inference reindexes incoming
    data to exactly this order, because a silently permuted column order produces
    confident nonsense rather than an error.
    """
    from .config import FEATURE_VERSION

    return {
        "feature_version": FEATURE_VERSION,
        "n_features": int(features.shape[1]),
        "feature_names": list(features.columns),
        "config": {
            "return_windows": list(config.return_windows),
            "momentum_windows": list(config.momentum_windows),
            "sma_windows": list(config.sma_windows),
            "volatility_windows": list(config.volatility_windows),
            "positive_day_windows": list(config.positive_day_windows),
            "volume_window": config.volume_window,
            "use_benchmark_features": config.use_benchmark_features,
            "benchmark_windows": list(config.benchmark_windows),
            "benchmark_rolling_window": config.benchmark_rolling_window,
        },
        "max_window": config.max_window,
    }
