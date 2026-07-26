"""Uncertainty quantification.

A point estimate on ~800 noisy sessions is close to meaningless on its own. Two
tools here:

**Student-t intervals** for fold metrics. With 5 folds the normal approximation is
badly wrong in the tails, so the t distribution with n-1 degrees of freedom is the
right choice.

**Moving-block bootstrap** for daily strategy returns. A plain i.i.d. bootstrap
destroys the autocorrelation and volatility clustering that dominate financial
series, which makes intervals far too narrow and Sharpe ratios look far more
certain than they are. Resampling contiguous blocks of 20 sessions preserves
short-range dependence.

Everything is seeded and therefore reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    confidence_level: float
    method: str
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def excludes(self, value: float) -> bool:
        """Whether `value` lies outside the interval — e.g. 0.50 for accuracy."""
        return not (self.low <= value <= self.high)


def t_confidence_interval(
    values: Any,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Two-sided Student-t interval for the mean of a small sample."""
    from scipy import stats

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)

    if n == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), confidence_level, "student_t", 0)
    mean = float(array.mean())
    if n == 1:
        return ConfidenceInterval(mean, float("nan"), float("nan"), confidence_level, "student_t", 1)

    standard_error = float(array.std(ddof=1) / np.sqrt(n))
    if standard_error == 0.0:
        return ConfidenceInterval(mean, mean, mean, confidence_level, "student_t", n)

    critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, df=n - 1))
    margin = critical * standard_error
    return ConfidenceInterval(mean, mean - margin, mean + margin, confidence_level, "student_t", n)


# --------------------------------------------------------------------------
# moving-block bootstrap
# --------------------------------------------------------------------------
def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block resample of length `n`."""
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def _annualize(returns: np.ndarray, periods: int) -> float:
    if len(returns) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + returns))
    years = len(returns) / periods
    if years <= 0 or growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def _sharpe(returns: np.ndarray, periods: int) -> float:
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods))


@dataclass
class BootstrapResult:
    statistic: str
    point: float
    interval: ConfidenceInterval
    n_samples: int
    block_length: int
    seed: int
    #: Fraction of resamples where the statistic exceeded zero — a one-sided read.
    fraction_above_zero: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "point": self.point,
            "ci_low": self.interval.low,
            "ci_high": self.interval.high,
            "confidence_level": self.interval.confidence_level,
            "n_samples": self.n_samples,
            "block_length": self.block_length,
            "seed": self.seed,
            "fraction_above_zero": self.fraction_above_zero,
            "interval_excludes_zero": self.interval.excludes(0.0),
        }


def _percentile_interval(
    draws: np.ndarray, point: float, confidence_level: float, n: int
) -> ConfidenceInterval:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return ConfidenceInterval(point, float("nan"), float("nan"), confidence_level, "block_bootstrap", n)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return ConfidenceInterval(point, float(low), float(high), confidence_level, "block_bootstrap", n)


def block_bootstrap_returns(
    returns: pd.Series | np.ndarray,
    n_samples: int = 2000,
    block_length: int = 20,
    seed: int = 42,
    confidence_level: float = 0.95,
    trading_days_per_year: int = 252,
) -> dict[str, BootstrapResult]:
    """Bootstrap intervals for mean daily return, annualised return and Sharpe."""
    array = np.asarray(returns, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)

    statistics: dict[str, Callable[[np.ndarray], float]] = {
        "mean_daily_return": lambda r: float(r.mean()) if len(r) else float("nan"),
        "annualized_return": lambda r: _annualize(r, trading_days_per_year),
        "sharpe_ratio": lambda r: _sharpe(r, trading_days_per_year),
    }

    if n < 2:
        return {
            name: BootstrapResult(
                name,
                fn(array),
                ConfidenceInterval(
                    fn(array), float("nan"), float("nan"), confidence_level, "block_bootstrap", n
                ),
                0,
                block_length,
                seed,
                float("nan"),
            )
            for name, fn in statistics.items()
        }

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(n_samples):
        sample = array[_block_indices(n, block_length, rng)]
        for name, fn in statistics.items():
            draws[name].append(fn(sample))

    results: dict[str, BootstrapResult] = {}
    for name, fn in statistics.items():
        values = np.asarray(draws[name], dtype=float)
        point = fn(array)
        finite = values[np.isfinite(values)]
        results[name] = BootstrapResult(
            statistic=name,
            point=point,
            interval=_percentile_interval(values, point, confidence_level, n),
            n_samples=n_samples,
            block_length=block_length,
            seed=seed,
            fraction_above_zero=float((finite > 0).mean()) if len(finite) else float("nan"),
        )
    return results


def block_bootstrap_difference(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_samples: int = 2000,
    block_length: int = 20,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Interval for the difference in mean daily return, on **aligned dates**.

    Resampling the two series independently would break the pairing and understate
    the uncertainty, so blocks are drawn once and applied to both.
    """
    if not strategy_returns.index.equals(benchmark_returns.index):
        shared = strategy_returns.index.intersection(benchmark_returns.index)
        if len(shared) == 0:
            raise ValueError("strategy and benchmark returns share no dates")
        strategy_returns = strategy_returns.loc[shared]
        benchmark_returns = benchmark_returns.loc[shared]

    difference = (strategy_returns - benchmark_returns).to_numpy(dtype=float)
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    point = float(difference.mean()) if n else float("nan")

    if n < 2:
        return BootstrapResult(
            "mean_daily_return_difference",
            point,
            ConfidenceInterval(point, float("nan"), float("nan"), confidence_level, "block_bootstrap", n),
            0,
            block_length,
            seed,
            float("nan"),
        )

    rng = np.random.default_rng(seed)
    draws = np.array(
        [float(difference[_block_indices(n, block_length, rng)].mean()) for _ in range(n_samples)]
    )

    return BootstrapResult(
        statistic="mean_daily_return_difference",
        point=point,
        interval=_percentile_interval(draws, point, confidence_level, n),
        n_samples=n_samples,
        block_length=block_length,
        seed=seed,
        fraction_above_zero=float((draws > 0).mean()),
    )
