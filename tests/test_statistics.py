"""Uncertainty quantification (P1.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_movement.statistics import (
    block_bootstrap_difference,
    block_bootstrap_returns,
    t_confidence_interval,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


# --------------------------------------------------------------------------
# Student-t intervals
# --------------------------------------------------------------------------
def test_t_interval_matches_manual_example():
    """Five folds, mean 0.52, hand-computed against the t(4) critical value."""
    values = np.array([0.50, 0.51, 0.52, 0.53, 0.54])
    interval = t_confidence_interval(values, confidence_level=0.95)

    mean = 0.52
    standard_error = values.std(ddof=1) / np.sqrt(5)
    critical = 2.776445  # t(0.975, df=4)

    assert interval.point == pytest.approx(mean)
    assert interval.low == pytest.approx(mean - critical * standard_error, rel=1e-4)
    assert interval.high == pytest.approx(mean + critical * standard_error, rel=1e-4)
    assert interval.n == 5
    assert interval.method == "student_t"


def test_t_interval_is_wider_than_a_normal_interval_for_small_samples():
    """With 5 folds the normal approximation is too optimistic."""
    values = np.array([0.48, 0.51, 0.52, 0.55, 0.53])
    interval = t_confidence_interval(values)

    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    normal_margin = 1.959964 * standard_error
    t_margin = interval.high - interval.point

    assert t_margin > normal_margin


def test_t_interval_excludes_detects_a_meaningful_result():
    clearly_above = t_confidence_interval(np.array([0.60, 0.61, 0.62, 0.59, 0.60]))
    assert clearly_above.excludes(0.50) is True

    noisy = t_confidence_interval(np.array([0.40, 0.60, 0.45, 0.55, 0.52]))
    assert noisy.excludes(0.50) is False


def test_t_interval_handles_degenerate_inputs():
    empty = t_confidence_interval(np.array([]))
    assert np.isnan(empty.point) and empty.n == 0

    single = t_confidence_interval(np.array([0.5]))
    assert single.point == pytest.approx(0.5) and np.isnan(single.low)

    constant = t_confidence_interval(np.array([0.5, 0.5, 0.5]))
    assert constant.low == constant.high == pytest.approx(0.5)


def test_t_interval_ignores_non_finite_values():
    interval = t_confidence_interval(np.array([0.5, np.nan, 0.6, np.inf]))
    assert interval.n == 2
    assert interval.point == pytest.approx(0.55)


# --------------------------------------------------------------------------
# block bootstrap
# --------------------------------------------------------------------------
def test_block_bootstrap_is_reproducible():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0004, 0.01, 500), index=_index(500))

    first = block_bootstrap_returns(returns, n_samples=200, seed=42)
    second = block_bootstrap_returns(returns, n_samples=200, seed=42)

    for name in first:
        assert first[name].interval.low == pytest.approx(second[name].interval.low)
        assert first[name].interval.high == pytest.approx(second[name].interval.high)


def test_block_bootstrap_changes_with_the_seed():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0004, 0.01, 500), index=_index(500))

    a = block_bootstrap_returns(returns, n_samples=200, seed=1)
    b = block_bootstrap_returns(returns, n_samples=200, seed=2)

    assert a["mean_daily_return"].interval.low != b["mean_daily_return"].interval.low


def test_bootstrap_output_length():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 0.01, 300), index=_index(300))

    results = block_bootstrap_returns(returns, n_samples=250, block_length=20, seed=7)

    assert set(results) == {"mean_daily_return", "annualized_return", "sharpe_ratio"}
    for result in results.values():
        assert result.n_samples == 250
        assert result.block_length == 20
        assert result.seed == 7


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, 400), index=_index(400))

    result = block_bootstrap_returns(returns, n_samples=500, seed=42)["mean_daily_return"]

    assert result.interval.low <= result.point <= result.interval.high


def test_bootstrap_detects_a_clearly_positive_series():
    returns = pd.Series(np.full(300, 0.001), index=_index(300))
    result = block_bootstrap_returns(returns, n_samples=200, seed=42)["mean_daily_return"]

    assert result.interval.excludes(0.0)
    assert result.fraction_above_zero == pytest.approx(1.0)


def test_bootstrap_does_not_claim_significance_for_pure_noise():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0, 0.01, 500), index=_index(500))

    result = block_bootstrap_returns(returns, n_samples=800, seed=42)["mean_daily_return"]

    assert not result.interval.excludes(0.0)


def test_bootstrap_handles_zero_variance():
    returns = pd.Series(np.zeros(100), index=_index(100))
    results = block_bootstrap_returns(returns, n_samples=50, seed=42)

    assert results["mean_daily_return"].point == pytest.approx(0.0)
    assert np.isnan(results["sharpe_ratio"].point)


def test_bootstrap_handles_too_few_rows():
    results = block_bootstrap_returns(pd.Series([0.01], index=_index(1)), n_samples=100)
    assert results["mean_daily_return"].n_samples == 0


def test_block_length_longer_than_the_series_is_clamped():
    returns = pd.Series(np.full(10, 0.001), index=_index(10))
    result = block_bootstrap_returns(returns, n_samples=20, block_length=1000, seed=42)
    assert np.isfinite(result["mean_daily_return"].point)


def test_bootstrap_serialisation_includes_the_verdict():
    returns = pd.Series(np.full(200, 0.001), index=_index(200))
    payload = block_bootstrap_returns(returns, n_samples=100, seed=42)["mean_daily_return"].to_dict()

    for key in ("statistic", "point", "ci_low", "ci_high", "interval_excludes_zero", "block_length"):
        assert key in payload


# --------------------------------------------------------------------------
# paired difference
# --------------------------------------------------------------------------
def test_bootstrap_comparison_uses_aligned_dates():
    """Overlapping but unequal indices must be inner-joined, not zipped."""
    strategy = pd.Series([0.01] * 100, index=_index(100))
    benchmark = pd.Series([0.005] * 80, index=_index(100)[:80])

    result = block_bootstrap_difference(strategy, benchmark, n_samples=100, seed=42)

    assert result.interval.n == 80
    assert result.point == pytest.approx(0.005)


def test_bootstrap_comparison_detects_a_real_difference():
    strategy = pd.Series([0.002] * 300, index=_index(300))
    benchmark = pd.Series([0.001] * 300, index=_index(300))

    result = block_bootstrap_difference(strategy, benchmark, n_samples=200, seed=42)

    assert result.point == pytest.approx(0.001)
    assert result.interval.excludes(0.0)


def test_bootstrap_comparison_on_identical_series_is_zero():
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(0.0, 0.01, 200), index=_index(200))

    result = block_bootstrap_difference(series, series.copy(), n_samples=100, seed=42)

    assert result.point == pytest.approx(0.0)
    assert result.interval.low == pytest.approx(0.0)
    assert result.interval.high == pytest.approx(0.0)


def test_bootstrap_comparison_without_shared_dates_fails():
    strategy = pd.Series([0.01] * 10, index=_index(10))
    benchmark = pd.Series([0.01] * 10, index=pd.bdate_range("2030-01-01", periods=10))

    with pytest.raises(ValueError, match="share no dates"):
        block_bootstrap_difference(strategy, benchmark)
