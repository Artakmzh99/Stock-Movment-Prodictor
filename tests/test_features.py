"""Feature correctness — above all, causality — plus P1.5/P1.6 fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_movement.config import FeatureConfig, config_from_dict
from stock_movement.dataset import align_benchmark
from stock_movement.features import (
    benchmark_features,
    build_features,
    feature_manifest,
    lagged_returns,
    positive_day_counts,
    range_and_gap_features,
    skip_momentum,
    sma_ratios,
    volatility_features,
    volume_features,
)
from stock_movement.validation import DataValidationError
from tests.conftest import small_config_payload


def test_lagged_returns_match_manual_computation():
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=pd.bdate_range("2020-01-01", periods=4))
    result = lagged_returns(close, [1, 2])

    assert np.isnan(result["return_1d"].iloc[0])
    assert result["return_1d"].iloc[1] == pytest.approx(0.10)
    assert result["return_1d"].iloc[2] == pytest.approx(-0.10)
    assert result["return_2d"].iloc[2] == pytest.approx(99.0 / 100.0 - 1.0)


def test_rolling_features_are_nan_until_the_window_is_full(ohlcv):
    """A 10-day average cannot exist on day 9. If it does, the window is wrong."""
    close = ohlcv["Close"]
    ratios = sma_ratios(close, [5, 10])

    assert ratios["close_to_sma_10"].iloc[:9].isna().all()
    assert not np.isnan(ratios["close_to_sma_10"].iloc[9])

    vol = volatility_features(close.pct_change(), [5])
    # One extra NaN because pct_change itself consumes the first observation.
    assert vol["volatility_5d"].iloc[:5].isna().all()
    assert not np.isnan(vol["volatility_5d"].iloc[5])


def test_sma_ratio_equals_explicit_formula(ohlcv):
    close = ohlcv["Close"]
    expected = close / close.rolling(20, min_periods=20).mean() - 1.0
    pd.testing.assert_series_equal(
        sma_ratios(close, [10, 20])["close_to_sma_20"], expected, check_names=False
    )


def test_changing_a_future_price_cannot_change_a_past_feature(ohlcv, config):
    """The central causality guarantee.

    Rewrite the second half of the price history, rebuild everything, and require
    that every feature value in the first half is bit-for-bit identical. Any
    forward-looking window, centred rolling, or full-sample statistic breaks this.
    """
    cutoff = len(ohlcv) // 2
    original = build_features(ohlcv, config.features)

    tampered_prices = ohlcv.copy()
    tampered_prices.iloc[cutoff:] *= 3.7
    tampered_prices.loc[tampered_prices.index[cutoff:], "Volume"] = 12345.0

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff - 1],
        build_features(tampered_prices, config.features).iloc[: cutoff - 1],
        check_exact=True,
    )


def test_no_feature_uses_a_negative_shift():
    """Static guard: a `shift(-n)` in this module would silently leak the future."""
    from pathlib import Path

    import stock_movement.features as features_module

    source = Path(features_module.__file__).read_text()
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith(("#", '"', "*")))

    assert "shift(-" not in code, "negative shift found in features.py — that is future data"
    assert "center=True" not in code, "centred rolling window found — it straddles the present"


def test_close_position_in_range_handles_a_zero_range_bar():
    frame = pd.DataFrame(
        {"Open": [10.0], "High": [10.0], "Low": [10.0], "Close": [10.0]},
        index=pd.bdate_range("2020-01-01", periods=1),
    )
    result = range_and_gap_features(frame)

    assert result["close_position_in_range"].iloc[0] == pytest.approx(0.5)
    assert np.isfinite(result["intraday_range"].iloc[0])


def test_volume_features_survive_a_zero_volume_day():
    volume = pd.Series([0.0] * 12, index=pd.bdate_range("2020-01-01", periods=12))
    values = volume_features(volume, window=5).to_numpy()
    assert not np.isinf(values[~np.isnan(values)]).any()


def test_positive_day_fraction_is_bounded():
    close = pd.Series(np.linspace(100, 120, 30), index=pd.bdate_range("2020-01-01", periods=30))
    fraction = positive_day_counts(close, [5])["positive_days_last_5"].dropna()

    assert ((fraction >= 0) & (fraction <= 1)).all()
    assert fraction.iloc[-1] == pytest.approx(1.0)  # monotonically rising series


def test_build_features_produces_unique_finite_columns(ohlcv, config):
    features = build_features(ohlcv, config.features)
    values = features.to_numpy()

    assert features.columns.is_unique
    assert not np.isinf(values[~np.isnan(values)]).any()
    assert features.index.equals(ohlcv.index)


# --------------------------------------------------------------------------
# P1.6 momentum windows are independent of volatility windows
# --------------------------------------------------------------------------
def test_momentum_windows_independent_of_volatility(ohlcv):
    config = FeatureConfig(momentum_windows=(3, 7), volatility_windows=(5, 10, 20))
    features = build_features(ohlcv, config)

    assert "momentum_3d" in features.columns
    assert "momentum_7d" in features.columns
    assert "momentum_5d" not in features.columns  # would leak from volatility windows
    assert "volatility_20d" in features.columns


def test_changing_volatility_does_not_change_momentum_columns(ohlcv):
    """Regression test: momentum used to be derived from volatility_windows."""
    base = build_features(ohlcv, FeatureConfig(momentum_windows=(5, 10), volatility_windows=(5, 10)))
    changed = build_features(ohlcv, FeatureConfig(momentum_windows=(5, 10), volatility_windows=(15, 30)))

    momentum_base = [c for c in base.columns if c.startswith("momentum_")]
    momentum_changed = [c for c in changed.columns if c.startswith("momentum_")]

    assert momentum_base == momentum_changed == ["momentum_5d", "momentum_10d"]
    pd.testing.assert_frame_equal(base[momentum_base], changed[momentum_changed])


def test_manifest_records_momentum_windows(ohlcv):
    config = FeatureConfig(momentum_windows=(4, 9))
    manifest = feature_manifest(build_features(ohlcv, config), config)

    assert manifest["config"]["momentum_windows"] == [4, 9]
    assert manifest["feature_version"]
    assert manifest["max_window"] == config.max_window


def test_skip_momentum_excludes_the_most_recent_day():
    close = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0, 200.0], index=pd.bdate_range("2020-01-01", periods=6)
    )
    momentum = skip_momentum(close, [5])["momentum_5d"]
    # The final jump lands on day t and must be excluded from t's own momentum.
    assert momentum.iloc[5] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# P1.5 benchmark alignment
# --------------------------------------------------------------------------
def test_benchmark_uses_shared_dates_only():
    index = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(np.linspace(100, 120, 100), index=index)
    # The benchmark missed 3 sessions.
    benchmark = close.drop(index[[10, 20, 30]]) * 0.5

    asset, aligned, alignment = align_benchmark(close, benchmark, "SPY", max_join_loss=0.05)

    assert len(asset) == len(aligned) == 97
    assert alignment.n_removed == 3
    assert alignment.removed_fraction == pytest.approx(0.03)
    assert asset.index.equals(aligned.index)


def test_benchmark_does_not_forward_fill_missing_session():
    """Forward-filling invents a price for a day the benchmark did not trade."""
    index = pd.bdate_range("2020-01-01", periods=50)
    close = pd.Series(np.linspace(100, 110, 50), index=index)
    benchmark = close.drop(index[[25]])

    _, aligned, _ = align_benchmark(close, benchmark, "SPY")

    assert index[25] not in aligned.index
    assert not aligned.index.duplicated().any()


def test_benchmark_join_records_removed_dates():
    index = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.linspace(100, 110, 60), index=index)
    benchmark = close.drop(index[[5, 6]])

    # 2 of 60 rows is 3.3%, above the 2% default; this test is about the record,
    # not the limit, so the limit is relaxed here.
    _, _, alignment = align_benchmark(close, benchmark, "SPY", max_join_loss=0.05)
    payload = alignment.to_dict()

    assert payload["benchmark_ticker"] == "SPY"
    assert payload["n_target_rows_before"] == 60
    assert payload["n_rows_after"] == 58
    assert str(index[5].date()) in payload["removed_dates"]


def test_excessive_join_loss_fails():
    """Two instruments that barely share a calendar must not be silently paired."""
    index = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(np.linspace(100, 120, 100), index=index)
    benchmark = close.iloc[:50]

    with pytest.raises(DataValidationError, match="do not share a trading"):
        align_benchmark(close, benchmark, "SPY", max_join_loss=0.02)


def test_benchmark_features_require_pre_aligned_input(ohlcv):
    close = ohlcv["Close"]
    misaligned = close.iloc[:-5] * 0.5

    with pytest.raises(ValueError, match="pre-aligned"):
        benchmark_features(close, misaligned, [1, 5], 20)


def test_benchmark_features_never_read_the_future(ohlcv):
    config = FeatureConfig(use_benchmark_features=True)
    benchmark = ohlcv["Close"] * 0.5
    cutoff = len(ohlcv) // 2

    original = build_features(ohlcv, config, benchmark_close=benchmark)

    tampered = benchmark.copy()
    tampered.iloc[cutoff:] *= 5.0

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff - 1],
        build_features(ohlcv, config, benchmark_close=tampered).iloc[: cutoff - 1],
        check_exact=True,
    )


def test_benchmark_features_are_present_when_enabled(ohlcv):
    config = FeatureConfig(use_benchmark_features=True, benchmark_rolling_window=20)
    features = build_features(ohlcv, config, benchmark_close=ohlcv["Close"] * 0.5)

    for name in ("benchmark_return_1d", "relative_return_1d", "rolling_beta_20", "rolling_correlation_20"):
        assert name in features.columns


def test_missing_benchmark_series_is_rejected(ohlcv):
    with pytest.raises(ValueError, match="no benchmark series"):
        build_features(ohlcv, FeatureConfig(use_benchmark_features=True))


def test_benchmark_dataset_records_alignment(monkeypatch, tmp_path):
    """The full dataset path must record the join, not just perform it."""
    from stock_movement.dataset import build_dataset
    from tests.conftest import make_ohlcv

    asset = make_ohlcv(n=1400, seed=11)
    benchmark = make_ohlcv(n=1400, seed=12)

    def fake_download(ticker, start, end, interval="1d", auto_adjust=True):
        return (benchmark if ticker == "SPY" else asset).copy()

    monkeypatch.setattr("stock_movement.data.download_ohlcv", fake_download)

    config = config_from_dict(
        small_config_payload(
            data={"ticker": "TEST", "benchmark_ticker": "SPY", "min_rows": 200, "end_date": "2026-01-02"},
            features={"use_benchmark_features": True},
            paths={"data_raw": str(tmp_path / "raw"), "runs": str(tmp_path / "runs")},
        )
    )

    dataset = build_dataset(config)
    alignment = dataset.metadata["benchmark_alignment"]

    assert alignment["benchmark_ticker"] == "SPY"
    assert alignment["n_rows_after"] > 0
    assert any(c.startswith("benchmark_") for c in dataset.feature_names)
