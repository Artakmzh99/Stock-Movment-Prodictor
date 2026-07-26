"""Backtest arithmetic, verified against hand-computed examples.

The cost tests are regression tests for a real defect: next-open execution
charged cost only on position *changes*, so three consecutive intraday long days
paid one round trip instead of three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_movement.backtest import (
    ALWAYS_LONG_INTRADAY,
    BUY_AND_HOLD,
    always_active,
    backtest_comparison,
    buy_and_hold_close_to_close,
    cash,
    compute_close_to_close_costs,
    compute_costs,
    compute_intraday_round_trip_costs,
    drawdown_series,
    max_drawdown,
    positions_from_probability,
    run_backtest,
    trade_returns_net,
)
from stock_movement.config import BacktestConfig


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


@pytest.fixture
def free_intraday() -> BacktestConfig:
    return BacktestConfig(execution_mode="next_open", round_trip_cost_bps=0.0, one_way_cost_bps=0.0)


@pytest.fixture
def free_holding() -> BacktestConfig:
    return BacktestConfig(execution_mode="close_to_close", round_trip_cost_bps=0.0, one_way_cost_bps=0.0)


# --------------------------------------------------------------------------
# P0.1 — cost models
# --------------------------------------------------------------------------
def test_next_open_charges_cost_every_active_day():
    """The documented example: position [1,1,0,1] at 0.001 round trip."""
    position = pd.Series([1.0, 1.0, 0.0, 1.0], index=_index(4))

    costs = compute_intraday_round_trip_costs(position, 0.001)

    np.testing.assert_allclose(costs.to_numpy(), [0.001, 0.001, 0.000, 0.001])


def test_three_consecutive_intraday_long_days_pay_three_costs():
    """Regression test for the original defect.

    Each session is entered at the open and exited at the close, so three
    consecutive long sessions are three round trips. Charging on position change
    would have produced a single cost here.
    """
    position = pd.Series([1.0, 1.0, 1.0], index=_index(3))

    costs = compute_intraday_round_trip_costs(position, 0.001)

    assert costs.sum() == pytest.approx(0.003)
    assert (costs == 0.001).all()

    # And the buggy model really does differ, so this test is not vacuous.
    change_based = compute_close_to_close_costs(position, 0.001)
    assert change_based.sum() == pytest.approx(0.001)


def test_intraday_cash_days_pay_zero_cost():
    position = pd.Series([0.0, 0.0, 1.0, 0.0], index=_index(4))
    costs = compute_intraday_round_trip_costs(position, 0.002)

    np.testing.assert_allclose(costs.to_numpy(), [0.0, 0.0, 0.002, 0.0])


def test_close_to_close_costs_follow_position_changes():
    position = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=_index(5))

    costs = compute_close_to_close_costs(position, 0.0007)

    assert costs.iloc[0] == pytest.approx(0.0007)  # open from flat
    assert costs.iloc[1] == pytest.approx(0.0)  # held: free
    assert costs.iloc[2] == pytest.approx(0.0)
    assert costs.iloc[3] == pytest.approx(0.0007)  # closed
    assert costs.iloc[4] == pytest.approx(0.0)
    assert costs.sum() == pytest.approx(0.0014)


def test_execution_modes_produce_different_cost_series():
    """execution_mode must change the arithmetic, not just metadata."""
    position = pd.Series([1.0, 1.0, 1.0, 1.0], index=_index(4))

    intraday = compute_costs(position, BacktestConfig(execution_mode="next_open", round_trip_cost_bps=10.0))
    holding = compute_costs(
        position,
        BacktestConfig(execution_mode="close_to_close", one_way_cost_bps=5.0),
    )

    assert intraday.sum() > holding.sum()
    assert intraday.sum() == pytest.approx(4 * 0.001)
    assert holding.sum() == pytest.approx(0.0005)
    assert not np.allclose(intraday.to_numpy(), holding.to_numpy())


def test_short_positions_pay_cost_on_absolute_size():
    position = pd.Series([-1.0, 0.0, 1.0], index=_index(3))
    costs = compute_intraday_round_trip_costs(position, 0.001)
    np.testing.assert_allclose(costs.to_numpy(), [0.001, 0.0, 0.001])


def test_negative_cost_rate_is_rejected():
    position = pd.Series([1.0], index=_index(1))
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_intraday_round_trip_costs(position, -0.001)
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_close_to_close_costs(position, -0.001)


def test_unknown_execution_mode_is_rejected():
    position = pd.Series([1.0], index=_index(1))
    config = BacktestConfig().model_copy(update={"execution_mode": "teleportation"})
    with pytest.raises(ValueError, match="unknown execution_mode"):
        compute_costs(position, config)


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------
def test_intraday_strategy_is_not_named_buy_and_hold():
    """An intraday strategy that re-enters daily is not buy-and-hold."""
    future_return = pd.Series([0.01] * 5, index=_index(5))
    result = always_active(future_return, BacktestConfig(execution_mode="next_open"))

    assert result.name == ALWAYS_LONG_INTRADAY
    assert result.name != BUY_AND_HOLD
    assert "buy_and_hold" not in result.name


def test_close_to_close_always_active_is_buy_and_hold():
    future_return = pd.Series([0.01] * 5, index=_index(5))
    result = always_active(future_return, BacktestConfig(execution_mode="close_to_close"))
    assert result.name == BUY_AND_HOLD


def test_always_long_intraday_is_worse_than_holding():
    """Paying a round trip every session must cost more than holding."""
    close = pd.Series(100.0 * (1.01 ** np.arange(60)), index=_index(60))
    future_return = pd.Series([0.01] * 60, index=_index(60))
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=10.0)

    intraday = always_active(future_return, config)
    holding = buy_and_hold_close_to_close(close, config)

    assert intraday.metrics["total_cost_paid"] > holding.metrics["total_cost_paid"]
    assert intraday.metrics["cumulative_return"] < holding.metrics["cumulative_return"]


def test_buy_and_hold_close_to_close_uses_market_returns():
    close = pd.Series([100.0, 110.0, 99.0, 108.9], index=_index(4))
    config = BacktestConfig(execution_mode="next_open", one_way_cost_bps=0.0, round_trip_cost_bps=0.0)

    result = buy_and_hold_close_to_close(close, config)

    # First row return is 0 (no prior close), then the actual market moves.
    assert result.execution_mode == "close_to_close"
    assert result.metrics["cumulative_return"] == pytest.approx(108.9 / 100.0 - 1.0)
    assert result.metrics["n_position_changes"] == 1


# --------------------------------------------------------------------------
# trade metrics
# --------------------------------------------------------------------------
def test_intraday_completed_trades_equal_active_sessions():
    position = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0], index=_index(5))
    future_return = pd.Series([0.01, -0.02, 0.05, 0.03, -0.01], index=_index(5))
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=10.0)

    result = run_backtest(position, future_return, config)

    assert result.metrics["n_completed_trades"] == result.metrics["n_active_sessions"] == 4
    # Position changes are a different quantity entirely.
    assert result.metrics["n_position_changes"] == 3


def test_trade_metrics_are_net_of_costs():
    """A tiny gross gain must become a net loss once the round trip is paid."""
    position = pd.Series([1.0, 1.0], index=_index(2))
    future_return = pd.Series([0.0005, 0.0005], index=_index(2))  # +5 bps gross
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=10.0)  # -10 bps

    result = run_backtest(position, future_return, config)

    assert result.metrics["average_trade_return"] < 0
    assert result.metrics["best_trade"] < 0
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    np.testing.assert_allclose(result.trade_returns, [-0.0005, -0.0005])


def test_close_to_close_trades_compound_within_a_block():
    position = pd.Series([1.0, 1.0, 0.0, 1.0], index=_index(4))
    net = pd.Series([0.10, 0.10, 0.0, -0.20], index=_index(4))

    trades = trade_returns_net(position, net, "close_to_close")

    assert len(trades) == 2
    assert trades[0] == pytest.approx(1.10 * 1.10 - 1.0)
    assert trades[1] == pytest.approx(-0.20)


def test_intraday_trades_are_per_session():
    position = pd.Series([1.0, 1.0, 0.0], index=_index(3))
    net = pd.Series([0.10, 0.20, 0.0], index=_index(3))

    trades = trade_returns_net(position, net, "next_open")

    assert trades == pytest.approx([0.10, 0.20])


def test_trade_statistics_on_a_hand_computed_example():
    position = pd.Series([1.0, 1.0, 1.0, 1.0], index=_index(4))
    future_return = pd.Series([0.10, -0.05, 0.20, -0.01], index=_index(4))
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=0.0)

    result = run_backtest(position, future_return, config)

    assert result.metrics["best_trade"] == pytest.approx(0.20)
    assert result.metrics["worst_trade"] == pytest.approx(-0.05)
    assert result.metrics["median_trade_return"] == pytest.approx(0.045)
    assert result.metrics["win_rate"] == pytest.approx(0.5)
    assert result.metrics["profit_factor"] == pytest.approx(0.30 / 0.06)


def test_profit_factor_edge_cases():
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=0.0)

    all_wins = run_backtest(
        pd.Series([1.0, 1.0], index=_index(2)), pd.Series([0.01, 0.02], index=_index(2)), config
    )
    assert all_wins.metrics["profit_factor"] == float("inf")

    no_trades = run_backtest(
        pd.Series([0.0, 0.0], index=_index(2)), pd.Series([0.01, 0.02], index=_index(2)), config
    )
    assert np.isnan(no_trades.metrics["profit_factor"])


# --------------------------------------------------------------------------
# alignment and arithmetic
# --------------------------------------------------------------------------
def test_signal_is_aligned_with_the_return_it_earns(free_intraday):
    index = _index(4)
    position = pd.Series([1.0, 0.0, 1.0, 0.0], index=index)
    future_return = pd.Series([0.10, -0.50, 0.20, -0.90], index=index)

    result = run_backtest(position, future_return, free_intraday)

    np.testing.assert_allclose(result.returns.to_numpy(), [0.10, 0.0, 0.20, 0.0])
    assert result.equity_curve.iloc[-1] == pytest.approx(1.10 * 1.20)


def test_shifting_the_signal_changes_the_result(free_intraday):
    """Guards the test above: if a shift were harmless, that test proves nothing."""
    index = _index(4)
    position = pd.Series([1.0, 0.0, 1.0, 0.0], index=index)
    future_return = pd.Series([0.10, -0.50, 0.20, -0.90], index=index)

    aligned = run_backtest(position, future_return, free_intraday)
    shifted = run_backtest(position.shift(1).fillna(0.0), future_return, free_intraday)

    assert aligned.equity_curve.iloc[-1] != pytest.approx(shifted.equity_curve.iloc[-1])
    assert shifted.equity_curve.iloc[-1] < 1.0  # the shifted version buys the crashes


def test_cumulative_return_matches_manual_compounding(free_intraday):
    position = pd.Series([1.0, 1.0, 1.0], index=_index(3))
    future_return = pd.Series([0.10, 0.10, -0.20], index=_index(3))

    result = run_backtest(position, future_return, free_intraday)

    assert result.metrics["cumulative_return"] == pytest.approx(1.10 * 1.10 * 0.80 - 1.0)


def test_costs_reduce_net_return_below_gross():
    config = BacktestConfig(execution_mode="next_open", round_trip_cost_bps=15.0)
    position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=_index(6))
    future_return = pd.Series([0.01] * 6, index=_index(6))

    result = run_backtest(position, future_return, config)

    assert result.metrics["cumulative_return"] < result.metrics["gross_cumulative_return"]
    assert result.metrics["total_cost_paid"] == pytest.approx(6 * 0.0015)
    assert result.metrics["average_cost_per_active_session"] == pytest.approx(0.0015)


def test_cash_earns_exactly_zero(free_intraday):
    future_return = pd.Series([0.05, -0.03, 0.02], index=_index(3))
    result = cash(future_return, free_intraday)

    assert (result.returns == 0).all()
    assert result.metrics["cumulative_return"] == pytest.approx(0.0)
    assert result.metrics["exposure"] == pytest.approx(0.0)
    assert result.metrics["n_active_sessions"] == 0
    assert result.metrics["n_completed_trades"] == 0
    assert result.metrics["total_cost_paid"] == pytest.approx(0.0)


def test_max_drawdown_on_a_hand_computed_series():
    #  100 -> 120 -> 60 -> 90 : worst peak-to-trough is 120 -> 60 == -50%
    equity = pd.Series([1.00, 1.20, 0.60, 0.90], index=_index(4))
    assert max_drawdown(equity) == pytest.approx(-0.5)

    drawdown = drawdown_series(equity)
    np.testing.assert_allclose(drawdown.to_numpy(), [0.0, 0.0, -0.5, -0.25])


def test_max_drawdown_of_a_rising_series_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3], index=_index(4))) == pytest.approx(0.0)


def test_threshold_governs_position_taking():
    proba = pd.Series([0.10, 0.49, 0.51, 0.90], index=_index(4))

    np.testing.assert_array_equal(
        positions_from_probability(proba, threshold=0.5).to_numpy(), [0.0, 0.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(
        positions_from_probability(proba, threshold=0.6).to_numpy(), [0.0, 0.0, 0.0, 1.0]
    )


def test_short_positions_when_enabled():
    proba = pd.Series([0.10, 0.50, 0.80], index=_index(3))
    position = positions_from_probability(proba, threshold=0.6, allow_short=True, short_threshold=0.4)
    np.testing.assert_array_equal(position.to_numpy(), [-1.0, 0.0, 1.0])


def test_short_threshold_above_threshold_is_rejected():
    proba = pd.Series([0.5], index=_index(1))
    with pytest.raises(ValueError, match="must be <="):
        positions_from_probability(proba, threshold=0.4, allow_short=True, short_threshold=0.6)


def test_mismatched_index_is_rejected(free_intraday):
    position = pd.Series([1.0, 0.0], index=_index(2))
    future_return = pd.Series([0.01, 0.02], index=pd.bdate_range("2021-06-01", periods=2))

    with pytest.raises(ValueError, match="identical index"):
        run_backtest(position, future_return, free_intraday)


def test_empty_position_is_rejected(free_intraday):
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty position"):
        run_backtest(empty, empty, free_intraday)


def test_sharpe_is_annualized_consistently(free_holding):
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 504), index=_index(504))
    result = always_active(returns, free_holding)

    manual = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert result.metrics["sharpe_ratio"] == pytest.approx(manual, rel=1e-9)
    assert result.metrics["years"] == pytest.approx(2.0)


def test_comparison_table_reports_execution_mode():
    future_return = pd.Series([0.01] * 10, index=_index(10))
    config = BacktestConfig(execution_mode="next_open")

    frame = backtest_comparison([always_active(future_return, config), cash(future_return, config)])

    assert "execution_mode" in frame.columns
    assert (frame["execution_mode"] == "next_open").all()
    assert ALWAYS_LONG_INTRADAY in frame.index
    for column in ("n_position_changes", "n_active_sessions", "n_completed_trades"):
        assert column in frame.columns
