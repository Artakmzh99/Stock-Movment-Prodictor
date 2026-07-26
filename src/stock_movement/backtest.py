"""Cost-aware backtesting with two genuinely different execution models.

## Alignment

::

    strategy_return[t] = position[t] * future_return[t] - cost[t]

``future_return[t]`` is the return earned *after* t, and ``position[t]`` is
decided using information available *at* t. They are aligned by construction,
which is why no ``.shift()`` appears here. A stray shift either destroys the
signal or leaks the future, and both produce plausible-looking equity curves.
``tests/test_backtest.py`` pins this with hand-computed examples.

## The two cost models

These are not presentational variants — they compute different numbers, and using
the wrong one understates costs by an order of magnitude.

**close_to_close** — a position is held across sessions. Cost is paid only when
the position *changes*::

    cost_t = |position_t - position_{t-1}| * one_way_cost_rate

**next_open** — enter at the open of the session, exit at its close. Every active
session is a *complete round trip*, so every active session pays::

    cost_t = |position_t| * round_trip_cost_rate

Holding `position = [1, 1, 1]` means three separate intraday trades and three
round-trip costs, not one. Charging on position change here would report a third
of the real cost on a three-day run and near-zero on a long one.

## Benchmark naming

In intraday mode the always-active strategy is ``always_long_intraday``: it
re-enters every session and pays a round trip each time. It is emphatically *not*
buy-and-hold. True buy-and-hold is reported separately as
``buy_and_hold_close_to_close`` using close-to-close market returns, since that is
what actually holding the asset would have earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig, ExecutionMode

ALWAYS_LONG_INTRADAY = "always_long_intraday"
BUY_AND_HOLD = "buy_and_hold_close_to_close"
CASH = "cash"


@dataclass
class BacktestResult:
    name: str
    execution_mode: str
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    position: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    trade_returns: list[float] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": self.position,
                "gross_return": self.gross_returns,
                "cost": self.costs,
                "net_return": self.returns,
                "equity": self.equity_curve,
                "drawdown": self.drawdown,
            }
        )


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------
def positions_from_probability(
    proba_up: pd.Series,
    threshold: float,
    allow_short: bool = False,
    short_threshold: float = 0.45,
) -> pd.Series:
    """Map probabilities to positions: long/cash by default, long/short optionally."""
    if allow_short:
        if short_threshold > threshold:
            raise ValueError(f"short_threshold ({short_threshold}) must be <= threshold ({threshold})")
        position = pd.Series(0.0, index=proba_up.index)
        position[proba_up >= threshold] = 1.0
        position[proba_up <= short_threshold] = -1.0
        return position
    return (proba_up >= threshold).astype(float)


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------
def compute_close_to_close_costs(position: pd.Series, one_way_cost_rate: float) -> pd.Series:
    """Cost proportional to how much the position changed.

    The first row is charged for opening from flat, so a strategy cannot get a
    free entry by starting invested.
    """
    if one_way_cost_rate < 0:
        raise ValueError("one_way_cost_rate must be >= 0")
    turnover = position.diff()
    turnover.iloc[0] = position.iloc[0]
    return turnover.abs() * one_way_cost_rate


def compute_intraday_round_trip_costs(position: pd.Series, round_trip_cost_rate: float) -> pd.Series:
    """Cost for every active session, because each one is a full round trip.

    Independent of the previous session's position: yesterday's trade was already
    closed at yesterday's close, so today's entry is a new one.
    """
    if round_trip_cost_rate < 0:
        raise ValueError("round_trip_cost_rate must be >= 0")
    return position.abs() * round_trip_cost_rate


def compute_costs(position: pd.Series, config: BacktestConfig) -> pd.Series:
    """Dispatch on execution mode. The mode changes the arithmetic, not a label."""
    if config.execution_mode == "next_open":
        return compute_intraday_round_trip_costs(position, config.round_trip_cost_rate)
    if config.execution_mode == "close_to_close":
        return compute_close_to_close_costs(position, config.one_way_cost_rate)
    raise ValueError(f"unknown execution_mode {config.execution_mode!r}")


# --------------------------------------------------------------------------
# trades
# --------------------------------------------------------------------------
def trade_returns_net(
    position: pd.Series,
    net_returns: pd.Series,
    execution_mode: ExecutionMode,
) -> list[float]:
    """Per-trade returns, **net of costs**.

    Reporting gross per-trade statistics next to net portfolio statistics is a
    common way to make a strategy look better than it is, so both come from the
    same net series here.

    In ``next_open`` mode each active session is one completed trade. In
    ``close_to_close`` mode a trade is a contiguous block of identical non-zero
    position, and its return is the compounded net return over that block.
    """
    if execution_mode == "next_open":
        return [float(r) for r in net_returns[position != 0]]

    trades: list[float] = []
    if (position == 0).all():
        return trades

    block_id = (position != position.shift()).cumsum()
    for _, block in net_returns.groupby(block_id):
        if position.loc[block.index].iloc[0] == 0:
            continue
        values = np.asarray(block, dtype=float)
        trades.append(float(np.prod(1.0 + values)) - 1.0)
    return trades


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def economic_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    costs: pd.Series,
    position: pd.Series,
    config: BacktestConfig,
    trade_returns: list[float],
) -> dict[str, object]:
    periods = config.trading_days_per_year
    n = len(net_returns)
    equity = (1.0 + net_returns).cumprod()

    cumulative = float(equity.iloc[-1] - 1.0) if n else float("nan")
    years = n / periods
    annualized = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if n and years > 0 else float("nan")

    volatility = float(net_returns.std(ddof=1) * np.sqrt(periods)) if n > 1 else float("nan")
    excess = net_returns - config.risk_free_rate_annual / periods
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))
        if n > 1 and excess.std(ddof=1) > 0
        else float("nan")
    )

    downside = excess[excess < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (
        float(excess.mean() / downside_std * np.sqrt(periods))
        if downside_std and downside_std > 0
        else float("nan")
    )

    turnover = position.diff()
    turnover.iloc[0] = position.iloc[0]
    turnover = turnover.abs()

    n_position_changes = int((turnover > 0).sum())
    n_active_sessions = int((position != 0).sum())
    n_completed_trades = len(trade_returns)

    wins = [t for t in trade_returns if t > 0]
    losses = [t for t in trade_returns if t < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif wins:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")

    active_net = net_returns[position != 0]

    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else float("nan"),
        "exposure": float((position != 0).mean()),
        "average_position": float(position.mean()),
        # Distinct counters: these three coincide only by accident.
        "n_position_changes": n_position_changes,
        "n_active_sessions": n_active_sessions,
        "n_completed_trades": n_completed_trades,
        # Every trade statistic below is net of costs.
        "average_trade_return": float(np.mean(trade_returns)) if trade_returns else float("nan"),
        "median_trade_return": float(np.median(trade_returns)) if trade_returns else float("nan"),
        "win_rate": float(len(wins) / len(trade_returns)) if trade_returns else float("nan"),
        "best_trade": float(max(trade_returns)) if trade_returns else float("nan"),
        "worst_trade": float(min(trade_returns)) if trade_returns else float("nan"),
        "profit_factor": profit_factor,
        "total_cost_paid": float(costs.sum()),
        "average_cost_per_active_session": (
            float(costs.sum() / n_active_sessions) if n_active_sessions else 0.0
        ),
        "gross_cumulative_return": float((1.0 + gross_returns).cumprod().iloc[-1] - 1.0)
        if n
        else float("nan"),
        "n_periods": n,
        "years": float(years),
    }


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------
def run_backtest(
    position: pd.Series,
    future_return: pd.Series,
    config: BacktestConfig,
    name: str = "strategy",
) -> BacktestResult:
    """Apply a position series to forward returns and score it under `execution_mode`."""
    if not position.index.equals(future_return.index):
        raise ValueError("position and future_return must share an identical index")
    if position.empty:
        raise ValueError("cannot backtest an empty position series")

    position = position.astype(float)
    future_return = future_return.astype(float)

    gross = position * future_return
    costs = compute_costs(position, config)
    net = gross - costs

    trades = trade_returns_net(position, net, config.execution_mode)
    equity = (1.0 + net).cumprod()

    metrics = economic_metrics(net, gross, costs, position, config, trades)
    metrics["name"] = name
    metrics["execution_mode"] = config.execution_mode

    return BacktestResult(
        name=name,
        execution_mode=config.execution_mode,
        returns=net,
        gross_returns=gross,
        costs=costs,
        position=position,
        equity_curve=equity,
        drawdown=drawdown_series(equity),
        trade_returns=trades,
        metrics=metrics,
    )


def always_active(future_return: pd.Series, config: BacktestConfig) -> BacktestResult:
    """Always in the market, under the configured execution model.

    Named ``always_long_intraday`` in next_open mode because it re-enters every
    session and pays a round trip each time — a materially worse deal than
    holding, and one that must not be labelled buy-and-hold.
    """
    position = pd.Series(1.0, index=future_return.index)
    name = ALWAYS_LONG_INTRADAY if config.execution_mode == "next_open" else BUY_AND_HOLD
    return run_backtest(position, future_return, config, name=name)


def buy_and_hold_close_to_close(
    close: pd.Series,
    config: BacktestConfig,
    index: pd.Index | None = None,
) -> BacktestResult:
    """Genuine buy-and-hold: hold the asset, pay one entry cost, use close-to-close returns.

    Always available as a comparison even in intraday mode, because "would I have
    done better just holding it?" is the question a reader actually has.
    """
    returns = close.pct_change().fillna(0.0)
    if index is not None:
        returns = returns.reindex(index).fillna(0.0)

    hold_config = config.model_copy(update={"execution_mode": "close_to_close"})
    position = pd.Series(1.0, index=returns.index)
    return run_backtest(position, returns, hold_config, name=BUY_AND_HOLD)


def cash(future_return: pd.Series, config: BacktestConfig) -> BacktestResult:
    position = pd.Series(0.0, index=future_return.index)
    return run_backtest(position, future_return, config, name=CASH)


PREFERRED_METRIC_ORDER: tuple[str, ...] = (
    "execution_mode",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "hit_rate",
    "exposure",
    "n_position_changes",
    "n_active_sessions",
    "n_completed_trades",
    "average_trade_return",
    "median_trade_return",
    "win_rate",
    "best_trade",
    "worst_trade",
    "profit_factor",
    "total_cost_paid",
    "average_cost_per_active_session",
    "gross_cumulative_return",
)


def backtest_comparison(results: list[BacktestResult]) -> pd.DataFrame:
    frame = pd.DataFrame({result.name: result.metrics for result in results}).T
    ordered = [c for c in PREFERRED_METRIC_ORDER if c in frame.columns]
    remaining = [c for c in frame.columns if c not in ordered and c != "name"]
    return frame[ordered + remaining]
