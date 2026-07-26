"""Shared fixtures.

Every test runs on synthetic data with a mocked clock. Nothing touches the
network, so the guarantees under test hold deterministically rather than
depending on what the market happened to do.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_movement.config import Config, config_from_dict

#: A Thursday evening after the NYSE close, used as the default "now".
FIXED_NOW = datetime(2026, 3, 5, 22, 0, tzinfo=UTC)


def make_ohlcv(n: int = 1400, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    """A plausible OHLCV frame on NYSE sessions, with a coherent bar shape.

    High/Low genuinely bracket Open and Close, so the frame passes the OHLC
    consistency checks that real data must also pass.
    """
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6) + 60)
    )
    index = pd.DatetimeIndex(sessions[:n]).tz_localize(None)
    index.name = "Date"

    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0004, scale=0.012, size=len(index))
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, size=len(index)))

    spread = np.abs(rng.normal(0.008, 0.004, size=len(index))) * close
    high = np.maximum(close, open_) + spread / 2
    low = np.minimum(close, open_) - spread / 2
    volume = rng.integers(1_000_000, 50_000_000, size=len(index)).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv(n=1400)


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    return make_ohlcv(n=400, seed=3)


def small_config_payload(**overrides: Any) -> dict[str, Any]:
    """A config small and fast enough for unit tests, still fully valid."""
    payload: dict[str, Any] = {
        "random_seed": 42,
        "data": {"ticker": "TEST", "min_rows": 200, "start_date": "2015-01-02"},
        "features": {
            "return_windows": [1, 2, 5],
            "momentum_windows": [5, 10],
            "sma_windows": [5, 10, 20],
            "volatility_windows": [5, 10],
            "positive_day_windows": [5],
            "volume_window": 10,
        },
        "split": {"walk_forward_splits": 3, "gap": 1},
        "selection": {"families": ["logistic"], "min_fold_wins": 2},
        "statistics": {"bootstrap_samples": 50},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


@pytest.fixture
def config() -> Config:
    return config_from_dict(small_config_payload())


@pytest.fixture
def offline_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A full config wired to synthetic data, a temp output dir and a fixed clock."""
    frame = make_ohlcv(n=1400, seed=11)

    monkeypatch.setattr(
        "stock_movement.data.download_ohlcv",
        lambda ticker, start, end, interval="1d", auto_adjust=True: frame.copy(),
    )

    return config_from_dict(
        small_config_payload(
            run_name="test-run",
            data={"ticker": "TEST", "min_rows": 200, "end_date": "2026-01-02"},
            paths={
                "data_raw": str(tmp_path / "raw"),
                "data_processed": str(tmp_path / "processed"),
                "runs": str(tmp_path / "runs"),
            },
        )
    )


@pytest.fixture
def dataset(offline_config: Config) -> Any:
    from stock_movement.dataset import build_dataset

    return build_dataset(offline_config)
