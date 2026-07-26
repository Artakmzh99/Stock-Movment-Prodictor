"""Label generation.

This is the only module allowed to look forward in time, and it does so
explicitly. The label at row `t` describes what happens *after* `t`; the last
`horizon` rows therefore have no label and are removed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import LabelConfig

LABEL_COLUMNS = ["future_return", "target"]


def future_return_close_to_close(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Close(t+h) / Close(t) - 1.

    The research target. Note that acting on it assumes you can trade at the same
    close you used to make the decision — see ``backtest.py`` for the caveat.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def future_return_open_to_close(open_: pd.Series, close: pd.Series, horizon: int = 1) -> pd.Series:
    """Close(t+h) / Open(t+1) - 1.

    The executable target: decide after the close of day t, enter at the next
    open, exit at the close of day t+h.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / open_.shift(-1) - 1.0


def make_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Build ``future_return`` and the binary ``target`` from an OHLCV frame."""
    close = df["Close"].astype(float)

    if config.target_definition == "close_to_close":
        future_return = future_return_close_to_close(close, config.horizon)
    elif config.target_definition == "open_to_close":
        future_return = future_return_open_to_close(df["Open"].astype(float), close, config.horizon)
    else:
        raise ValueError(
            f"unknown target_definition {config.target_definition!r}; "
            "expected 'close_to_close' or 'open_to_close'"
        )

    future_return = future_return.replace([np.inf, -np.inf], np.nan)

    labels = pd.DataFrame(index=df.index)
    labels["future_return"] = future_return
    # A flat day (return exactly 0) counts as "not up", matching target > 0.
    labels["target"] = (future_return > 0).astype("float")
    labels.loc[future_return.isna(), "target"] = np.nan
    return labels


def drop_unlabeled_tail(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Remove the trailing rows whose label cannot exist yet."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return frame.iloc[:-horizon] if horizon < len(frame) else frame.iloc[0:0]


def label_summary(labels: pd.DataFrame) -> dict[str, Any]:
    target = labels["target"].dropna()
    return {
        "n_labeled": len(target),
        "n_up": int((target == 1).sum()),
        "n_down": int((target == 0).sum()),
        "up_rate": float(target.mean()) if len(target) else float("nan"),
        "mean_future_return": float(labels["future_return"].mean()),
    }
