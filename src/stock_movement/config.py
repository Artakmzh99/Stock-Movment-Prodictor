"""Configuration schema, loading, and cross-field validation.

Pydantic models with ``extra="forbid"``, so a typo is an error rather than a
silently ignored key. Every model is frozen: once a config is loaded its hash
identifies it for the rest of the run, and a mutable config would make that hash
a lie. CLI overrides are applied to the raw payload *before* construction.

The important work here is **cross-field** validation. Individually valid fields
can still describe an incoherent experiment — the clearest example being a
`close_to_close` label paired with `next_open` execution, which silently measures
a return the strategy never earns. Those combinations fail at load time.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Bumped when the artifact layout changes in a way readers must know about.
ARTIFACT_SCHEMA_VERSION = 2

#: Feature-logic version. Changing feature semantics must bump this, because a
#: saved model is only valid against the feature code it was trained on.
FEATURE_VERSION = "v2"

TargetDefinition = Literal["open_to_close", "close_to_close"]
ExecutionMode = Literal["next_open", "close_to_close"]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(ge=1)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
class DataConfig(_Base):
    ticker: str = "AAPL"
    benchmark_ticker: str | None = "SPY"
    start_date: date = date(2010, 1, 1)
    end_date: date | None = None
    interval: str = "1d"
    auto_adjust: bool = True
    #: Drop the final bar only when it is a still-open session (see market_calendar).
    drop_last_incomplete: bool = True
    #: Rows required *after* feature warm-up and label construction.
    min_rows: PositiveInt = 1000
    exchange: str = "XNYS"
    #: Identical duplicate rows are deduplicated; conflicting ones always fail.
    deduplicate_identical_rows: bool = True

    @model_validator(mode="after")
    def _check_dates(self) -> DataConfig:
        if self.end_date is not None and self.start_date >= self.end_date:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        return self


class FeatureConfig(_Base):
    return_windows: tuple[PositiveInt, ...] = (1, 2, 3, 5, 10, 20)
    #: Independent of volatility_windows — coupling them was a latent bug.
    momentum_windows: tuple[PositiveInt, ...] = (5, 10, 20)
    sma_windows: tuple[PositiveInt, ...] = (5, 10, 20, 50)
    volatility_windows: tuple[PositiveInt, ...] = (5, 10, 20)
    positive_day_windows: tuple[PositiveInt, ...] = (5, 10)
    volume_window: PositiveInt = 20
    use_benchmark_features: bool = False
    benchmark_windows: tuple[PositiveInt, ...] = (1, 5)
    benchmark_rolling_window: PositiveInt = 20
    #: Fraction of target dates that may be lost to the benchmark inner join.
    max_benchmark_join_loss: Probability = 0.02

    @property
    def max_window(self) -> int:
        """Longest look-back any feature needs; drives the warm-up requirement."""
        windows = (
            self.return_windows
            + self.momentum_windows
            + self.sma_windows
            + self.volatility_windows
            + self.positive_day_windows
            + (self.volume_window,)
        )
        if self.use_benchmark_features:
            windows = windows + self.benchmark_windows + (self.benchmark_rolling_window,)
        return max(windows)


class LabelConfig(_Base):
    horizon: PositiveInt = 1
    target_definition: TargetDefinition = "open_to_close"

    @model_validator(mode="after")
    def _check_horizon(self) -> LabelConfig:
        if self.target_definition == "open_to_close" and self.horizon != 1:
            raise ValueError(
                "open_to_close is a single-session target; horizon must be 1 "
                f"(got {self.horizon}). Use close_to_close for multi-day horizons."
            )
        return self


class SplitConfig(_Base):
    train_fraction: Probability = 0.60
    validation_fraction: Probability = 0.20
    test_fraction: Probability = 0.20
    walk_forward_splits: Annotated[int, Field(ge=2)] = 5
    gap: Annotated[int, Field(ge=0)] = 1

    @model_validator(mode="after")
    def _check_fractions(self) -> SplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        if self.test_fraction <= 0:
            raise ValueError("test_fraction must be > 0; the final holdout cannot be empty")
        return self

    @property
    def development_fraction(self) -> float:
        return self.train_fraction + self.validation_fraction


class ThresholdConfig(_Base):
    """Two distinct thresholds, deliberately separated.

    ``classification_value`` turns a probability into a predicted label for
    metrics. ``trading_value`` decides whether to take a position. They answer
    different questions and conflating them means a trading decision silently
    changes a reported accuracy.
    """

    classification_value: Probability = 0.50
    trading_value: Probability = 0.55
    tune_classification_threshold: bool = False
    tune_trading_threshold: bool = False
    candidates: tuple[Probability, ...] = tuple(round(0.30 + 0.01 * i, 2) for i in range(41))
    objective: Literal["balanced_accuracy", "f1_macro", "mcc", "net_sharpe"] = "balanced_accuracy"


class BacktestConfig(_Base):
    execution_mode: ExecutionMode = "next_open"
    #: next_open: charged on every active session (enter at open, exit at close).
    round_trip_cost_bps: NonNegativeFloat = 10.0
    #: close_to_close: charged per unit of position *change*.
    one_way_cost_bps: NonNegativeFloat = 5.0
    allow_short: bool = False
    short_threshold: Probability = 0.45
    trading_days_per_year: PositiveInt = 252
    risk_free_rate_annual: float = 0.0

    @property
    def round_trip_cost_rate(self) -> float:
        return self.round_trip_cost_bps / 10_000.0

    @property
    def one_way_cost_rate(self) -> float:
        return self.one_way_cost_bps / 10_000.0


class SelectionConfig(_Base):
    """Development-only candidate selection and the predictive-edge gate."""

    #: Candidate families to evaluate. LSTM is skipped when TensorFlow is absent.
    families: tuple[str, ...] = ("logistic", "hist_gradient_boosting")
    include_lstm: bool = False
    #: Minimum balanced-accuracy margin over the best baseline.
    min_balanced_accuracy_margin: NonNegativeFloat = 0.01
    #: Folds (out of walk_forward_splits) the winner must win outright.
    min_fold_wins: PositiveInt = 4
    min_roc_auc: Probability = 0.50
    min_mcc: float = 0.0
    #: The random baseline is a diagnostic, not a serious opponent.
    include_random_baseline: bool = True


class StatisticsConfig(_Base):
    confidence_level: Probability = 0.95
    bootstrap_samples: PositiveInt = 2000
    #: Blocks preserve autocorrelation that an i.i.d. bootstrap would destroy.
    bootstrap_block_length: PositiveInt = 20
    bootstrap_seed: int = 42


class PathsConfig(_Base):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    runs: str = "artifacts/runs"


class Config(_Base):
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig()
    split: SplitConfig = SplitConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    backtest: BacktestConfig = BacktestConfig()
    selection: SelectionConfig = SelectionConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    paths: PathsConfig = PathsConfig()
    random_seed: int = 42
    run_name: str | None = None

    # -- cross-section validation ----------------------------------------
    @model_validator(mode="after")
    def _check_coherence(self) -> Config:
        # The label must describe the return the execution actually earns.
        expected_execution: dict[str, str] = {
            "open_to_close": "next_open",
            "close_to_close": "close_to_close",
        }
        required = expected_execution[self.labels.target_definition]
        if self.backtest.execution_mode != required:
            raise ValueError(
                f"incoherent experiment: labels.target_definition="
                f"{self.labels.target_definition!r} measures a return that "
                f"backtest.execution_mode={self.backtest.execution_mode!r} does not earn. "
                f"Use execution_mode={required!r}, or change the target definition."
            )

        if self.split.gap < self.labels.horizon:
            raise ValueError(
                f"split.gap ({self.split.gap}) must be >= labels.horizon "
                f"({self.labels.horizon}), otherwise the label of the last training row "
                "overlaps the first evaluation row"
            )

        if self.backtest.allow_short and self.backtest.short_threshold > self.threshold.trading_value:
            raise ValueError(
                f"backtest.short_threshold ({self.backtest.short_threshold}) must be <= "
                f"threshold.trading_value ({self.threshold.trading_value})"
            )

        warm_up = self.features.max_window + self.labels.horizon + 1
        if self.data.min_rows <= warm_up:
            raise ValueError(
                f"data.min_rows ({self.data.min_rows}) does not exceed the feature warm-up "
                f"requirement of {warm_up} rows (longest window "
                f"{self.features.max_window} + horizon {self.labels.horizon} + 1)"
            )

        if self.selection.min_fold_wins > self.split.walk_forward_splits:
            raise ValueError(
                f"selection.min_fold_wins ({self.selection.min_fold_wins}) exceeds "
                f"split.walk_forward_splits ({self.split.walk_forward_splits})"
            )

        if self.features.use_benchmark_features and not self.data.benchmark_ticker:
            raise ValueError("features.use_benchmark_features is true but data.benchmark_ticker is unset")

        return self

    # -- paths ------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Relative paths hang off the project root; absolute paths are honoured."""
        configured = Path(str(getattr(self.paths, key)))
        return configured if configured.is_absolute() else PROJECT_ROOT / configured

    # -- serialisation / identity -----------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        """Key-order-independent JSON. Two configs meaning the same thing hash the same."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def short_hash(self, length: int = 8) -> str:
        return self.sha256()[:length]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
#: Paths replaced wholesale rather than merged when a child config extends a
#: parent. Hyperparameters only make sense as a set: merging a logistic `C` into
#: a gradient-boosting block yields a config that cannot be instantiated.
_REPLACE_NOT_MERGE: frozenset[tuple[str, ...]] = frozenset(
    {("selection", "families"), ("threshold", "candidates")}
)


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any], _path: tuple[str, ...] = ()
) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        path = _path + (key,)
        if path in _REPLACE_NOT_MERGE:
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value, path)
        else:
            out[key] = value
    return out


def config_from_dict(payload: dict[str, Any]) -> Config:
    return Config.model_validate(payload or {})


def resolve_config_payload(path: str | Path) -> dict[str, Any]:
    """Read YAML and resolve a single-level ``extends:`` into one flat payload."""
    resolved = Path(path)
    if not resolved.is_absolute():
        candidate = PROJECT_ROOT / resolved
        resolved = candidate if candidate.exists() else resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"config not found: {resolved}")

    with open(resolved) as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    parent = payload.pop("extends", None)
    if parent:
        base_path = (resolved.parent / parent).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"config {resolved.name} extends missing file: {base_path}")
        with open(base_path) as handle:
            base_payload: dict[str, Any] = yaml.safe_load(handle) or {}
        base_payload.pop("extends", None)
        payload = _deep_merge(base_payload, payload)

    return payload


def apply_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI overrides to a raw payload before validation.

    Overrides use dotted paths (``data.ticker``). Applying them pre-construction
    keeps ``Config`` frozen, so its hash cannot drift after the fact.
    """
    result = dict(payload)
    for dotted, value in overrides.items():
        if value is None:
            continue
        keys = dotted.split(".")
        cursor = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            cursor[key] = dict(existing) if isinstance(existing, dict) else {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    payload = resolve_config_payload(path)
    if overrides:
        payload = apply_overrides(payload, overrides)
    return config_from_dict(payload)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional dependency, present only for the LSTM phase
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - tensorflow is optional
        pass
