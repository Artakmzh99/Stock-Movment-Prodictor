"""Config schema, cross-field validation, and hashing."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from stock_movement.config import (
    Config,
    apply_overrides,
    config_from_dict,
    load_config,
)
from tests.conftest import small_config_payload


def _write(tmp_path: Any, name: str, payload: dict[str, Any]) -> Any:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


# --------------------------------------------------------------------------
# P0.8 defaults
# --------------------------------------------------------------------------
def test_defaults_are_the_executable_setup():
    config = Config()

    assert config.labels.target_definition == "open_to_close"
    assert config.backtest.execution_mode == "next_open"
    assert config.backtest.round_trip_cost_bps == 10.0
    assert config.threshold.classification_value == 0.50
    assert config.threshold.trading_value == 0.55
    assert config.data.min_rows == 1000
    assert config.split.gap >= config.labels.horizon
    assert config.random_seed == 42


def test_shipped_default_config_matches_the_specified_defaults():
    config = load_config("configs/default.yaml")
    assert config.labels.target_definition == "open_to_close"
    assert config.backtest.execution_mode == "next_open"
    assert config.data.min_rows == 1000
    assert config.features.momentum_windows == (5, 10, 20)


# --------------------------------------------------------------------------
# P0.7 cross-field validation
# --------------------------------------------------------------------------
def test_next_open_requires_open_to_close_target():
    """The label must describe the return the execution actually earns."""
    with pytest.raises(ValidationError, match="incoherent experiment"):
        config_from_dict(
            {
                "labels": {"target_definition": "close_to_close"},
                "backtest": {"execution_mode": "next_open"},
            }
        )


def test_open_to_close_requires_next_open_execution():
    with pytest.raises(ValidationError, match="incoherent experiment"):
        config_from_dict(
            {
                "labels": {"target_definition": "open_to_close"},
                "backtest": {"execution_mode": "close_to_close"},
            }
        )


def test_coherent_close_to_close_pair_is_accepted():
    config = config_from_dict(
        {
            "labels": {"target_definition": "close_to_close"},
            "backtest": {"execution_mode": "close_to_close"},
        }
    )
    assert config.labels.target_definition == "close_to_close"


def test_gap_smaller_than_horizon_rejected():
    with pytest.raises(ValidationError, match=r"must be >= labels\.horizon"):
        config_from_dict(
            {
                "labels": {"horizon": 3, "target_definition": "close_to_close"},
                "backtest": {"execution_mode": "close_to_close"},
                "split": {"gap": 1},
            }
        )


def test_negative_cost_rejected():
    with pytest.raises(ValidationError):
        config_from_dict({"backtest": {"round_trip_cost_bps": -1.0}})
    with pytest.raises(ValidationError):
        config_from_dict({"backtest": {"one_way_cost_bps": -0.5}})


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValidationError):
        config_from_dict({"threshold": {"trading_value": value}})


def test_invalid_date_range_rejected():
    with pytest.raises(ValidationError, match="must be before"):
        config_from_dict({"data": {"start_date": "2020-01-01", "end_date": "2019-01-01"}})


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
        config_from_dict({"split": {"train_fraction": 0.7, "validation_fraction": 0.2, "test_fraction": 0.2}})


def test_empty_test_fraction_rejected():
    with pytest.raises(ValidationError, match=r"test_fraction must be > 0"):
        config_from_dict({"split": {"train_fraction": 0.8, "validation_fraction": 0.2, "test_fraction": 0.0}})


def test_walk_forward_splits_below_two_rejected():
    with pytest.raises(ValidationError):
        config_from_dict({"split": {"walk_forward_splits": 1}})


def test_min_rows_must_exceed_the_feature_warm_up():
    """A config that cannot possibly produce enough rows should fail at load."""
    with pytest.raises(ValidationError, match="feature warm-up"):
        config_from_dict({"data": {"min_rows": 30}, "features": {"sma_windows": [5, 200]}})


def test_short_threshold_above_trading_threshold_rejected():
    with pytest.raises(ValidationError, match="short_threshold"):
        config_from_dict(
            {
                "threshold": {"trading_value": 0.55},
                "backtest": {"allow_short": True, "short_threshold": 0.70},
            }
        )


def test_min_fold_wins_cannot_exceed_the_number_of_folds():
    with pytest.raises(ValidationError, match="min_fold_wins"):
        config_from_dict({"split": {"walk_forward_splits": 3}, "selection": {"min_fold_wins": 5}})


def test_benchmark_features_require_a_benchmark_ticker():
    with pytest.raises(ValidationError, match="benchmark_ticker is unset"):
        config_from_dict({"features": {"use_benchmark_features": True}, "data": {"benchmark_ticker": None}})


def test_open_to_close_rejects_a_multi_day_horizon():
    with pytest.raises(ValidationError, match="single-session target"):
        config_from_dict({"labels": {"target_definition": "open_to_close", "horizon": 3}})


# --------------------------------------------------------------------------
# unknown keys and immutability
# --------------------------------------------------------------------------
def test_unknown_keys_are_rejected():
    """A typo must fail loudly, not silently do nothing."""
    with pytest.raises(ValidationError):
        config_from_dict({"data": {"tickr": "AAPL"}})
    with pytest.raises(ValidationError):
        config_from_dict({"randomseed": 1})


def test_config_is_frozen():
    """Mutating a config after hashing would make its recorded hash a lie."""
    config = Config()
    with pytest.raises(ValidationError):
        config.random_seed = 7  # type: ignore[misc]


# --------------------------------------------------------------------------
# P1.3 / P0.6 hashing
# --------------------------------------------------------------------------
def test_config_hash_ignores_yaml_key_order(tmp_path):
    first = _write(tmp_path, "a.yaml", {"random_seed": 7, "data": {"ticker": "AAPL", "min_rows": 1000}})
    second = _write(tmp_path, "b.yaml", {"data": {"min_rows": 1000, "ticker": "AAPL"}, "random_seed": 7})

    assert load_config(first).sha256() == load_config(second).sha256()


def test_same_resolved_config_has_same_hash():
    assert (
        config_from_dict(small_config_payload()).sha256() == config_from_dict(small_config_payload()).sha256()
    )


def test_different_configs_have_different_hashes():
    base = config_from_dict(small_config_payload())
    changed = config_from_dict(small_config_payload(random_seed=43))
    assert base.sha256() != changed.sha256()

    ticker_changed = config_from_dict(small_config_payload(data={"ticker": "MSFT"}))
    assert base.sha256() != ticker_changed.sha256()


def test_short_hash_is_a_prefix_of_the_full_hash():
    config = Config()
    assert config.sha256().startswith(config.short_hash())
    assert len(config.short_hash()) == 8


def test_cost_rates_convert_from_basis_points():
    config = config_from_dict({"backtest": {"round_trip_cost_bps": 10.0, "one_way_cost_bps": 5.0}})
    assert config.backtest.round_trip_cost_rate == pytest.approx(0.001)
    assert config.backtest.one_way_cost_rate == pytest.approx(0.0005)


# --------------------------------------------------------------------------
# extends / overrides
# --------------------------------------------------------------------------
def test_extends_inherits_and_overrides(tmp_path):
    _write(tmp_path, "base.yaml", {"random_seed": 1, "data": {"ticker": "AAPL", "min_rows": 1200}})
    child = _write(tmp_path, "child.yaml", {"extends": "base.yaml", "data": {"ticker": "MSFT"}})

    config = load_config(child)

    assert config.data.ticker == "MSFT"  # overridden
    assert config.data.min_rows == 1200  # inherited
    assert config.random_seed == 1  # inherited


def test_extends_missing_parent_fails_clearly(tmp_path):
    child = _write(tmp_path, "child.yaml", {"extends": "nope.yaml"})
    with pytest.raises(FileNotFoundError, match="extends missing file"):
        load_config(child)


def test_selection_families_are_replaced_not_merged(tmp_path):
    """A list-valued field must not accumulate across inheritance."""
    _write(tmp_path, "base.yaml", {"selection": {"families": ["logistic", "hist_gradient_boosting"]}})
    child = _write(tmp_path, "child.yaml", {"extends": "base.yaml", "selection": {"families": ["logistic"]}})

    assert load_config(child).selection.families == ("logistic",)


def test_apply_overrides_uses_dotted_paths():
    payload = apply_overrides({"data": {"ticker": "AAPL", "min_rows": 1000}}, {"data.ticker": "NVDA"})
    assert payload["data"]["ticker"] == "NVDA"
    assert payload["data"]["min_rows"] == 1000


def test_apply_overrides_ignores_none():
    payload = apply_overrides({"data": {"ticker": "AAPL"}}, {"data.ticker": None})
    assert payload["data"]["ticker"] == "AAPL"


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


# --------------------------------------------------------------------------
# shipped configs
# --------------------------------------------------------------------------
def test_every_shipped_config_loads_and_is_coherent():
    from pathlib import Path

    from stock_movement.config import PROJECT_ROOT
    from stock_movement.models import build_model

    configs = sorted(Path(PROJECT_ROOT / "configs").rglob("*.yaml"))
    assert configs, "no configs found"

    for path in configs:
        config = load_config(path)
        assert config.split.gap >= config.labels.horizon, path.name
        for family in config.selection.families:
            build_model(family, {}, random_state=config.random_seed)


def test_reproduction_config_pins_an_explicit_end_date():
    """A frozen config with end_date: null cannot reproduce a published table."""
    config = load_config("configs/reproduction/readme_aapl_2026_07.yaml")
    assert config.data.end_date is not None
    assert isinstance(config.data.end_date, date)


def test_research_config_is_close_to_close_and_labelled():
    from pathlib import Path

    from stock_movement.config import PROJECT_ROOT

    path = Path(PROJECT_ROOT / "configs/experiments/research_close_to_close.yaml")
    config = load_config(path)

    assert config.labels.target_definition == "close_to_close"
    assert config.backtest.execution_mode == "close_to_close"
    text = path.read_text().upper()
    assert "RESEARCH ONLY" in text or "NOT EXECUTABLE" in text


def test_max_window_reflects_every_configured_window():
    config = config_from_dict({"features": {"sma_windows": [5, 50], "return_windows": [1, 120]}})
    assert config.features.max_window == 120
