"""Inference: predict from a saved model, without retraining.

This is what makes the project a model rather than a report. It loads the exact
saved estimator, the exact resolved config, and the exact ordered feature list,
rebuilds features from the *same* code path used in training, and refuses to guess
when anything is missing.

Deliberate refusals — each of these would otherwise produce a confident number
with no meaning:

* the run has no saved model (predates persistence, or `final-test` never ran);
* the feature version recorded with the model differs from the current code;
* a feature named in the manifest cannot be built from current data;
* the requested signal date is not a completed trading session;
* any feature value is NaN or infinite.

The signal date is the session whose close supplies the features. Under
``next_open`` execution, the trade would be entered at the *next* session's open,
which is reported as ``expected_execution_date``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import open_run
from .config import FEATURE_VERSION, Config, config_from_dict
from .data import get_ohlcv
from .dataset import align_benchmark
from .features import build_features
from .market_calendar import last_completed_session, next_session
from .persistence import ModelMetadata, load_model, load_model_metadata
from .validation import DataValidationError, assert_finite


class InferenceError(RuntimeError):
    """Prediction cannot be made safely."""


@dataclass
class Prediction:
    signal_date: str
    expected_execution_date: str | None
    ticker: str
    probability_up: float
    predicted_class: int
    trading_signal: str
    classification_threshold: float
    trading_threshold: float
    model_run_id: str
    model_candidate: str
    feature_version: str
    target_definition: str
    execution_mode: str
    #: Development-only gate outcome.
    edge_detected: bool
    #: Whether the sealed final test confirmed it. None when unknown.
    final_test_confirmed_edge: bool | None
    generated_at_utc: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def load_run_config(run_path: Path) -> Config:
    """Reload the exact resolved config the model was trained under."""
    path = Path(run_path) / "resolved_config.json"
    if not path.exists():
        raise InferenceError(f"no resolved_config.json in {run_path}; cannot reproduce the training setup")
    return config_from_dict(json.loads(path.read_text()))


def load_feature_manifest(run_path: Path) -> dict[str, Any]:
    path = Path(run_path) / "feature_manifest.json"
    if not path.exists():
        raise InferenceError(f"no feature_manifest.json in {run_path}; feature order is unknown")
    manifest: dict[str, Any] = json.loads(path.read_text())
    if "feature_names" not in manifest:
        raise InferenceError("feature_manifest.json has no feature_names")
    return manifest


def _check_feature_version(metadata: ModelMetadata, manifest: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if metadata.feature_version != FEATURE_VERSION:
        raise InferenceError(
            f"model was trained with feature_version {metadata.feature_version!r} but this "
            f"code produces {FEATURE_VERSION!r}. Feature semantics have changed; the saved "
            "model is not valid against current features. Retrain."
        )
    manifest_version = manifest.get("feature_version")
    if manifest_version and manifest_version != metadata.feature_version:
        warnings.append(
            f"manifest feature_version {manifest_version!r} differs from model {metadata.feature_version!r}"
        )
    if list(manifest["feature_names"]) != list(metadata.feature_names):
        raise InferenceError(
            "feature_manifest.json and model_metadata.json disagree on the feature list; "
            "the run's artifacts are inconsistent"
        )
    return warnings


def _resolve_signal_date(
    available: pd.DatetimeIndex,
    as_of: date | None,
    exchange: str,
    now_utc: datetime | None,
) -> pd.Timestamp:
    if as_of is None:
        target = last_completed_session(exchange=exchange, now_utc=now_utc)
        candidates = available[available <= target]
        if len(candidates) == 0:
            raise InferenceError(
                f"no feature rows available at or before the last completed session ({target.date()})"
            )
        return pd.Timestamp(candidates[-1])

    requested = pd.Timestamp(as_of).normalize()
    if requested not in available:
        earlier = available[available <= requested]
        nearest = f" nearest earlier available session: {earlier[-1].date()}" if len(earlier) else ""
        raise InferenceError(
            f"{requested.date()} is not a completed session with a full feature row "
            f"(it may be a holiday, a weekend, or still forming).{nearest}"
        )
    return requested


def build_inference_features(
    config: Config,
    manifest: dict[str, Any],
    force_refresh: bool = False,
    now_utc: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild features through the latest completed session, using training logic.

    Note there is no label here and therefore no ``drop_unlabeled_tail``: the most
    recent session is exactly the row we want to predict *from*.
    """
    bundle = get_ohlcv(config, force_refresh=force_refresh, now_utc=now_utc)
    prices = bundle.prices

    benchmark_close = None
    if config.features.use_benchmark_features:
        assert config.data.benchmark_ticker is not None
        bench = get_ohlcv(
            config,
            ticker=config.data.benchmark_ticker,
            force_refresh=force_refresh,
            now_utc=now_utc,
        )
        asset_close, benchmark_close, _ = align_benchmark(
            prices["Close"].astype(float),
            bench.prices["Close"].astype(float),
            config.data.benchmark_ticker,
            max_join_loss=config.features.max_benchmark_join_loss,
        )
        prices = prices.loc[asset_close.index]

    features = build_features(prices, config.features, benchmark_close=benchmark_close)

    expected = list(manifest["feature_names"])
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise InferenceError(
            f"the current feature code cannot produce {len(missing)} feature(s) the model "
            f"requires: {missing[:10]}. The model and the code are out of step; retrain."
        )

    # Reindex to the manifest's exact order. A permuted column order would not
    # raise anywhere downstream — it would just produce wrong answers.
    features = features[expected]
    return features.dropna(), bundle.metadata


def predict(
    run_id: str,
    config: Config,
    as_of: date | None = None,
    force_refresh: bool = False,
    now_utc: datetime | None = None,
) -> Prediction:
    """Produce one prediction from a saved run. No training happens here."""
    from .provenance import utc_now_iso

    run = open_run(config, run_id)
    metadata = load_model_metadata(run.path)
    run_config = load_run_config(run.path)
    manifest = load_feature_manifest(run.path)

    warnings = _check_feature_version(metadata, manifest)

    if metadata.ticker != run_config.data.ticker:  # pragma: no cover - defensive
        warnings.append(f"model ticker {metadata.ticker!r} differs from config {run_config.data.ticker!r}")

    # Predicting the present needs live data, so the run's frozen end_date is lifted.
    inference_config = config_from_dict(
        {**run_config.to_dict(), "data": {**run_config.to_dict()["data"], "end_date": None}}
    )

    features, _ = build_inference_features(
        inference_config, manifest, force_refresh=force_refresh, now_utc=now_utc
    )
    if features.empty:
        raise InferenceError("no complete feature rows are available")

    index = features.index
    assert isinstance(index, pd.DatetimeIndex)
    signal_date = _resolve_signal_date(index, as_of, inference_config.data.exchange, now_utc)

    row = features.loc[[signal_date]]
    try:
        assert_finite(row, context=f"feature row for {signal_date.date()}")
    except DataValidationError as exc:
        raise InferenceError(str(exc)) from exc

    model = load_model(run.path)
    proba = np.asarray(model.predict_proba(row))
    probability_up = float(proba[:, 1][0] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()[0])

    classification_threshold = metadata.classification_threshold
    trading_threshold = metadata.trading_threshold

    execution_date: str | None = None
    if metadata.execution_mode == "next_open":
        following = next_session(signal_date, exchange=inference_config.data.exchange)
        execution_date = str(following.date()) if following is not None else None
    else:
        warnings.append(
            "close_to_close execution is a research convention and is not literally "
            "tradable, so no execution date is reported"
        )

    if not metadata.edge_detected:
        warnings.append(
            "this model did not pass the predictive-edge gate during selection; the "
            "signal below has no demonstrated out-of-sample edge and must not be traded"
        )
    elif metadata.final_test_beat_baselines is False:
        # A development edge the holdout contradicted is the normal outcome, and by
        # far the most misleading thing this artifact could stay quiet about.
        warnings.append(
            "the development-only edge gate passed, but the FINAL TEST CONTRADICTED it: "
            f"balanced accuracy {metadata.final_test_balanced_accuracy:.4f} versus the best "
            f"baseline's {metadata.final_test_best_baseline_balanced_accuracy:.4f}. "
            "The apparent edge did not survive out of sample. Do not trade this signal."
        )

    return Prediction(
        signal_date=str(signal_date.date()),
        expected_execution_date=execution_date,
        ticker=metadata.ticker,
        probability_up=round(probability_up, 6),
        predicted_class=int(probability_up >= classification_threshold),
        trading_signal="long" if probability_up >= trading_threshold else "cash",
        classification_threshold=classification_threshold,
        trading_threshold=trading_threshold,
        model_run_id=run_id,
        model_candidate=metadata.candidate_name,
        feature_version=metadata.feature_version,
        target_definition=metadata.target_definition,
        execution_mode=metadata.execution_mode,
        edge_detected=metadata.edge_detected,
        final_test_confirmed_edge=metadata.final_test_beat_baselines,
        generated_at_utc=utc_now_iso(),
        warnings=warnings,
    )
