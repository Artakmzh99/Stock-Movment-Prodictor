"""Immutable run directories (P0.6) and the final-test lock (P0.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from stock_movement.artifacts import (
    FinalTestAlreadyCompletedError,
    RunAlreadyExistsError,
    create_run,
    list_runs,
    make_run_id,
    open_run,
)
from stock_movement.config import config_from_dict
from tests.conftest import small_config_payload


@pytest.fixture
def run_config(tmp_path):
    return config_from_dict(small_config_payload(paths={"runs": str(tmp_path / "runs")}))


def _lock_args(config):
    return {
        "config": config,
        "test_start": "2025-01-02",
        "test_end": "2026-01-02",
        "n_test_rows": 250,
        "selected_candidate": "logistic[C=1.0]",
        "data_sha256": "abc123",
    }


# --------------------------------------------------------------------------
# P0.6 run identity and immutability
# --------------------------------------------------------------------------
def test_run_id_contains_config_hash(run_config):
    run_id = make_run_id(run_config, now=datetime(2026, 7, 26, 18, 45, 1, tzinfo=UTC))

    assert run_id.startswith("20260726T184501Z_")
    assert run_id.endswith(run_config.short_hash())
    assert len(run_id.split("_")[1]) == 8


def test_same_resolved_config_has_same_hash_component(run_config):
    now = datetime(2026, 7, 26, 18, 45, 1, tzinfo=UTC)
    duplicate = config_from_dict(run_config.to_dict())

    assert make_run_id(run_config, now=now) == make_run_id(duplicate, now=now)


def test_different_configs_have_different_hashes(run_config, tmp_path):
    now = datetime(2026, 7, 26, 18, 45, 1, tzinfo=UTC)
    changed = config_from_dict(small_config_payload(random_seed=7, paths={"runs": str(tmp_path / "runs")}))

    assert make_run_id(run_config, now=now) != make_run_id(changed, now=now)


def test_existing_run_is_never_overwritten(run_config):
    run = create_run(run_config, run_id="fixed-id")
    run.write_text("marker.txt", "original")

    with pytest.raises(RunAlreadyExistsError, match="already exists"):
        create_run(run_config, run_id="fixed-id")

    # The original content survived the failed attempt.
    assert (run.path / "marker.txt").read_text() == "original"


def test_open_run_can_reopen_an_existing_run(run_config):
    created = create_run(run_config, run_id="fixed-id")
    created.write_json("payload.json", {"value": 1})

    reopened = open_run(run_config, "fixed-id")

    assert reopened.path == created.path
    assert reopened.read_json("payload.json") == {"value": 1}


def test_opening_a_missing_run_lists_what_exists(run_config):
    create_run(run_config, run_id="run-a")

    with pytest.raises(FileNotFoundError, match="run-a"):
        open_run(run_config, "run-does-not-exist")


def test_list_runs_reports_created_runs(run_config):
    assert list_runs(run_config) == []
    create_run(run_config, run_id="run-a")
    create_run(run_config, run_id="run-b")
    assert list_runs(run_config) == ["run-a", "run-b"]


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def test_run_directory_writes_every_artifact_type(run_config):
    import pandas as pd

    run = create_run(run_config, run_id="writers")
    frame = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})

    run.write_json("x.json", {"k": 1})
    run.write_yaml("x.yaml", {"k": 1})
    run.write_csv("x.csv", frame)
    run.write_parquet("x.parquet", frame)
    run.write_text("x.txt", "hello")

    for name in ("x.json", "x.yaml", "x.csv", "x.parquet", "x.txt"):
        assert run.exists(name)
    assert run.read_json("x.json") == {"k": 1}


def test_json_encoder_handles_numpy_and_timestamps(run_config):
    import numpy as np
    import pandas as pd

    run = create_run(run_config, run_id="encoder")
    run.write_json(
        "x.json",
        {
            "int": np.int64(3),
            "float": np.float64(1.5),
            "nan": np.float64("nan"),
            "bool": np.bool_(True),
            "array": np.array([1, 2]),
            "stamp": pd.Timestamp("2026-01-01"),
        },
    )

    payload = run.read_json("x.json")
    assert payload["int"] == 3
    assert payload["float"] == 1.5
    assert payload["nan"] == "nan"  # non-finite values are stringified, not dropped
    assert payload["array"] == [1, 2]


def test_reading_a_missing_artifact_raises(run_config):
    run = create_run(run_config, run_id="missing")
    with pytest.raises(FileNotFoundError):
        run.read_json("nope.json")


# --------------------------------------------------------------------------
# P0.3 final-test lock
# --------------------------------------------------------------------------
def test_final_test_writes_lock(run_config):
    run = create_run(run_config, run_id="locked")

    lock = run.write_lock(**_lock_args(run_config))

    assert run.lock_path.exists()
    assert lock.completed is True
    assert lock.rerun_count == 0
    assert lock.config_sha256 == run_config.sha256()
    assert lock.test_start == "2025-01-02"
    assert lock.n_test_rows == 250


def test_a_fresh_run_has_no_lock(run_config):
    run = create_run(run_config, run_id="fresh")
    assert run.read_lock() is None
    assert run.assert_final_test_allowed() is None


def test_final_test_refuses_second_execution(run_config):
    run = create_run(run_config, run_id="second")
    run.write_lock(**_lock_args(run_config))

    with pytest.raises(FinalTestAlreadyCompletedError, match="already completed"):
        run.assert_final_test_allowed()


def test_rerun_requires_explicit_flag(run_config):
    run = create_run(run_config, run_id="flag")
    run.write_lock(**_lock_args(run_config))

    with pytest.raises(FinalTestAlreadyCompletedError):
        run.assert_final_test_allowed(allow_rerun=False, rerun_reason="a good reason")

    assert run.assert_final_test_allowed(allow_rerun=True, rerun_reason="a good reason") is not None


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_rerun_requires_nonempty_reason(run_config, reason):
    run = create_run(run_config, run_id="reason")
    run.write_lock(**_lock_args(run_config))

    with pytest.raises(ValueError, match="non-empty explanation"):
        run.assert_final_test_allowed(allow_rerun=True, rerun_reason=reason)


def test_reruns_are_recorded_in_the_lock_history(run_config):
    run = create_run(run_config, run_id="history")
    first = run.write_lock(**_lock_args(run_config))

    previous = run.assert_final_test_allowed(allow_rerun=True, rerun_reason="found a data bug")
    second = run.write_lock(**_lock_args(run_config), previous=previous, rerun_reason="found a data bug")

    assert first.rerun_count == 0
    assert second.rerun_count == 1
    assert len(second.rerun_history) == 1
    assert second.rerun_history[0]["reason"] == "found a data bug"
    assert second.rerun_history[0]["rerun_number"] == 1

    # A third pass keeps accumulating rather than resetting.
    previous = run.assert_final_test_allowed(allow_rerun=True, rerun_reason="second look")
    third = run.write_lock(**_lock_args(run_config), previous=previous, rerun_reason="second look")
    assert third.rerun_count == 2
    assert len(third.rerun_history) == 2


def test_lock_survives_a_round_trip(run_config):
    run = create_run(run_config, run_id="roundtrip")
    written = run.write_lock(**_lock_args(run_config))

    reloaded = open_run(run_config, "roundtrip").read_lock()

    assert reloaded is not None
    assert reloaded.completed_at_utc == written.completed_at_utc
    assert reloaded.selected_candidate == written.selected_candidate


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------
def test_common_artifacts_record_config_environment_and_data(run_config, dataset):
    from stock_movement.artifacts import write_common_artifacts
    from stock_movement.provenance import utc_now_iso

    run = create_run(run_config, run_id="common")
    write_common_artifacts(
        run, run_config, dataset.metadata, dataset.summary(), dataset.manifest, utc_now_iso()
    )

    for name in (
        "resolved_config.yaml",
        "resolved_config.json",
        "environment.json",
        "data_manifest.json",
        "feature_manifest.json",
    ):
        assert run.exists(name), name

    environment = run.read_json("environment.json")
    assert environment["config_sha256"] == run_config.sha256()
    assert "git" in environment
    assert "packages" in environment

    manifest = run.read_json("data_manifest.json")
    assert manifest["config_sha256"] == run_config.sha256()
    assert manifest["artifact_schema_version"] == 2


def test_finalize_environment_stamps_the_finish_time(run_config, dataset):
    from stock_movement.artifacts import finalize_environment, write_common_artifacts
    from stock_movement.provenance import utc_now_iso

    run = create_run(run_config, run_id="finalize")
    write_common_artifacts(
        run, run_config, dataset.metadata, dataset.summary(), dataset.manifest, utc_now_iso()
    )
    assert run.read_json("environment.json")["run_finished_utc"] is None

    finalize_environment(run)
    assert run.read_json("environment.json")["run_finished_utc"] is not None
