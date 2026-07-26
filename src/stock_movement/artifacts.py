"""Run artifacts: everything needed to reproduce, audit, or disbelieve a result.

## Immutability

A run directory is created once and never overwritten. Re-running a config
produces a *new* directory; a collision raises ``RunAlreadyExistsError``. Silently
overwriting means comparing today's number against a number whose config you no
longer have, which is how phantom improvements get reported.

Run IDs are ``<UTC timestamp>_<short resolved-config hash>``, e.g.
``20260726T184501Z_a13f94c2``. The hash comes from canonical JSON of the fully
resolved config, so it ignores YAML key order and formatting: two configs that
mean the same thing hash the same, and any semantic difference changes the ID.

## The final-test lock

``final_test.lock.json`` records that the holdout has been opened. A second
``final-test`` on the same run fails unless the caller passes both
``--allow-test-rerun`` and a non-empty ``--rerun-reason``, and every rerun is
appended to the lock's history. This is the mechanism that makes "the test set was
used once" a property of the code rather than a claim in a README.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import ARTIFACT_SCHEMA_VERSION, Config
from .provenance import environment_metadata, git_metadata, sha256_canonical_json, utc_now_iso

LOCK_FILE = "final_test.lock.json"


class RunAlreadyExistsError(FileExistsError):
    """A run directory with this ID already exists. Runs are never overwritten."""


class FinalTestAlreadyCompletedError(RuntimeError):
    """The holdout has already been opened for this run."""


def make_run_id(config: Config, now: datetime | None = None) -> str:
    """``<UTC timestamp>_<config hash>``. Encodes *what* was run, not just when."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{config.short_hash()}"


def _unique_run_id(config: Config, root: Path) -> str:
    """An auto-generated run ID that does not collide with an existing directory.

    Timestamps have one-second resolution, so re-running the same config twice
    inside a second would otherwise collide. That is a clock artefact, not a real
    conflict: both are legitimate distinct runs. An explicit ``run_id`` still
    raises on collision, which is the case immutability actually needs to protect.
    """
    base = make_run_id(config)
    if not (root / base).exists():
        return base
    for suffix in range(2, 1000):
        candidate = f"{base}-{suffix}"
        if not (root / candidate).exists():
            return candidate
    raise RunAlreadyExistsError(f"could not find a free run id derived from {base}")


class _JsonEncoder(json.JSONEncoder):
    """numpy scalars, pandas timestamps and paths are not JSON-serialisable."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return value if np.isfinite(value) else str(value)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        return super().default(o)


def _sanitize_for_json(value: Any) -> Any:
    """Replace non-finite floats with strings, recursively.

    ``JSONEncoder.default`` never sees NaN: ``float`` (and ``np.float64``, which
    subclasses it) is a type json handles natively, so it emits the bare literals
    ``NaN`` / ``Infinity``. Those are valid Python-json but *invalid* JSON — strict
    parsers such as JavaScript's ``JSON.parse`` reject them, which would make these
    artifacts unreadable outside Python. NaN is meaningful here (an undefined
    ROC-AUC, say), so it is preserved as a string rather than dropped or zeroed.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def dumps_json(payload: Any) -> str:
    """Serialise to strictly-valid JSON, or fail rather than emit a bad literal."""
    return json.dumps(_sanitize_for_json(payload), indent=2, cls=_JsonEncoder, allow_nan=False)


@dataclass
class FinalTestLock:
    completed: bool
    completed_at_utc: str
    git_commit: str | None
    data_sha256: str | None
    config_sha256: str
    test_start: str
    test_end: str
    n_test_rows: int
    selected_candidate: str
    rerun_count: int = 0
    rerun_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunDirectory:
    """Handle for one run's output directory."""

    def __init__(self, root: Path, run_id: str, must_be_new: bool = True) -> None:
        self.run_id = run_id
        self.path = root / run_id
        self.figures = self.path / "figures"

        if must_be_new and self.path.exists():
            raise RunAlreadyExistsError(
                f"run directory already exists: {self.path}\n"
                "Runs are immutable. Delete it explicitly, or change the config "
                "(the run ID includes the config hash, so a real change yields a new ID)."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    # -- writers ----------------------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        target.write_text(dumps_json(payload))
        return target

    def write_yaml(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.path / name
        target.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
        return target

    def write_csv(self, name: str, frame: pd.DataFrame, index: bool = True) -> Path:
        target = self.path / name
        frame.to_csv(target, index=index)
        return target

    def write_parquet(self, name: str, frame: pd.DataFrame) -> Path:
        target = self.path / name
        frame.to_parquet(target)
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path / name
        target.write_text(text)
        return target

    def figure_path(self, name: str) -> Path:
        return self.figures / name

    def read_json(self, name: str) -> Any:
        target = self.path / name
        if not target.exists():
            raise FileNotFoundError(f"{name} not found in {self.path}")
        return json.loads(target.read_text())

    def exists(self, name: str) -> bool:
        return (self.path / name).exists()

    # -- final-test lock --------------------------------------------------
    @property
    def lock_path(self) -> Path:
        return self.path / LOCK_FILE

    def read_lock(self) -> FinalTestLock | None:
        if not self.lock_path.exists():
            return None
        payload = json.loads(self.lock_path.read_text())
        known = set(FinalTestLock.__dataclass_fields__)
        return FinalTestLock(**{k: v for k, v in payload.items() if k in known})

    def assert_final_test_allowed(
        self, allow_rerun: bool = False, rerun_reason: str | None = None
    ) -> FinalTestLock | None:
        """Refuse a second look at the holdout unless it is explicit and justified."""
        lock = self.read_lock()
        if lock is None or not lock.completed:
            return None

        if not allow_rerun:
            raise FinalTestAlreadyCompletedError(
                f"the final test for run {self.run_id} already completed at "
                f"{lock.completed_at_utc} (rerun count {lock.rerun_count}).\n"
                "Re-scoring the holdout turns it into a validation set. If you genuinely "
                "need to, pass --allow-test-rerun together with --rerun-reason '...'; "
                "every rerun is recorded in the lock."
            )

        if not rerun_reason or not rerun_reason.strip():
            raise ValueError(
                "--rerun-reason must be a non-empty explanation. An unexplained rerun is "
                "indistinguishable from fishing for a better number."
            )
        return lock

    def write_lock(
        self,
        config: Config,
        test_start: str,
        test_end: str,
        n_test_rows: int,
        selected_candidate: str,
        data_sha256: str | None,
        previous: FinalTestLock | None = None,
        rerun_reason: str | None = None,
    ) -> FinalTestLock:
        history = list(previous.rerun_history) if previous else []
        rerun_count = previous.rerun_count if previous else 0

        if previous is not None:
            rerun_count += 1
            history.append(
                {
                    "rerun_number": rerun_count,
                    "at_utc": utc_now_iso(),
                    "reason": rerun_reason,
                    "previous_completed_at_utc": previous.completed_at_utc,
                    "git_commit": git_metadata().get("commit"),
                }
            )

        lock = FinalTestLock(
            completed=True,
            completed_at_utc=utc_now_iso(),
            git_commit=git_metadata().get("commit"),
            data_sha256=data_sha256,
            config_sha256=config.sha256(),
            test_start=test_start,
            test_end=test_end,
            n_test_rows=n_test_rows,
            selected_candidate=selected_candidate,
            rerun_count=rerun_count,
            rerun_history=history,
        )
        self.lock_path.write_text(dumps_json(lock.to_dict()))
        return lock


def create_run(config: Config, run_id: str | None = None, must_be_new: bool = True) -> RunDirectory:
    root = config.path("runs")
    if run_id is None:
        root.mkdir(parents=True, exist_ok=True)
        run_id = _unique_run_id(config, root)
    return RunDirectory(root, run_id, must_be_new=must_be_new)


def open_run(config: Config, run_id: str) -> RunDirectory:
    """Open an existing run for reading or for the final-test stage."""
    root = config.path("runs")
    path = root / run_id
    if not path.exists():
        available = sorted(p.name for p in root.glob("*") if p.is_dir())
        hint = "\n  ".join(available[-10:]) if available else "(no runs yet)"
        raise FileNotFoundError(f"run {run_id!r} not found in {root}.\nAvailable runs:\n  {hint}")
    return RunDirectory(root, run_id, must_be_new=False)


def list_runs(config: Config) -> list[str]:
    root = config.path("runs")
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def build_data_manifest(
    config: Config,
    dataset_metadata: dict[str, Any],
    dataset_summary: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Hashes and provenance for every input that shaped the result."""
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "ticker": config.data.ticker,
        "benchmark_ticker": config.data.benchmark_ticker,
        "interval": config.data.interval,
        "auto_adjust": config.data.auto_adjust,
        "start_date": str(config.data.start_date),
        "end_date": str(config.data.end_date) if config.data.end_date else None,
        "config_sha256": config.sha256(),
        "feature_manifest_sha256": sha256_canonical_json(feature_manifest),
        "dataset_summary": dataset_summary,
        "sources": dataset_metadata,
    }
    target = dataset_metadata.get("target", {})
    manifest["raw_sha256"] = target.get("raw_sha256")
    manifest["partial_bar_decision"] = target.get("partial_bar_decision")
    return manifest


def write_common_artifacts(
    run: RunDirectory,
    config: Config,
    dataset_metadata: dict[str, Any],
    dataset_summary: dict[str, Any],
    feature_manifest: dict[str, Any],
    run_started_utc: str,
) -> dict[str, Any]:
    """Config, environment and data manifests — written by every stage."""
    run.write_yaml("resolved_config.yaml", config.to_dict())
    run.write_json("resolved_config.json", config.to_dict())

    environment = environment_metadata(run_started_utc=run_started_utc)
    environment["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    environment["run_id"] = run.run_id
    environment["config_sha256"] = config.sha256()
    run.write_json("environment.json", environment)

    manifest = build_data_manifest(config, dataset_metadata, dataset_summary, feature_manifest)
    run.write_json("data_manifest.json", manifest)
    run.write_json("feature_manifest.json", feature_manifest)
    return manifest


def finalize_environment(run: RunDirectory) -> None:
    """Stamp the finish time onto environment.json."""
    if not run.exists("environment.json"):
        return
    environment = run.read_json("environment.json")
    environment["run_finished_utc"] = utc_now_iso()
    run.write_json("environment.json", environment)
