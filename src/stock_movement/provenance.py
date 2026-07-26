"""Checksums, environment capture, and git state.

"Reproducible" is a property, not a claim. It requires knowing exactly which
bytes went in and exactly which code processed them, so this module records:

* SHA-256 of every data file, the resolved config, and the feature manifest;
* the git commit, branch, and whether the working tree was dirty — a result
  produced from uncommitted code is not reproducible and must say so;
* interpreter, OS, and package versions.

Cached files are verified against their recorded digest on load, so silent
corruption or hand-editing of a Parquet cache surfaces as an error rather than as
a mysteriously different result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

CHUNK_SIZE = 1 << 20  # 1 MiB

TRACKED_PACKAGES = (
    "yfinance",
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "pyarrow",
    "pydantic",
    "exchange-calendars",
    "matplotlib",
    "joblib",
    "tensorflow",
)


class ChecksumError(RuntimeError):
    """A file's digest does not match the value recorded when it was written."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_canonical_json(payload: Any) -> str:
    """Hash a structure independently of key order."""
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def verify_file(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ChecksumError(
            f"checksum mismatch for {path}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            "The cached file has changed since it was written. Re-run with --force-refresh."
        )


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=PROJECT_ROOT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_metadata() -> dict[str, Any]:
    """Commit, branch, and dirty state.

    `dirty` matters: a metric produced from a modified working tree cannot be
    reproduced from the recorded commit, and the artifact should admit that.
    """
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_file_count": len(status.splitlines()) if status else 0,
    }


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _accelerator_summary() -> str:
    """Whether a GPU is visible, since it can change floating-point results."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return f"{len(gpus)} GPU(s): {[g.name for g in gpus]}" if gpus else "cpu only"
    except Exception:
        return "cpu only (tensorflow not installed)"


def environment_metadata(run_started_utc: str | None = None) -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "accelerator": _accelerator_summary(),
        "packages": package_versions(),
        "git": git_metadata(),
        "run_started_utc": run_started_utc or utc_now_iso(),
        "run_finished_utc": None,  # filled in when the run completes
    }
