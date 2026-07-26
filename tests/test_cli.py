"""CLI behaviour (P2.5).

The contract under test: every command works offline against synthetic data, and
**every failure exits non-zero**. A pipeline step that fails quietly with status 0
is worse than one that crashes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from stock_movement.cli import main
from tests.conftest import make_ohlcv, small_config_payload


@pytest.fixture
def cli_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """A config file on disk wired to synthetic data and a temp run directory."""
    frame = make_ohlcv(n=1400, seed=11)
    monkeypatch.setattr(
        "stock_movement.data.download_ohlcv",
        lambda ticker, start, end, interval="1d", auto_adjust=True: frame.copy(),
    )

    payload = small_config_payload(
        run_name="cli-test",
        data={"ticker": "TEST", "min_rows": 200, "end_date": "2026-01-02"},
        paths={
            "data_raw": str(tmp_path / "raw"),
            "data_processed": str(tmp_path / "processed"),
            "runs": str(tmp_path / "runs"),
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def _run(*args: str) -> int:
    return main(list(args))


# --------------------------------------------------------------------------
# read-only commands
# --------------------------------------------------------------------------
def test_download_reports_provenance(cli_config, capsys):
    assert _run("download", "--config", cli_config) == 0

    out = capsys.readouterr().out
    assert "TEST:" in out
    assert "raw sha256:" in out
    assert "final bar:" in out


def test_build_features_lists_the_manifest_order(cli_config, capsys):
    assert _run("build-features", "--config", cli_config) == 0

    out = capsys.readouterr().out
    assert "features (manifest order)" in out
    assert "open_to_close_return" in out
    assert "written:" in out


def test_evaluate_candidates_computes_no_test_metrics(cli_config, capsys):
    assert _run("evaluate-candidates", "--config", cli_config) == 0

    out = capsys.readouterr().out
    assert "Development walk-forward comparison" in out
    assert "would select:" in out
    assert "No final-test metric was computed" in out
    assert "no run directory was created" in out


def test_list_runs_on_an_empty_directory(cli_config, capsys):
    assert _run("list-runs", "--config", cli_config) == 0
    assert "no runs yet" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the two-stage flow
# --------------------------------------------------------------------------
def _select(cli_config, capsys) -> str:
    assert _run("select-model", "--config", cli_config) == 0
    out = capsys.readouterr().out
    assert "SELECTED:" in out
    run_id = next(line.split("run id:")[1].strip() for line in out.splitlines() if "run id:" in line)
    return run_id


def test_select_model_then_final_test(cli_config, capsys):
    run_id = _select(cli_config, capsys)

    assert _run("final-test", "--config", cli_config, "--run-id", run_id) == 0
    out = capsys.readouterr().out

    assert "Final-test comparison (scored once)" in out
    assert "SELECTED MODEL:" in out
    assert "EDGE DETECTED:" in out
    assert "final test locked at" in out


def test_second_final_test_exits_non_zero(cli_config, capsys):
    run_id = _select(cli_config, capsys)
    assert _run("final-test", "--config", cli_config, "--run-id", run_id) == 0
    capsys.readouterr()

    assert _run("final-test", "--config", cli_config, "--run-id", run_id) == 1
    assert "already completed" in capsys.readouterr().err


def test_rerun_without_a_reason_exits_non_zero(cli_config, capsys):
    run_id = _select(cli_config, capsys)
    assert _run("final-test", "--config", cli_config, "--run-id", run_id) == 0
    capsys.readouterr()

    assert _run("final-test", "--config", cli_config, "--run-id", run_id, "--allow-test-rerun") == 1
    assert "non-empty explanation" in capsys.readouterr().err


def test_rerun_with_a_reason_succeeds(cli_config, capsys):
    run_id = _select(cli_config, capsys)
    assert _run("final-test", "--config", cli_config, "--run-id", run_id) == 0

    assert (
        _run(
            "final-test",
            "--config",
            cli_config,
            "--run-id",
            run_id,
            "--allow-test-rerun",
            "--rerun-reason",
            "verifying a data fix",
        )
        == 0
    )
    assert "reruns: 1" in capsys.readouterr().out


def test_run_all_does_both_stages(cli_config, capsys):
    assert _run("run-all", "--config", cli_config) == 0

    out = capsys.readouterr().out
    assert "Final-test comparison (scored once)" in out
    assert "Backtest on the final-test window" in out
    assert "VERDICT:" in out


def test_run_all_reports_distinct_trade_counters(cli_config, capsys):
    assert _run("run-all", "--config", cli_config) == 0

    out = capsys.readouterr().out
    assert "n_active_sessions" in out
    assert "n_completed_trades" in out
    assert "always_long_intraday" in out
    assert "buy_and_hold_close_to_close" in out


# --------------------------------------------------------------------------
# reporting commands
# --------------------------------------------------------------------------
def test_evaluate_and_backtest_read_a_finished_run(cli_config, capsys):
    assert _run("run-all", "--config", cli_config, "--run-id", "finished") == 0
    capsys.readouterr()

    assert _run("evaluate", "--config", cli_config, "--run-id", "finished") == 0
    out = capsys.readouterr().out
    assert "Development candidates" in out
    assert "Final-test comparison" in out
    assert "edge_detected:" in out

    assert _run("backtest", "--config", cli_config, "--run-id", "finished") == 0
    out = capsys.readouterr().out
    assert "Backtest (net of costs)" in out
    assert "Moving-block bootstrap" in out
    assert "execution_mode" in out


def test_evaluate_before_the_final_test_says_so(cli_config, capsys):
    run_id = _select(cli_config, capsys)

    assert _run("evaluate", "--config", cli_config, "--run-id", run_id) == 0
    assert "holdout has not been opened" in capsys.readouterr().out


def test_backtest_before_the_final_test_exits_non_zero(cli_config, capsys):
    run_id = _select(cli_config, capsys)

    assert _run("backtest", "--config", cli_config, "--run-id", run_id) == 1
    assert "run final-test first" in capsys.readouterr().err


def test_show_run_summarises_provenance(cli_config, capsys):
    assert _run("run-all", "--config", cli_config, "--run-id", "shown") == 0
    capsys.readouterr()

    assert _run("show-run", "--config", cli_config, "--run-id", "shown") == 0
    out = capsys.readouterr().out

    for label in (
        "run id",
        "git commit",
        "config sha256",
        "data sha256",
        "target definition",
        "execution mode",
        "selected candidate",
        "edge detected",
        "final test",
        "model artifact",
    ):
        assert label in out, label
    assert "model.joblib" in out


def test_list_runs_shows_created_runs(cli_config, capsys):
    assert _run("select-model", "--config", cli_config, "--run-id", "run-one") == 0
    capsys.readouterr()

    assert _run("list-runs", "--config", cli_config) == 0
    assert "run-one" in capsys.readouterr().out


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
def test_predict_latest_emits_json(cli_config, capsys):
    assert _run("run-all", "--config", cli_config, "--run-id", "predictable") == 0
    capsys.readouterr()

    assert _run("predict", "--config", cli_config, "--run-id", "predictable", "--latest") == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ticker"] == "TEST"
    assert 0.0 <= payload["probability_up"] <= 1.0
    assert payload["trading_signal"] in ("long", "cash")


def test_predict_requires_a_date_or_latest(cli_config, capsys):
    assert _run("run-all", "--config", cli_config, "--run-id", "needs-date") == 0
    capsys.readouterr()

    assert _run("predict", "--config", cli_config, "--run-id", "needs-date") == 2
    assert "--as-of" in capsys.readouterr().err


def test_predict_on_a_run_without_a_model_exits_non_zero(cli_config, capsys):
    run_id = _select(cli_config, capsys)

    assert _run("predict", "--config", cli_config, "--run-id", run_id, "--latest") == 1
    assert "no model metadata" in capsys.readouterr().err


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------
def test_unknown_run_exits_non_zero(cli_config, capsys):
    assert _run("show-run", "--config", cli_config, "--run-id", "nope") == 1
    assert "not found" in capsys.readouterr().err


def test_missing_config_exits_two(capsys):
    assert _run("download", "--config", "configs/not_a_real_config.yaml") == 2
    assert "could not load config" in capsys.readouterr().err


def test_invalid_config_exits_two(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"data": {"not_a_field": 1}}))

    assert _run("download", "--config", str(bad)) == 2
    assert "could not load config" in capsys.readouterr().err


def test_incoherent_config_is_refused(tmp_path, capsys):
    """A close-to-close label with next_open execution must not run at all."""
    bad = tmp_path / "incoherent.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "labels": {"target_definition": "close_to_close"},
                "backtest": {"execution_mode": "next_open"},
            }
        )
    )

    assert _run("download", "--config", str(bad)) == 2
    assert "could not load config" in capsys.readouterr().err


def test_ticker_override_is_applied(cli_config, capsys, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_download(ticker, start, end, interval="1d", auto_adjust=True):
        captured["ticker"] = ticker
        return make_ohlcv(n=1400, seed=11)

    monkeypatch.setattr("stock_movement.data.download_ohlcv", fake_download)

    assert _run("download", "--config", cli_config, "--ticker", "NVDA") == 0
    assert captured["ticker"] == "NVDA"


def test_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_verbose_flag_is_accepted(cli_config):
    assert _run("-v", "download", "--config", cli_config) == 0
