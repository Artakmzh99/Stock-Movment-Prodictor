"""Command line interface.

    uv run stock-movement select-model --config configs/default.yaml
    uv run stock-movement final-test  --run-id <RUN_ID>
    uv run stock-movement predict     --run-id <RUN_ID> --latest

Every failure exits non-zero. A pipeline step that fails quietly with status 0 is
worse than one that crashes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from typing import Any

import pandas as pd

from .artifacts import (
    FinalTestAlreadyCompletedError,
    RunAlreadyExistsError,
    list_runs,
    open_run,
)
from .backtest import backtest_comparison
from .config import Config, load_config, set_global_seed
from .data import get_ohlcv
from .dataset import build_dataset
from .evaluation import PRIMARY_METRIC
from .inference import InferenceError, predict
from .persistence import ModelPersistenceError
from .pipeline import run_all, run_final_test_stage, run_selection_stage
from .provenance import ChecksumError
from .selection import development_bounds, run_selection
from .validation import DataValidationError

DEFAULT_CONFIG = "configs/default.yaml"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_frame(frame: pd.DataFrame, title: str, decimals: int = 4) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(frame.round(decimals).to_string())


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if getattr(args, "ticker", None):
        overrides["data.ticker"] = args.ticker
    if getattr(args, "end_date", None):
        overrides["data.end_date"] = args.end_date
    return load_config(args.config, overrides=overrides or None)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_download(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    bundle = get_ohlcv(config, force_refresh=args.force_refresh)
    prices = bundle.prices

    print(
        f"\n{config.data.ticker}: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}"
    )
    print(f"cache hit: {bundle.metadata.get('cache_hit')}")
    print(f"raw sha256: {bundle.metadata.get('raw_sha256')}")
    print(f"\n{prices.tail().to_string()}")

    decision = bundle.partial_bar
    print(f"\nfinal bar: {'DROPPED' if decision.drop_last_row else 'kept'} — {decision.reason}")
    print(f"exchange {decision.exchange} ({decision.exchange_timezone})")

    report = bundle.report
    print(f"\nvalidation: {report.n_rows} rows, max calendar gap {report.max_calendar_gap_days}d")
    for warning in report.warnings:
        print(f"  ! {warning}")
    if not report.warnings:
        print("  no warnings")
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    dataset = build_dataset(config, force_refresh=args.force_refresh)

    print(f"\ndataset: {json.dumps(dataset.summary(), indent=2)}")
    print(f"\n{dataset.manifest['n_features']} features (manifest order):")
    for i, name in enumerate(dataset.feature_names, 1):
        print(f"  {i:2d}. {name}")
    print(f"\nrows dropped for NaN (rolling warm-up): {dataset.manifest['rows_dropped_for_nan']}")
    print(f"label balance: {json.dumps(dataset.manifest['labels'], indent=2)}")

    out_dir = config.path("data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = dataset.X.copy()
    frame["target"] = dataset.y
    frame["future_return"] = dataset.future_return
    target = out_dir / f"{config.data.ticker}_{config.labels.target_definition}_features.parquet"
    frame.to_parquet(target)
    (out_dir / f"{config.data.ticker}_feature_manifest.json").write_text(
        json.dumps(dataset.manifest, indent=2)
    )
    print(f"\nwritten: {target}")
    return 0


def cmd_evaluate_candidates(args: argparse.Namespace) -> int:
    """Development-only candidate comparison, without creating a run directory."""
    config = _config_from_args(args)
    dataset = build_dataset(config, force_refresh=args.force_refresh)
    n_development, test_start = development_bounds(dataset, config)

    print(
        f"\ndevelopment: {dataset.index[0].date()} .. {dataset.index[n_development - 1].date()} "
        f"({n_development} rows) | holdout sealed from {dataset.index[test_start].date()}"
    )

    outcome = run_selection(dataset, config)
    summary = outcome.summary_frame
    columns = [
        c
        for c in (
            "candidate",
            "is_baseline",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "roc_auc_mean",
            "log_loss_mean",
            "complexity_rank",
        )
        if c in summary.columns
    ]
    _print_frame(summary[columns].set_index("candidate"), "Development walk-forward comparison")

    print(f"\nwould select: {outcome.winner.name}")
    print(f"edge_detected: {outcome.edge.edge_detected}")
    print(f"{outcome.edge.reason}")
    print("\nNo final-test metric was computed and no run directory was created.")
    return 0


def cmd_select_model(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    output = run_selection_stage(config, force_refresh=args.force_refresh, run_id=args.run_id)
    outcome = output.outcome

    summary = outcome.summary_frame
    columns = [
        c
        for c in (
            "candidate",
            "is_baseline",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "roc_auc_mean",
            "log_loss_mean",
        )
        if c in summary.columns
    ]
    _print_frame(summary[columns].set_index("candidate"), "Development walk-forward comparison")

    print(f"\nSELECTED: {outcome.winner.name}")
    print(f"thresholds: {output.thresholds}")
    print(f"edge_detected: {outcome.edge.edge_detected}")
    print(f"{outcome.edge.reason}")
    print(f"\nrun id: {output.run.run_id}")
    print(f"artifacts: {output.run.path}")
    print(f"\nnext: stock-movement final-test --run-id {output.run.run_id}")
    return 0


def cmd_final_test(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    output = run_final_test_stage(
        config,
        run_id=args.run_id,
        allow_rerun=args.allow_test_rerun,
        rerun_reason=args.rerun_reason,
        force_refresh=args.force_refresh,
    )
    _report_final(output)
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _, final = run_all(config, force_refresh=args.force_refresh, run_id=args.run_id)
    _report_final(final)
    return 0


def _report_final(output: Any) -> None:
    columns = [
        c
        for c in (
            "n",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "f1_macro",
            "mcc",
            "log_loss",
            "brier_score",
        )
        if c in output.comparison.columns
    ]
    _print_frame(output.comparison[columns], "Final-test comparison (scored once)")

    backtest = backtest_comparison(list(output.backtests.values()))
    display = [
        c
        for c in (
            "cumulative_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "exposure",
            "n_active_sessions",
            "n_completed_trades",
            "total_cost_paid",
        )
        if c in backtest.columns
    ]
    _print_frame(backtest[display], "Backtest on the final-test window (net of costs)")

    print(f"\nSELECTED MODEL: {output.selected_candidate}")
    print(f"EDGE DETECTED:  {output.edge_detected}")
    print(f"VERDICT: {output.verdict}")
    print(f"\nfinal test locked at {output.lock.completed_at_utc} (reruns: {output.lock.rerun_count})")
    print(f"artifacts: {output.run.path}")
    print(f"model card: {output.run.path / 'model_card.md'}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    run = open_run(config, args.run_id)

    summary = pd.read_csv(run.path / "candidate_summary.csv")
    columns = [
        c
        for c in (
            "candidate",
            "is_baseline",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "roc_auc_mean",
            "log_loss_mean",
        )
        if c in summary.columns
    ]
    _print_frame(summary[columns].set_index("candidate"), f"Development candidates — {args.run_id}")

    if run.exists("final_test_comparison.csv"):
        comparison = pd.read_csv(run.path / "final_test_comparison.csv", index_col=0)
        available = [
            c
            for c in (
                "n",
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "f1_macro",
                "mcc",
                "log_loss",
                "brier_score",
            )
            if c in comparison.columns
        ]
        _print_frame(comparison[available], "Final-test comparison")
    else:
        print("\nno final-test metrics yet — the holdout has not been opened for this run")

    decision = run.read_json("selection_decision.json")
    print(f"\nselected: {decision['selected_candidate']['name']}")
    print(f"edge_detected: {decision['edge_detected']}")
    print(f"walk-forward {PRIMARY_METRIC}: {decision['selection_score']:.4f}")
    print(f"\nmodel card: {run.path / 'model_card.md'}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    run = open_run(config, args.run_id)

    if not run.exists("backtest_comparison.csv"):
        print(f"no backtest for run {args.run_id} — run final-test first", file=sys.stderr)
        return 1

    metrics = pd.read_csv(run.path / "backtest_comparison.csv", index_col=0)
    display = [
        c
        for c in (
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
            "total_cost_paid",
        )
        if c in metrics.columns
    ]
    _print_frame(metrics[display], f"Backtest (net of costs) — {args.run_id}")

    if run.exists("bootstrap_summary.json"):
        bootstrap = run.read_json("bootstrap_summary.json")
        print("\nMoving-block bootstrap (95% CI)")
        print("==============================")
        for name, entry in bootstrap.items():
            print(
                f"  {name:34s} {entry.get('point', float('nan')):+.6f} "
                f"[{entry.get('ci_low', float('nan')):+.6f}, {entry.get('ci_high', float('nan')):+.6f}]"
                f"  excludes zero: {entry.get('interval_excludes_zero')}"
            )

    print("\nExecution assumptions and cost model: see model_card.md.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    as_of: date | None = None
    if not args.latest:
        if not args.as_of:
            print("either --as-of YYYY-MM-DD or --latest is required", file=sys.stderr)
            return 2
        as_of = date.fromisoformat(args.as_of)

    prediction = predict(run_id=args.run_id, config=config, as_of=as_of, force_refresh=args.force_refresh)
    print(prediction.to_json())
    for warning in prediction.warnings:
        print(f"\n! {warning}", file=sys.stderr)
    return 0


def cmd_show_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    run = open_run(config, args.run_id)

    decision = run.read_json("selection_decision.json") if run.exists("selection_decision.json") else {}
    manifest = run.read_json("data_manifest.json") if run.exists("data_manifest.json") else {}
    environment = run.read_json("environment.json") if run.exists("environment.json") else {}
    lock = run.read_lock()

    model_dir = run.path / "model"
    model_files = sorted(p.name for p in model_dir.iterdir()) if model_dir.exists() else []

    rows = [
        ("run id", run.run_id),
        ("git commit", (environment.get("git") or {}).get("commit") or "n/a"),
        ("git dirty", (environment.get("git") or {}).get("dirty")),
        ("config sha256", manifest.get("config_sha256") or "n/a"),
        ("data sha256", manifest.get("raw_sha256") or "n/a"),
        (
            "target definition",
            (run.read_json("resolved_config.json") or {}).get("labels", {}).get("target_definition"),
        ),
        (
            "execution mode",
            (run.read_json("resolved_config.json") or {}).get("backtest", {}).get("execution_mode"),
        ),
        ("selected candidate", (decision.get("selected_candidate") or {}).get("name", "n/a")),
        ("edge detected", decision.get("edge_detected")),
        (
            "final test",
            f"completed {lock.completed_at_utc} (reruns {lock.rerun_count})" if lock else "not run",
        ),
        ("model artifact", ", ".join(model_files) if model_files else "none saved"),
    ]

    width = max(len(label) for label, _ in rows)
    print(f"\nRun {run.run_id}")
    print("=" * (len(run.run_id) + 4))
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
    print(f"\npath: {run.path}")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    runs = list_runs(config)
    if not runs:
        print("no runs yet")
        return 0
    print(f"{len(runs)} run(s) in {config.path('runs')}:")
    for name in runs:
        print(f"  {name}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-movement",
        description=(
            "Next-session stock direction prediction with development-only model "
            "selection, a locked one-time final test, and executable cost modelling."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def base(name: str, help_text: str, refresh: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=DEFAULT_CONFIG, help="path to a YAML config")
        if refresh:
            p.add_argument(
                "--force-refresh", action="store_true", help="re-download instead of using the cache"
            )
        else:
            p.set_defaults(force_refresh=False)
        return p

    def with_data_overrides(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--ticker", default=None, help="override the configured ticker")
        p.add_argument("--end-date", default=None, help="override the data end date (YYYY-MM-DD)")
        return p

    with_data_overrides(base("download", "download and validate OHLCV")).set_defaults(func=cmd_download)
    with_data_overrides(base("build-features", "build the feature/label table")).set_defaults(
        func=cmd_build_features
    )
    with_data_overrides(
        base("evaluate-candidates", "compare candidates on development folds only")
    ).set_defaults(func=cmd_evaluate_candidates)

    select = with_data_overrides(
        base("select-model", "evaluate candidates and lock one selection (no test metrics)")
    )
    select.add_argument("--run-id", default=None, help="explicit run id")
    select.set_defaults(func=cmd_select_model)

    final = base("final-test", "score the sealed holdout exactly once")
    final.add_argument("--run-id", required=True)
    final.add_argument(
        "--allow-test-rerun",
        action="store_true",
        help="permit re-scoring an already-completed holdout (requires --rerun-reason)",
    )
    final.add_argument("--rerun-reason", default=None, help="why the holdout is being re-scored")
    final.set_defaults(func=cmd_final_test)

    run_all_parser = with_data_overrides(base("run-all", "select-model then final-test in one command"))
    run_all_parser.add_argument("--run-id", default=None)
    run_all_parser.set_defaults(func=cmd_run_all)

    for name, fn, help_text in (
        ("evaluate", cmd_evaluate, "print metrics from a saved run"),
        ("backtest", cmd_backtest, "print backtest metrics from a saved run"),
        ("show-run", cmd_show_run, "summarise a run's provenance and status"),
    ):
        p = base(name, help_text, refresh=False)
        p.add_argument("--run-id", required=True)
        p.set_defaults(func=fn)

    predict_parser = base("predict", "predict from a saved model without retraining")
    predict_parser.add_argument("--run-id", required=True)
    predict_parser.add_argument("--as-of", default=None, help="signal date (YYYY-MM-DD)")
    predict_parser.add_argument("--latest", action="store_true", help="use the most recent completed session")
    predict_parser.set_defaults(func=cmd_predict)

    base("list-runs", "list saved runs", refresh=False).set_defaults(func=cmd_list_runs)
    return parser


EXPECTED_ERRORS = (
    DataValidationError,
    ChecksumError,
    InferenceError,
    ModelPersistenceError,
    RunAlreadyExistsError,
    FinalTestAlreadyCompletedError,
    FileNotFoundError,
    ValueError,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: could not load config {args.config!r}: {exc}", file=sys.stderr)
        return 2

    set_global_seed(config.random_seed)

    try:
        result = args.func(args)
        return int(result)
    except EXPECTED_ERRORS as exc:
        # Expected, actionable failures: report cleanly and exit non-zero.
        print(f"\nerror: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
