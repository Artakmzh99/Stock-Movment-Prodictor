"""Stock Movement Predictor.

Next-day price *direction* classification with mandatory baselines, enforced
leakage controls, walk-forward validation and cost-aware backtesting.

Quick start::

    from stock_movement.config import load_config
    from stock_movement.pipeline import run_pipeline

    output = run_pipeline(load_config("configs/experiments/logistic_aapl.yaml"))
    print(output.verdict)

Research and education only. Not investment advice.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

__all__ = ["__version__", "build_dataset", "load_config", "run_pipeline"]

_LAZY_EXPORTS = {
    "load_config": ("stock_movement.config", "load_config"),
    "build_dataset": ("stock_movement.dataset", "build_dataset"),
    "run_pipeline": ("stock_movement.pipeline", "run_pipeline"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-exports, so `import stock_movement` stays cheap."""
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attribute = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
