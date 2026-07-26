# Stock Movement Predictor — common tasks.
#
# Everything runs through uv against the locked environment, so `make quality`
# reproduces the CI gate exactly.

.DEFAULT_GOAL := help

.PHONY: help install format lint typecheck test test-fast test-slow test-cov quality \
        download features candidates select final run predict show clean clean-runs

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## create/sync the locked environment with all extras
	uv sync --all-extras

format:  ## rewrite code to the canonical style
	uv run ruff format .

lint:  ## check style and lint rules
	uv run ruff check .

typecheck:  ## strict type check of src/
	uv run mypy src

test:  ## run the full test suite
	uv run pytest -q

test-fast:  ## skip the slow (TensorFlow) tests
	uv run pytest -m "not slow" -q

test-slow:  ## run only the slow LSTM tests (needs the lstm extra)
	uv run pytest -m slow

test-cov:  ## run tests with a coverage report
	uv run pytest -m "not slow" --cov=stock_movement --cov-report=term-missing --cov-fail-under=90

quality:  ## the full CI gate: format, lint, types, tests, coverage
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest -m "not slow" --cov=stock_movement --cov-fail-under=90

# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------
CONFIG ?= configs/default.yaml
RUN_ID ?=

download:  ## download and validate OHLCV
	uv run stock-movement download --config $(CONFIG)

features:  ## build the feature/label table
	uv run stock-movement build-features --config $(CONFIG)

candidates:  ## compare candidates on development folds only (no run directory)
	uv run stock-movement evaluate-candidates --config $(CONFIG)

select:  ## stage 1 — evaluate candidates and lock one selection
	uv run stock-movement select-model --config $(CONFIG)

final:  ## stage 2 — score the sealed holdout once (needs RUN_ID=...)
	@test -n "$(RUN_ID)" || (echo "usage: make final RUN_ID=<run id>" && exit 1)
	uv run stock-movement final-test --config $(CONFIG) --run-id $(RUN_ID)

run:  ## reproduce the published README results end to end
	uv run stock-movement run-all --config configs/reproduction/readme_aapl_2026_07.yaml

predict:  ## predict from a saved model without retraining (needs RUN_ID=...)
	@test -n "$(RUN_ID)" || (echo "usage: make predict RUN_ID=<run id>" && exit 1)
	uv run stock-movement predict --config $(CONFIG) --run-id $(RUN_ID) --latest

show:  ## summarise a run's provenance and status (needs RUN_ID=...)
	@test -n "$(RUN_ID)" || (echo "usage: make show RUN_ID=<run id>" && exit 1)
	uv run stock-movement show-run --config $(CONFIG) --run-id $(RUN_ID)

clean:  ## remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist coverage.xml .coverage \
	       src/*.egg-info

clean-runs:  ## delete every saved run (irreversible)
	rm -rf artifacts/runs/*/
