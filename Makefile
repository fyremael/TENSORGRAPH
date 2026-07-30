.PHONY: install lint format typecheck test test-cov demo demo-fx clean all

install:
	pip install -e ".[dev]"

lint:
	ruff check tensorgraph tests
	ruff format --check tensorgraph tests

format:
	ruff format tensorgraph tests
	ruff check --fix tensorgraph tests

typecheck:
	mypy tensorgraph

test:
	pytest tests -v

test-cov:
	pytest tests -v --cov=tensorgraph --cov-report=term-missing

demo:
	python -m tensorgraph.examples.demo_core

demo-fx:
	python -m tensorgraph.cli.optimize_fx --model toy_lora_chain --in-dim 16 --out-dim 8

clean:
ifeq ($(OS),Windows_NT)
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
endif

all: lint typecheck test
