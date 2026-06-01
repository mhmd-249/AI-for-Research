# Stable command interface for the project and the Sandcastle agents.
# The implement/review/merge prompts invoke `make typecheck` and `make test`.

.PHONY: install typecheck test lint format check

install:
	uv sync

typecheck:
	uv run mypy

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint typecheck test
