# Research Council

A multi-agent deliberation system that pressure-tests research proposals against a
panel of methodological lenses with per-claim literature verification. See the PRDs
in the repo root for the full design.

## Development

```sh
uv sync          # install deps into .venv
uv run pytest    # run tests
uv run mypy      # type-check (strict)
uv run ruff check # lint
```
