# Coding Standards

This is a Python 3.12 project managed with `uv`. The foundation slices (issues #1
and #2) established the conventions below; every later slice should follow them.

## Tooling

- `make typecheck` (mypy `--strict`) and `make test` (pytest) must both be clean
  before committing. `make lint` runs `ruff check .`; `make format` runs
  `ruff format .`.
- Add dependencies with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev), so
  `pyproject.toml` and `uv.lock` stay in sync. Never hand-edit the lockfile.

## Style

- Type-annotate everything; mypy strict must pass with no new `# type: ignore`
  unless justified with a comment. Prefer precise types over `Any`.
- Use modern 3.12 syntax: `X | None` (not `Optional[X]`), built-in generics
  (`list`, `dict`), PEP 695 type parameters (`def f[T](...)`), `StrEnum` for
  string enums.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE`
  for module constants.
- Line length 100. Let `ruff format` decide formatting; don't fight it.
- Modules and public functions get a short docstring stating intent and, where
  relevant, the PRD story number(s) the behavior implements.

## Domain model & schemas

- All domain objects are **frozen Pydantic models** subclassing `FrozenModel`
  (`frozen=True, extra="forbid"`). They are immutable: derive new state with
  `model_copy(update=...)`, never mutate in place.
- Pydantic models are the single source of truth for shape and validation — do
  not hand-write parallel dataclasses/`TypedDict`s for the same data.
- IDs are branded `NewType`s minted through an injectable `IdGenerator`
  (`new_*_id(gen)`); never build id strings ad hoc. Source ids are *computed*
  via `domain.source_ops`, not minted.
- Enums and `Literal`s live in `enums.py`; reuse them rather than passing bare
  strings.

## Architecture

- Keep the `store` a thin persistence boundary. Domain invariants (versioning,
  state transitions, identity) live in pure functions under `domain/`. Anything
  new the store persists must round-trip through the `SessionStore` interface and
  be honored by both the in-memory and SQLite implementations.
- Code enforces the contract; prompts do not. Schema validity, tool-access scope,
  anonymization, and round membership are enforced in code (see `runtime/`).
  Quality/behavior is guided by prompt + eval.
- Expected outcomes are returned as typed values (discriminated unions /
  `*Outcome` dataclasses), not raised as exceptions. Reserve exceptions for
  genuine infrastructure failures.
- Every LLM call goes through the `LlmClient` seam so it stays mockable; never
  import or call the Anthropic SDK directly from business logic.

## Testing

- Tests exercise external behavior, not implementation details. Lens/agent output
  *validity* is unit-tested; output *quality* is not (that is eval, a later
  slice).
- LLM-dependent code is tested with a scripted `FakeLlmClient` — deterministic,
  no API key, no token spend in CI. Real-API checks are opt-in scripts.
- Use the shared fixtures and builders in `tests/conftest.py`. Every public
  behavior added should come with at least one test; name tests for the behavior
  they assert.
