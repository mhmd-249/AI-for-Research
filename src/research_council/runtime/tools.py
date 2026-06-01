"""Lens tools.

A lens's "tools" are its granted domain tools (``verifier_query`` / ``source_fetch``,
wired in later slices) plus the mandatory ``emit_output`` tool that carries the
structured ``LensRunOutputDraft``. Unifying tool access and structured output under
one mechanism means the same code path that enforces tool grants also drives the
final typed emission.

``emit_output`` is handled specially by ``run_lens`` (it terminates the loop) and
is therefore not a registered :class:`Tool`.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..enums import ToolName
from ..models import LensRunOutputDraft, ScopedChallengeOutputDraft

EMIT_OUTPUT_TOOL = "emit_output"


class Tool(Protocol):
    """A domain tool the runner may dispatch on a lens's behalf."""

    @property
    def name(self) -> ToolName: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    async def handle(self, tool_input: dict[str, Any]) -> str:
        """Run the tool and return the text fed back as a tool_result."""
        ...


ToolRegistry = dict[ToolName, Tool]


def emit_output_tool_spec() -> dict[str, Any]:
    """JSON Schema for the round-1/2 lens output envelope."""
    return LensRunOutputDraft.model_json_schema()


def emit_scoped_challenge_tool_spec() -> dict[str, Any]:
    """JSON Schema for the scoped challenge envelope (used by the challenge slice)."""
    return ScopedChallengeOutputDraft.model_json_schema()
