"""The LLM client seam.

``run_lens`` depends only on the :class:`LlmClient` protocol, never on the
Anthropic SDK directly. That makes the whole runtime deterministically testable:
tests inject a :class:`FakeLlmClient` with scripted responses, while production
wraps :class:`AnthropicLlmClient`. :class:`TracingLlmClient` decorates any client
to emit a ``TraceRecord`` on every call (the "every LLM call emits a TraceRecord"
contract), so tracing is orthogonal to which client is underneath.

Internal message types are deliberately a thin, SDK-independent shape so fakes are
trivial to construct and the trace captures plain text.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..ids import IdGenerator, UuidGenerator, new_trace_record_id
from ..models import TraceRecord

if TYPE_CHECKING:
    from ..store.interface import TraceSink


# --- SDK-independent message shape ------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


Block = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: list[Block]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class LlmRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec]
    caller_ref: str
    tool_choice: Literal["auto", "any"] = "auto"
    max_tokens: int = 8192


@dataclass(frozen=True)
class LlmResponse:
    content: list[Block]
    stop_reason: str = "end_turn"

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class LlmClient(Protocol):
    async def create_message(self, request: LlmRequest) -> LlmResponse: ...


# --- Production client -------------------------------------------------------


class AnthropicLlmClient:
    """Wraps ``AsyncAnthropic``. Prompt caching is enabled on the system block —
    the lens frame and brief are stable across the loop's turns (claude-api
    guidance). Requires ``ANTHROPIC_API_KEY`` in the environment."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        # Typed as Any: this adapter builds request params dynamically, so the
        # SDK's strict create() overloads are not a useful check here.
        self._client: Any = client

    async def create_message(self, request: LlmRequest) -> LlmResponse:
        message = await self._client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[_message_to_api(m) for m in request.messages],
            tools=[_tool_to_api(t) for t in request.tools],
            tool_choice={"type": request.tool_choice},
        )
        return LlmResponse(
            content=[_block_from_api(b) for b in message.content],
            stop_reason=message.stop_reason or "end_turn",
        )


def _tool_to_api(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _message_to_api(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_block_to_api(b) for b in message.content]}


def _block_to_api(block: Block) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def _block_from_api(block: Any) -> Block:
    if block.type == "tool_use":
        return ToolUseBlock(id=block.id, name=block.name, input=dict(block.input))
    return TextBlock(text=getattr(block, "text", ""))


# --- Tracing decorator -------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TracingLlmClient:
    """Decorates any :class:`LlmClient`, appending a ``TraceRecord`` per call."""

    def __init__(
        self,
        inner: LlmClient,
        sink: TraceSink,
        id_generator: IdGenerator | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._inner = inner
        self._sink = sink
        self._ids = id_generator or UuidGenerator()
        self._clock = clock

    async def create_message(self, request: LlmRequest) -> LlmResponse:
        response = await self._inner.create_message(request)
        self._sink.append_trace(
            TraceRecord(
                id=new_trace_record_id(self._ids),
                caller_ref=request.caller_ref,
                prompt=_render_request(request),
                completion=_render_response(response),
                model=request.model,
                timestamp=self._clock(),
            )
        )
        return response


def _render_request(request: LlmRequest) -> str:
    lines = [f"SYSTEM:\n{request.system}", ""]
    for message in request.messages:
        lines.append(f"{message.role.upper()}:")
        for block in message.content:
            lines.append(_render_block(block))
    return "\n".join(lines)


def _render_response(response: LlmResponse) -> str:
    return "\n".join(_render_block(b) for b in response.content)


def _render_block(block: Block) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolUseBlock):
        return f"[tool_use {block.name}] {json.dumps(block.input, ensure_ascii=False)}"
    marker = "tool_result error" if block.is_error else "tool_result"
    return f"[{marker} {block.tool_use_id}] {block.content}"


# --- Fake client for tests ---------------------------------------------------


class FakeLlmClient:
    """Returns scripted responses in order and records the requests it received,
    so tests can assert on what the runner sent (anonymization, tool advertising)."""

    def __init__(self, responses: list[LlmResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[LlmRequest] = []

    async def create_message(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeLlmClient ran out of scripted responses")
        return self._responses.pop(0)
