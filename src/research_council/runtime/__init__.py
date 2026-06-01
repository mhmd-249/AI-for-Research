"""The lens agent runtime: turns a Lens into a stateless LLM call and enforces
the sub-agent contract in code (tool-access scope, two-strike validation,
refusal, anonymization, trace emission). See ``run_lens``.
"""

from .inputs import (
    AnonymizedPeer,
    LensRunInput,
    PeerOutput,
    anonymize_peers,
    build_lens_prompt,
    build_prior_self,
    build_system_prompt,
)
from .llm import (
    AnthropicLlmClient,
    FakeLlmClient,
    LlmClient,
    LlmRequest,
    LlmResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TracingLlmClient,
)
from .run_lens import (
    DEFAULT_MODEL,
    LensRunOutcome,
    LensRunRefused,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    persist_outcome,
    run_lens,
)
from .tools import Tool, ToolRegistry, emit_output_tool_spec

__all__ = [
    "DEFAULT_MODEL",
    "AnonymizedPeer",
    "AnthropicLlmClient",
    "FakeLlmClient",
    "LensRunInput",
    "LensRunOutcome",
    "LensRunRefused",
    "LensRunSchemaInvalid",
    "LensRunSucceeded",
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "PeerOutput",
    "TextBlock",
    "Tool",
    "ToolRegistry",
    "ToolResultBlock",
    "ToolUseBlock",
    "TracingLlmClient",
    "anonymize_peers",
    "build_lens_prompt",
    "build_prior_self",
    "build_system_prompt",
    "emit_output_tool_spec",
    "persist_outcome",
    "run_lens",
]
