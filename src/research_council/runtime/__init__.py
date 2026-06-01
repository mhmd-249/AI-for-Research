"""The lens agent runtime: turns a Lens into a stateless LLM call and enforces
the sub-agent contract in code (tool-access scope, two-strike validation,
refusal, anonymization, trace emission). See ``run_lens``.

Round 1 dispatch lives here too: the Round1Controller (``run_round_1``) takes a
panel + prepared per-lens tasks and runs them in parallel under both timeouts.
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
from .round1 import (
    DEFAULT_FAILURE_HALT_THRESHOLD,
    DEFAULT_HARD_TIMEOUT_SECONDS,
    DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS,
    LensTask,
    ProgressCallback,
    Round1Result,
    run_round_1,
)
from .run_lens import (
    DEFAULT_MODEL,
    LensRunOutcome,
    LensRunRefused,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    LensRunTimeout,
    persist_outcome,
    run_lens,
)
from .tools import Tool, ToolRegistry, emit_output_tool_spec

__all__ = [
    "DEFAULT_FAILURE_HALT_THRESHOLD",
    "DEFAULT_HARD_TIMEOUT_SECONDS",
    "DEFAULT_MODEL",
    "DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS",
    "AnonymizedPeer",
    "AnthropicLlmClient",
    "FakeLlmClient",
    "LensRunInput",
    "LensRunOutcome",
    "LensRunRefused",
    "LensRunSchemaInvalid",
    "LensRunSucceeded",
    "LensRunTimeout",
    "LensTask",
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "PeerOutput",
    "ProgressCallback",
    "Round1Result",
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
    "run_round_1",
]
