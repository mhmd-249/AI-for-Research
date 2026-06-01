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
    quarantine_block,
)
from .lens_tools import SourceBodyFetcher, SourceFetchTool, VerifierQueryTool
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
from .round2 import (
    MANDATORY_PARTICIPANTS,
    MIN_ROUND_1_SUCCESSES_FOR_ROUND_2,
    OPT_IN_HEURISTIC_PHRASE,
    Round2Result,
    Round2TaskBuilder,
    run_round_2,
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
    "MANDATORY_PARTICIPANTS",
    "MIN_ROUND_1_SUCCESSES_FOR_ROUND_2",
    "OPT_IN_HEURISTIC_PHRASE",
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
    "Round2Result",
    "Round2TaskBuilder",
    "SourceBodyFetcher",
    "SourceFetchTool",
    "TextBlock",
    "Tool",
    "ToolRegistry",
    "ToolResultBlock",
    "ToolUseBlock",
    "TracingLlmClient",
    "VerifierQueryTool",
    "anonymize_peers",
    "build_lens_prompt",
    "build_prior_self",
    "build_system_prompt",
    "emit_output_tool_spec",
    "persist_outcome",
    "quarantine_block",
    "run_lens",
    "run_round_1",
    "run_round_2",
]
