"""The run_lens loop: structured output, two-strike validation, refusal,
tool-access enforcement, claim-type restriction, and trace emission (stories
42-53, 63)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from research_council.enums import LensId, LensRunStatus, ToolName, VerificationStatus
from research_council.ids import SequentialIdGenerator
from research_council.lenses import get_lens_config
from research_council.models import Brief
from research_council.runtime import (
    LensRunInput,
    LensRunRefused,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    Tool,
    TracingLlmClient,
    persist_outcome,
    run_lens,
)
from research_council.runtime.llm import FakeLlmClient, LlmResponse, ToolUseBlock
from research_council.store import SessionStoreAndTrace

# --- helpers ----------------------------------------------------------------

VALID_FIRST_PRINCIPLES: dict[str, Any] = {
    "findings": [
        {
            "claim_text": "The minimal mechanism is selective gating, not a separate filter agent.",
            "claim_type": "mechanism_hypothesis",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": "If gating cannot be learned end-to-end, this fails.",
        },
        {
            "claim_text": "The proposal does not address how the filter avoids the same failure.",
            "claim_type": "gap",
            "confidence": "supporting",
            "failure_modes_if_wrong": "If the filter is a cheap heuristic, circularity weakens.",
        },
    ],
    "open_questions": ["Can the gate be learned without supervision?"],
    "disagreements_with_my_own_framing": ["I may be discounting real engineering constraints."],
    "brief_summary": "The proposal adds a filter agent to prune the context window.",
}


def emit(payload: dict[str, Any], use_id: str = "u1") -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name="emit_output", input=payload)])


def tool_call(name: str, use_id: str, payload: dict[str, Any] | None = None) -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name=name, input=payload or {})])


def fp_inputs(brief: Brief) -> LensRunInput:
    return LensRunInput(brief=brief, lens_config=get_lens_config(LensId.FIRST_PRINCIPLES), round=1)


# --- success & materialization ----------------------------------------------


async def test_successful_run_materializes_findings(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    client = FakeLlmClient([emit(VALID_FIRST_PRINCIPLES)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSucceeded)
    assert len(outcome.findings) == 2
    for finding in outcome.findings:
        assert finding.round == 1
        assert finding.verification_status is VerificationStatus.UNVERIFIED
        assert finding.id.startswith("finding_")


# --- two-strike validation --------------------------------------------------


async def test_invalid_then_valid_succeeds_on_retry(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    bad = {**VALID_FIRST_PRINCIPLES, "disagreements_with_my_own_framing": []}  # story 45 violation
    client = FakeLlmClient([emit(bad), emit(VALID_FIRST_PRINCIPLES)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSucceeded)
    assert len(client.requests) == 2  # one retry consumed


async def test_invalid_twice_persists_schema_invalid(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    bad = {**VALID_FIRST_PRINCIPLES, "disagreements_with_my_own_framing": []}
    client = FakeLlmClient([emit(bad), emit(bad)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSchemaInvalid)
    assert "disagreements" in outcome.validation_error


async def test_partial_validity_is_rejected_whole(brief: Brief, ids: SequentialIdGenerator) -> None:
    # One good finding, one with a bad enum: the whole envelope is invalid.
    mixed = {
        **VALID_FIRST_PRINCIPLES,
        "findings": [
            VALID_FIRST_PRINCIPLES["findings"][0],
            {**VALID_FIRST_PRINCIPLES["findings"][1], "claim_type": "not_a_type"},
        ],
    }
    client = FakeLlmClient([emit(mixed), emit(mixed)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)
    assert isinstance(outcome, LensRunSchemaInvalid)


# --- claim-type restriction (first-principles, story 63) --------------------


async def test_first_principles_rejects_disallowed_claim_type(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    empirical = {
        **VALID_FIRST_PRINCIPLES,
        "findings": [
            {
                "claim_text": "Paper X reports 92% on benchmark Y.",
                "claim_type": "empirical",  # forbidden for first-principles
                "confidence": "supporting",
                "failure_modes_if_wrong": "...",
            }
        ],
    }
    client = FakeLlmClient([emit(empirical), emit(empirical)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSchemaInvalid)
    assert "claim_type" in outcome.validation_error


# --- refusal ----------------------------------------------------------------


async def test_refusal_is_a_terminal_outcome(brief: Brief, ids: SequentialIdGenerator) -> None:
    client = FakeLlmClient(
        [emit({"refused": True, "refusal_reason": "This brief is outside my frame."})]
    )
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)
    assert isinstance(outcome, LensRunRefused)
    assert outcome.reason == "This brief is outside my frame."


# --- tool-access enforcement (story 47) -------------------------------------


class _FakeSourceFetch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> ToolName:
        return ToolName.SOURCE_FETCH

    @property
    def description(self) -> str:
        return "Fetch a source by id."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"id": {"type": "string"}}}

    async def handle(self, tool_input: dict[str, Any]) -> str:
        self.calls.append(tool_input)
        return "FETCHED"


async def test_granted_tool_dispatched_ungranted_refused(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # The adversarial lens grants source_fetch but NOT verifier_query.
    inputs = LensRunInput(brief=brief, lens_config=get_lens_config(LensId.ADVERSARIAL), round=1)
    fetch = _FakeSourceFetch()
    registry: dict[ToolName, Tool] = {ToolName.SOURCE_FETCH: fetch}

    adversarial_output = {
        "findings": [
            {
                "claim_text": "The consensus ignores the deployment cost.",
                "claim_type": "failure_mode",
                "confidence": "load_bearing",
                "failure_modes_if_wrong": "If cost is negligible, the objection is weak.",
            }
        ],
        "disagreements_with_my_own_framing": ["I might be over-weighting serving cost."],
        "brief_summary": "Attack on the proposed solution.",
    }

    client = FakeLlmClient(
        [
            tool_call("source_fetch", "t1", {"id": "arxiv:2307.03172"}),  # granted -> dispatched
            tool_call("verifier_query", "t2", {"claim": "x"}),  # ungranted -> refused
            emit(adversarial_output, use_id="t3"),
        ]
    )
    outcome = await run_lens(inputs=inputs, client=client, id_generator=ids, tools=registry)

    assert isinstance(outcome, LensRunSucceeded)
    assert fetch.calls == [{"id": "arxiv:2307.03172"}]  # granted tool ran exactly once

    # verifier_query was never advertised...
    advertised_names = {t.name for req in client.requests for t in req.tools}
    assert "verifier_query" not in advertised_names
    assert {"source_fetch", "emit_output"} <= advertised_names

    # ...and the ungranted call was answered with an error tool_result, not dispatched.
    last_request = client.requests[-1]
    error_results = [
        block
        for message in last_request.messages
        for block in message.content
        if getattr(block, "is_error", False)
    ]
    assert any("verifier_query" in getattr(b, "content", "") for b in error_results)


# --- trace emission (story 163) ---------------------------------------------


async def test_every_llm_call_emits_a_trace_record(
    brief: Brief, ids: SequentialIdGenerator, store: SessionStoreAndTrace
) -> None:
    fixed = datetime(2026, 6, 1, tzinfo=UTC)
    bad = {**VALID_FIRST_PRINCIPLES, "disagreements_with_my_own_framing": []}
    inner = FakeLlmClient([emit(bad), emit(VALID_FIRST_PRINCIPLES)])  # two LLM calls
    client = TracingLlmClient(inner, store, SequentialIdGenerator(), clock=lambda: fixed)

    outcome = await run_lens(
        inputs=fp_inputs(brief), client=client, id_generator=ids, caller_ref="lensrun_42"
    )

    assert isinstance(outcome, LensRunSucceeded)
    traces = store.list_traces("lensrun_42")
    assert len(traces) == 2
    assert all(t.model == "claude-opus-4-8" for t in traces)


# --- the tracer spine: persist a schema-valid first-principles run -----------


async def test_tracer_spine_persists_valid_output(
    brief: Brief, ids: SequentialIdGenerator, store: SessionStoreAndTrace
) -> None:
    client = FakeLlmClient([emit(VALID_FIRST_PRINCIPLES)])
    outcome = await run_lens(inputs=fp_inputs(brief), client=client, id_generator=ids)

    run = persist_outcome(
        store,
        outcome=outcome,
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=brief.session_id,
        brief_version=1,
        round=1,
        id_generator=ids,
    )

    assert run.status is LensRunStatus.SUCCEEDED
    persisted = store.list_findings_for_lens_run(run.id)
    assert len(persisted) == 2
    assert {f.claim_type.value for f in persisted} <= {"mechanism_hypothesis", "gap"}
