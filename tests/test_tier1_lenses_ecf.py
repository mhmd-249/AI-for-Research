"""Tier-1 lens validation on the ECF (Active Context Inhibition) brief.

The brief fixture in conftest is the ECF brief: long-context under-attention,
proposed filter-agent solution. The build-order gate (story 64) requires tier-1
output be trustworthy on this brief before tier-2 ships. These tests assert the
contract at the runtime boundary:

* the prior-art lens emits ``prior_art`` findings grounded in tool calls
  (``verifier_query`` + ``source_fetch``);
* the adversarial lens emits ``failure_mode`` findings in Round 1 *against the
  proposed solution* and Round 2 *against the emerging consensus*, with the
  system prompt reflecting the round-specific target;
* retrieved source text only enters lens context inside the labeled
  ``<untrusted_retrieved_content>`` quarantine block (story 159);
* both lenses produce schema-valid envelopes with a non-empty
  ``disagreements_with_my_own_framing`` list.
"""

from __future__ import annotations

from typing import Any

from research_council.enums import ClaimType, LensId, ToolName
from research_council.ids import SequentialIdGenerator
from research_council.lenses import get_lens_config
from research_council.models import Brief
from research_council.runtime import (
    AnonymizedPeer,
    LensRunInput,
    LensRunSucceeded,
    Tool,
    build_system_prompt,
    quarantine_block,
    run_lens,
)
from research_council.runtime.llm import FakeLlmClient, LlmResponse, ToolUseBlock
from research_council.runtime.tools import EMIT_OUTPUT_TOOL

# --- helpers ----------------------------------------------------------------


def _emit(payload: dict[str, Any], use_id: str = "u_final") -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name=EMIT_OUTPUT_TOOL, input=payload)])


def _tool_call(name: ToolName, use_id: str, payload: dict[str, Any]) -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name=name.value, input=payload)])


class _FakeSourceFetch:
    """Stand-in for the real ``source_fetch`` adapter — returns canned bodies
    wrapped in the quarantine block, exactly like the real tool would."""

    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> ToolName:
        return ToolName.SOURCE_FETCH

    @property
    def description(self) -> str:
        return "Fetch a paper body by canonical id; returns untrusted retrieved content."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"canonical_id": {"type": "string"}},
            "required": ["canonical_id"],
        }

    async def handle(self, tool_input: dict[str, Any]) -> str:
        self.calls.append(tool_input)
        canonical_id = str(tool_input["canonical_id"])
        body = self._bodies.get(canonical_id, "")
        if not body:
            return f"Source {canonical_id!r} not found."
        return quarantine_block(canonical_id, body)


class _FakeVerifierQuery:
    """Stand-in for the real ``verifier_query`` adapter — returns canned
    structured verdicts so the lens's tool-use path is exercised end-to-end."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self._verdicts = verdicts
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> ToolName:
        return ToolName.VERIFIER_QUERY

    @property
    def description(self) -> str:
        return "Verify a claim against the literature."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string"},
                "claim_type": {"type": "string"},
            },
            "required": ["claim_text", "claim_type"],
        }

    async def handle(self, tool_input: dict[str, Any]) -> str:
        self.calls.append(tool_input)
        return self._verdicts.get(
            str(tool_input.get("claim_text", "")), "status: source_not_found"
        )


# --- Prior-art lens on the ECF brief ----------------------------------------


PRIOR_ART_OUTPUT: dict[str, Any] = {
    "findings": [
        {
            "claim_text": (
                "Liu et al. 2023 ('Lost in the Middle') already characterizes the "
                "U-shape that motivates Active Context Inhibition; the brief does "
                "not cite it."
            ),
            "claim_type": "prior_art",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": (
                "If Lost-in-the-Middle were cited in researcher_context, this "
                "finding would collapse."
            ),
        },
        {
            "claim_text": (
                "Retrieval-augmented architectures (RAG) tackle the same selection "
                "problem; the brief does not engage with their results."
            ),
            "claim_type": "prior_art",
            "confidence": "supporting",
            "failure_modes_if_wrong": "If the brief did engage RAG, this duplicates.",
        },
    ],
    "open_questions": ["Has anyone tried a learned filter on long-context summarization?"],
    "disagreements_with_my_own_framing": [
        "I may be over-weighting cited prior art and under-weighting the brief's "
        "stated novelty contribution."
    ],
    "brief_summary": (
        "ECF proposal: add a filter-agent stage to prune the context window so the "
        "downstream model attends only to selected passages."
    ),
}


async def test_prior_art_lens_on_ecf_emits_prior_art_findings_grounded_in_tools(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    fetch = _FakeSourceFetch(
        bodies={
            "arxiv:2307.03172": (
                "Lost in the Middle: language models show a U-shaped accuracy "
                "curve across long-context positions."
            ),
        }
    )
    verifier_query = _FakeVerifierQuery(
        verdicts={
            "Lost in the Middle characterizes positional bias.": (
                "status: verified\n"
                "source_canonical_id: arxiv:2307.03172\n"
                "quoted_passage:\n"
                + quarantine_block(
                    "arxiv:2307.03172",
                    "U-shaped accuracy curve across long-context positions",
                )
            ),
        }
    )
    registry: dict[ToolName, Tool] = {
        ToolName.SOURCE_FETCH: fetch,
        ToolName.VERIFIER_QUERY: verifier_query,
    }

    inputs = LensRunInput(
        brief=brief,
        lens_config=get_lens_config(LensId.PRIOR_ART),
        round=1,
    )
    client = FakeLlmClient(
        [
            _tool_call(
                ToolName.VERIFIER_QUERY,
                "t_verify",
                {
                    "claim_text": "Lost in the Middle characterizes positional bias.",
                    "claim_type": "prior_art",
                    "arxiv_id": "2307.03172",
                },
            ),
            _tool_call(ToolName.SOURCE_FETCH, "t_fetch", {"canonical_id": "arxiv:2307.03172"}),
            _emit(PRIOR_ART_OUTPUT),
        ]
    )

    outcome = await run_lens(
        inputs=inputs, client=client, id_generator=ids, tools=registry
    )

    assert isinstance(outcome, LensRunSucceeded)
    # All emitted findings carry the prior_art claim_type — the lens's role.
    assert {f.claim_type for f in outcome.findings} == {ClaimType.PRIOR_ART}
    # Both tools were actually used (the findings are grounded in tool results,
    # not bare model knowledge).
    assert fetch.calls == [{"canonical_id": "arxiv:2307.03172"}]
    assert len(verifier_query.calls) == 1
    # Acceptance criterion: non-empty disagreements list.
    assert outcome.disagreements_with_my_own_framing

    # Story 159: the retrieved source body reached the model ONLY inside the
    # labeled quarantine block. The body string never appears un-wrapped in the
    # accumulated transcript.
    body = "U-shaped accuracy curve across long-context positions"
    last_request = client.requests[-1]
    body_blocks: list[str] = []
    for message in last_request.messages:
        for block in message.content:
            text = getattr(block, "text", None) or getattr(block, "content", None)
            if text and body in text:
                body_blocks.append(text)
    assert body_blocks, "the retrieved body must have entered the conversation"
    for chunk in body_blocks:
        assert "<untrusted_retrieved_content" in chunk
        assert "</untrusted_retrieved_content>" in chunk


# --- Adversarial lens, Round 1 vs Round 2 ------------------------------------


ADVERSARIAL_R1_OUTPUT: dict[str, Any] = {
    "findings": [
        {
            "claim_text": (
                "A learned filter doubles inference cost end-to-end; the brief omits "
                "the budget impact."
            ),
            "claim_type": "failure_mode",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": (
                "If the filter runs in a cheap pass-only mode, the cost objection weakens."
            ),
        }
    ],
    "open_questions": ["Has anyone measured filter latency under realistic batch sizes?"],
    "disagreements_with_my_own_framing": [
        "If the brief targets exploratory benchmarks rather than production, the "
        "cost objection has less force."
    ],
    "brief_summary": (
        "Attack: the filter agent in the ACI proposal pushes inference cost without "
        "addressing the underlying positional bias."
    ),
}


ADVERSARIAL_R2_OUTPUT: dict[str, Any] = {
    "findings": [
        {
            "claim_text": (
                "The emerging consensus assumes the filter signal can be supervised "
                "cheaply; Lens A and Lens B both rely on this without evidence."
            ),
            "claim_type": "failure_mode",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": (
                "If supervision turns out to be free (e.g. self-distillation), the "
                "consensus is fine."
            ),
            "responding_to": ["Lens A", "Lens B"],
        }
    ],
    "open_questions": [],
    "disagreements_with_my_own_framing": [
        "Lenses A and B may have evidence I haven't surfaced."
    ],
    "brief_summary": (
        "Attack: the council's consensus on a learned filter rests on an "
        "unverified assumption."
    ),
}


async def test_adversarial_lens_round_1_targets_proposed_solution(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    inputs = LensRunInput(
        brief=brief,
        lens_config=get_lens_config(LensId.ADVERSARIAL),
        round=1,
    )
    client = FakeLlmClient([_emit(ADVERSARIAL_R1_OUTPUT)])
    outcome = await run_lens(inputs=inputs, client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSucceeded)
    # Round 1 critic emits failure_mode findings.
    assert {f.claim_type for f in outcome.findings} == {ClaimType.FAILURE_MODE}
    assert outcome.disagreements_with_my_own_framing

    # Round 1 system prompt explicitly names the proposed solution as target.
    system = client.requests[-1].system
    assert "proposed solution" in system.lower()
    assert "emerging" not in system.lower()


async def test_adversarial_lens_round_2_targets_emerging_consensus(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    peers = [
        AnonymizedPeer(
            label="Lens A",
            findings=[],
            brief_summary="Round 1 summary from Lens A.",
            open_questions=[],
        ),
        AnonymizedPeer(
            label="Lens B",
            findings=[],
            brief_summary="Round 1 summary from Lens B.",
            open_questions=[],
        ),
    ]
    inputs = LensRunInput(
        brief=brief,
        lens_config=get_lens_config(LensId.ADVERSARIAL),
        round=2,
        cross_pollination=peers,
    )
    client = FakeLlmClient([_emit(ADVERSARIAL_R2_OUTPUT)])
    outcome = await run_lens(inputs=inputs, client=client, id_generator=ids)

    assert isinstance(outcome, LensRunSucceeded)
    # Round 2 still emits failure_mode findings (the lens's role doesn't change).
    assert {f.claim_type for f in outcome.findings} == {ClaimType.FAILURE_MODE}
    # ``responding_to`` is populated when the finding engages a peer (story 99).
    [finding] = outcome.findings
    assert finding.responding_to == ("Lens A", "Lens B")

    # Round 2 system prompt names the emerging Round 1 consensus, not the
    # original proposed solution.
    system = client.requests[-1].system
    assert "emerging" in system.lower()
    assert "proposed solution" not in system.lower()


# --- Tool-access guard: adversarial never sees verifier_query --------------


async def test_adversarial_lens_does_not_advertise_verifier_query(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # Even when a verifier_query tool is in the registry, the adversarial lens
    # doesn't carry that grant — the advertised tool list omits it.
    fetch = _FakeSourceFetch(bodies={})
    verifier_query = _FakeVerifierQuery(verdicts={})
    registry: dict[ToolName, Tool] = {
        ToolName.SOURCE_FETCH: fetch,
        ToolName.VERIFIER_QUERY: verifier_query,
    }

    inputs = LensRunInput(
        brief=brief,
        lens_config=get_lens_config(LensId.ADVERSARIAL),
        round=1,
    )
    client = FakeLlmClient([_emit(ADVERSARIAL_R1_OUTPUT)])
    await run_lens(inputs=inputs, client=client, id_generator=ids, tools=registry)

    advertised = {t.name for req in client.requests for t in req.tools}
    assert "verifier_query" not in advertised
    assert "source_fetch" in advertised


# --- Round-aware system prompt — wired through run_lens --------------------


def test_run_lens_passes_round_into_system_prompt_for_prior_art(
    brief: Brief,
) -> None:
    # Story 95: prior-art's Round 2 framing differs from Round 1.
    config = get_lens_config(LensId.PRIOR_ART)
    r1 = build_system_prompt(config, brief.mode.value, round=1)
    r2 = build_system_prompt(config, brief.mode.value, round=2)
    assert r1 != r2
    assert "Round: 1" in r1
    assert "Round: 2" in r2
