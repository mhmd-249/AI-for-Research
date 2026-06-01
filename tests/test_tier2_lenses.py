"""Tier-2 lens behavior (issue #5, PRD stories 57-62): the six remaining lenses
run as interchangeable panel peers through the generic runtime, with the two
per-lens constraints the PRD names as code-enforced:

* Mechanistic-interpretability (story 58): output is mechanism_hypothesis and
  failure_mode only, no tools.
* Information-theoretic (story 59): every finding makes at least one quantitative
  OR conditional prediction — the anti-decoration constraint that keeps the lens
  from drifting into pure reframings.

Tool-access enforcement is exercised generically in ``test_run_lens``; here we
parameterize each tier-2 lens through ``run_lens`` once to confirm the runtime
is lens-agnostic and dispatches all nine.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_council.enums import LensId, ToolName
from research_council.ids import SequentialIdGenerator
from research_council.lenses import get_lens_config
from research_council.models import Brief
from research_council.runtime import (
    LensRunInput,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    Tool,
    run_lens,
)
from research_council.runtime.llm import FakeLlmClient, LlmResponse, ToolUseBlock


class _StubTool:
    """A no-op tool with the registered ``ToolName`` — used by the
    tool-advertising test to confirm the runner advertises grants that have a
    registered handler."""

    def __init__(self, name: ToolName) -> None:
        self._name = name

    @property
    def name(self) -> ToolName:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub for {self._name.value}."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def handle(self, tool_input: dict[str, Any]) -> str:
        return ""


def _full_registry() -> dict[ToolName, Tool]:
    return {name: _StubTool(name) for name in ToolName}


def _emit(payload: dict[str, Any], use_id: str = "u1") -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name="emit_output", input=payload)])


def _inputs(brief: Brief, lens_id: LensId) -> LensRunInput:
    return LensRunInput(brief=brief, lens_config=get_lens_config(lens_id), round=1)


# --- mechanistic-interpretability: claim-type restriction (story 58) ----------

_VALID_MECH_INTERP: dict[str, Any] = {
    "findings": [
        {
            "claim_text": (
                "Mid-context under-attention is the induction head failing to fire when the "
                "query token is far from its key."
            ),
            "claim_type": "mechanism_hypothesis",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": "If induction heads fire normally, this hypothesis is dead.",
        },
        {
            "claim_text": (
                "Adding a filter agent doesn't fix the underlying circuit; it just adds noise "
                "the downstream MLP must learn to ignore."
            ),
            "claim_type": "failure_mode",
            "confidence": "supporting",
            "failure_modes_if_wrong": "If the filter signal is clean, the MLP need not adapt.",
        },
    ],
    "disagreements_with_my_own_framing": ["The circuit story may be too coarse-grained."],
    "brief_summary": (
        "Active context inhibition framed as a filter-agent gate over the context window."
    ),
}


async def test_mech_interp_succeeds_on_allowed_claim_types(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    client = FakeLlmClient([_emit(_VALID_MECH_INTERP)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.MECHANISTIC_INTERPRETABILITY),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSucceeded)
    assert {f.claim_type.value for f in outcome.findings} <= {
        "mechanism_hypothesis",
        "failure_mode",
    }


async def test_mech_interp_rejects_prior_art_claims(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    bad = {
        **_VALID_MECH_INTERP,
        "findings": [
            {
                "claim_text": "Liu et al. 2023 reports the same phenomenon.",
                "claim_type": "prior_art",  # explicitly disallowed by story 58
                "confidence": "supporting",
                "failure_modes_if_wrong": "...",
            }
        ],
    }
    client = FakeLlmClient([_emit(bad), _emit(bad)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.MECHANISTIC_INTERPRETABILITY),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSchemaInvalid)
    assert "claim_type" in outcome.validation_error
    assert "prior_art" in outcome.validation_error


async def test_mech_interp_advertises_no_tools(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    client = FakeLlmClient([_emit(_VALID_MECH_INTERP)])
    await run_lens(
        inputs=_inputs(brief, LensId.MECHANISTIC_INTERPRETABILITY),
        client=client,
        id_generator=ids,
    )
    advertised = {t.name for t in client.requests[0].tools}
    assert advertised == {"emit_output"}


# --- information-theoretic: quantitative-OR-conditional prediction (story 59) -

_QUANT_FINDING: dict[str, Any] = {
    # Quantitative: the prediction names a number / scaling.
    "claim_text": (
        "Channel capacity from query to mid-context is bounded by O(log n) attention bits "
        "per head; the proposed filter increases this by less than 1 bit at sequence length 8k."
    ),
    "claim_type": "mechanism_hypothesis",
    "confidence": "supporting",
    "failure_modes_if_wrong": "If capacity is not the binding constraint, the prediction misses.",
}

_CONDITIONAL_FINDING: dict[str, Any] = {
    # Conditional: explicit "if ... then" prediction.
    "claim_text": (
        "If mutual information between query and mid-context grows sub-logarithmically with "
        "depth, then deeper models will degrade faster on this task, not slower."
    ),
    "claim_type": "gap",
    "confidence": "exploratory",
    "failure_modes_if_wrong": "The conditional may hold only on synthetic data.",
}

_INFO_THEORETIC_BASE: dict[str, Any] = {
    "disagreements_with_my_own_framing": [
        "Channel-capacity framing may obscure path-dependence in attention."
    ],
    "brief_summary": (
        "Mid-context under-attention reframed as a channel-capacity bottleneck on the "
        "query-to-key path."
    ),
}


async def test_info_theoretic_succeeds_with_quantitative_prediction(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    payload = {**_INFO_THEORETIC_BASE, "findings": [_QUANT_FINDING]}
    client = FakeLlmClient([_emit(payload)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.INFORMATION_THEORETIC),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSucceeded)


async def test_info_theoretic_succeeds_with_conditional_prediction(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    payload = {**_INFO_THEORETIC_BASE, "findings": [_CONDITIONAL_FINDING]}
    client = FakeLlmClient([_emit(payload)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.INFORMATION_THEORETIC),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSucceeded)


async def test_info_theoretic_rejects_pure_reframe(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # No digits and no conditional marker — a pure information-theoretic
    # reframing with no testable content. Story 59 calls this "decoration".
    pure_reframe = {
        "claim_text": (
            "Mid-context under-attention can be viewed as a channel-capacity bottleneck "
            "on the query-to-key information pathway."
        ),
        "claim_type": "mechanism_hypothesis",
        "confidence": "exploratory",
        "failure_modes_if_wrong": "May simply restate the problem.",
    }
    payload = {**_INFO_THEORETIC_BASE, "findings": [pure_reframe]}
    client = FakeLlmClient([_emit(payload), _emit(payload)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.INFORMATION_THEORETIC),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSchemaInvalid)
    assert "quantitative" in outcome.validation_error.lower()


async def test_info_theoretic_rejects_when_any_finding_is_a_pure_reframe(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # Partial-validity rule (story 51): one offender invalidates the envelope.
    pure_reframe = {
        "claim_text": "Compression of the context is just a re-encoding of attention.",
        "claim_type": "mechanism_hypothesis",
        "confidence": "exploratory",
        "failure_modes_if_wrong": "...",
    }
    payload = {**_INFO_THEORETIC_BASE, "findings": [_QUANT_FINDING, pure_reframe]}
    client = FakeLlmClient([_emit(payload), _emit(payload)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.INFORMATION_THEORETIC),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSchemaInvalid)


async def test_info_theoretic_retry_can_recover(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    bad = {
        **_INFO_THEORETIC_BASE,
        "findings": [
            {
                "claim_text": "The problem is essentially one of entropy.",
                "claim_type": "mechanism_hypothesis",
                "confidence": "exploratory",
                "failure_modes_if_wrong": "...",
            }
        ],
    }
    good = {**_INFO_THEORETIC_BASE, "findings": [_QUANT_FINDING]}
    client = FakeLlmClient([_emit(bad), _emit(good)])
    outcome = await run_lens(
        inputs=_inputs(brief, LensId.INFORMATION_THEORETIC),
        client=client,
        id_generator=ids,
    )
    assert isinstance(outcome, LensRunSucceeded)
    assert len(client.requests) == 2  # one retry consumed


# --- the six tier-2 lenses run through the generic runtime --------------------

# A schema-valid envelope each tier-2 lens can return. The information-theoretic
# entry carries a quantitative claim; mechanistic-interpretability uses the two
# allowed claim types; the others use a generator+critic mix.

_TIER2_VALID_OUTPUTS: dict[LensId, dict[str, Any]] = {
    LensId.EMPIRICAL_BENCHMARKING: {
        "findings": [
            {
                "claim_text": (
                    "The brief proposes no ablation isolating the filter agent's effect from the "
                    "model's own attention recovery; without one, success is unattributable."
                ),
                "claim_type": "gap",
                "confidence": "load_bearing",
                "failure_modes_if_wrong": (
                    "If the filter ablation is in the appendix this objection dies on the spot."
                ),
            }
        ],
        "disagreements_with_my_own_framing": [
            "The benchmark I'd ask for may not exist yet."
        ],
        "brief_summary": (
            "Filter agent for mid-context attention recovery; eval design under-specified."
        ),
    },
    LensId.MECHANISTIC_INTERPRETABILITY: _VALID_MECH_INTERP,
    LensId.INFORMATION_THEORETIC: {
        **_INFO_THEORETIC_BASE,
        "findings": [_QUANT_FINDING, _CONDITIONAL_FINDING],
    },
    LensId.TRAINING_DATA_DISTRIBUTION: {
        "findings": [
            {
                "claim_text": (
                    "Pre-training corpora are top-and-tail-biased: mid-document salience is "
                    "rare, so the filter agent has no in-distribution supervision signal."
                ),
                "claim_type": "mechanism_hypothesis",
                "confidence": "supporting",
                "failure_modes_if_wrong": (
                    "If long mid-document QA datasets are common in the mix this collapses."
                ),
            }
        ],
        "disagreements_with_my_own_framing": [
            "I might be over-indexing on public corpus shape."
        ],
        "brief_summary": "Mid-context under-attention as a training-distribution artifact.",
    },
    LensId.DEPLOYMENT_SERVING: {
        "findings": [
            {
                "claim_text": (
                    "A separate filter pass doubles the KV-cache footprint and adds a sequential "
                    "stage to the critical path; the brief does not mention this."
                ),
                "claim_type": "failure_mode",
                "confidence": "load_bearing",
                "failure_modes_if_wrong": (
                    "If the filter shares the KV cache with the base model, the cost is "
                    "closer to 1.1x."
                ),
            }
        ],
        "disagreements_with_my_own_framing": [
            "The serving cost may be acceptable for the target deployment."
        ],
        "brief_summary": "Filter-agent approach with serving cost the brief omits.",
    },
    LensId.ARCHITECTURE: {
        "findings": [
            {
                "claim_text": (
                    "A position-encoding change such as ALiBi or rotary remapping addresses the "
                    "same failure with no second pass; the brief does not consider it."
                ),
                "claim_type": "mechanism_hypothesis",
                "confidence": "supporting",
                "failure_modes_if_wrong": (
                    "If position encodings have been tuned already, this slot is closed."
                ),
            }
        ],
        "disagreements_with_my_own_framing": [
            "The architectural space is large; I picked one cell."
        ],
        "brief_summary": (
            "Filter-agent for mid-context under-attention, evaluated against architectural "
            "alternatives."
        ),
    },
}


@pytest.mark.parametrize("lens_id", list(_TIER2_VALID_OUTPUTS))
async def test_each_tier2_lens_runs_through_runtime(
    lens_id: LensId, brief: Brief, ids: SequentialIdGenerator
) -> None:
    """All six tier-2 lenses run end-to-end through ``run_lens`` and produce a
    schema-valid LensRunSucceeded — the "interchangeable panel peer" acceptance
    criterion."""
    client = FakeLlmClient([_emit(_TIER2_VALID_OUTPUTS[lens_id])])
    outcome = await run_lens(
        inputs=_inputs(brief, lens_id), client=client, id_generator=ids
    )
    assert isinstance(outcome, LensRunSucceeded)


@pytest.mark.parametrize("lens_id", list(_TIER2_VALID_OUTPUTS))
async def test_tier2_lens_tools_advertised_match_roster(
    lens_id: LensId, brief: Brief, ids: SequentialIdGenerator
) -> None:
    """The runner advertises exactly the lens's granted tools plus emit_output —
    per-lens tool-access flags enforced by the runtime, not by the prompt.

    A registry covering every ToolName is passed so empirical-benchmarking's
    grants survive the runner's "must have a handler" intersection; lenses with
    no grants still see only ``emit_output``."""
    client = FakeLlmClient([_emit(_TIER2_VALID_OUTPUTS[lens_id])])
    await run_lens(
        inputs=_inputs(brief, lens_id),
        client=client,
        id_generator=ids,
        tools=_full_registry(),
    )

    config = get_lens_config(lens_id)
    advertised = {t.name for t in client.requests[0].tools}
    expected = {t.value for t in config.tool_access} | {"emit_output"}
    assert advertised == expected
