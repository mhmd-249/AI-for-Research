"""PanelRouter — hybrid lens-panel routing (Layer 3.2, stories 81-86).

Tests cover:

- Proposes a panel with one-sentence inclusion + exclusion reasoning for ALL nine
  lenses (story 81).
- Reasoning is per-brief-specific, not boilerplate (story 82): rationales come
  from the LLM call's emit_panel envelope, not a hardcoded template, so two
  briefs produce different text.
- ``panel_constraints`` overrides are honored (stories 86, acceptance criterion):
  if the brief carries pre-specified lenses, the proposal's panel field is
  exactly those, regardless of what the LLM proposes.
- Mode shifts defaults only; routing logic is identical (stories 83, 84): the
  same code path runs for both modes; the only difference is the prompt's
  target panel size.
- A panel edit during Screen 2 creates a new DispatchEvent without bumping
  brief version (story 85).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from research_council.domain.brief_ops import revise_brief
from research_council.enums import LensId, Mode
from research_council.ids import SequentialIdGenerator
from research_council.models import Brief
from research_council.router import (
    DEEP_DIVE_DEFAULT_SIZE,
    EXPLORATORY_DEFAULT_SIZE,
    PanelProposal,
    make_dispatch_event,
    propose_panel,
)
from research_council.runtime.llm import (
    FakeLlmClient,
    LlmResponse,
    ToolUseBlock,
)

# --- helpers ----------------------------------------------------------------


def emit_panel(
    rationales: list[dict[str, Any]], use_id: str = "p1"
) -> LlmResponse:
    return LlmResponse(
        content=[ToolUseBlock(id=use_id, name="emit_panel", input={"rationales": rationales})]
    )


def _rat(lens_id: LensId, included: bool, reasoning: str) -> dict[str, Any]:
    return {"lens_id": lens_id.value, "included": included, "reasoning": reasoning}


def _full_rationales(
    *, included: set[LensId], reasoning_by_lens: dict[LensId, str] | None = None
) -> list[dict[str, Any]]:
    """Build rationales for all nine lenses with realistic, distinct per-lens text."""
    text_by_lens = reasoning_by_lens or {}
    defaults = {
        LensId.PRIOR_ART: "Long-context attention has prior art to ground against.",
        LensId.ADVERSARIAL: "Filter-agent proposal has a clean attack surface (circularity).",
        LensId.EMPIRICAL_BENCHMARKING: "Evaluation needs baselines and ablations spelled out.",
        LensId.MECHANISTIC_INTERPRETABILITY: "Mid-context attention is mech-interp's home turf.",
        LensId.INFORMATION_THEORETIC: "Compression framings could reframe selective attention.",
        LensId.TRAINING_DATA_DISTRIBUTION: "Tangential to this architectural question.",
        LensId.DEPLOYMENT_SERVING: "Filter-agent inference cost matters at deployment.",
        LensId.ARCHITECTURE: "The proposal is fundamentally architectural.",
        LensId.FIRST_PRINCIPLES: "Useful: forces counterfactual reframing.",
    }
    return [
        _rat(lens_id, lens_id in included, text_by_lens.get(lens_id, defaults[lens_id]))
        for lens_id in LensId
    ]


_DEEP_DIVE_PANEL = {
    LensId.PRIOR_ART,
    LensId.ADVERSARIAL,
    LensId.EMPIRICAL_BENCHMARKING,
    LensId.MECHANISTIC_INTERPRETABILITY,
    LensId.INFORMATION_THEORETIC,
    LensId.DEPLOYMENT_SERVING,
    LensId.ARCHITECTURE,
    LensId.FIRST_PRINCIPLES,
}
_EXPLORATORY_PANEL = {
    LensId.PRIOR_ART,
    LensId.ADVERSARIAL,
    LensId.ARCHITECTURE,
    LensId.FIRST_PRINCIPLES,
}


# --- proposal & per-lens reasoning -------------------------------------------


async def test_proposes_rationale_for_all_nine_lenses(brief: Brief) -> None:
    client = FakeLlmClient([emit_panel(_full_rationales(included=_DEEP_DIVE_PANEL))])
    proposal = await propose_panel(brief, client=client)

    assert isinstance(proposal, PanelProposal)
    assert {r.lens_id for r in proposal.rationales} == set(LensId)
    assert all(r.reasoning.strip() for r in proposal.rationales)

    # Included rationales materialize as the proposal's panel.
    assert set(proposal.panel) == _DEEP_DIVE_PANEL
    included_in_rationales = {r.lens_id for r in proposal.rationales if r.included}
    assert included_in_rationales == set(proposal.panel)


async def test_excluded_lenses_have_exclusion_reasoning(brief: Brief) -> None:
    # Story 82: exclusion reasoning matters as much as inclusion reasoning.
    client = FakeLlmClient([emit_panel(_full_rationales(included=_EXPLORATORY_PANEL))])
    proposal = await propose_panel(brief, client=client)

    excluded = [r for r in proposal.rationales if not r.included]
    assert len(excluded) == len(LensId) - len(_EXPLORATORY_PANEL)
    for rationale in excluded:
        assert rationale.reasoning.strip(), f"missing exclusion reasoning for {rationale.lens_id}"


async def test_reasoning_is_llm_derived_not_boilerplate(brief: Brief) -> None:
    # Per-brief-specific text comes from the LLM; we round-trip whatever the
    # LLM emitted, proving the router doesn't substitute static template text.
    distinctive = "DISTINCTIVE-PER-BRIEF-TEXT: filter-agent circularity is the load-bearing risk."
    rationales = _full_rationales(
        included=_DEEP_DIVE_PANEL,
        reasoning_by_lens={LensId.ADVERSARIAL: distinctive},
    )
    client = FakeLlmClient([emit_panel(rationales)])
    proposal = await propose_panel(brief, client=client)

    adversarial = next(r for r in proposal.rationales if r.lens_id is LensId.ADVERSARIAL)
    assert adversarial.reasoning == distinctive


# --- panel_constraints override ---------------------------------------------


async def test_panel_constraints_override_LLM_panel(brief: Brief) -> None:
    # User has pre-specified a narrow panel of two lenses; even if the LLM
    # proposes more, the proposal's panel is exactly the constraints.
    constrained = brief.model_copy(
        update={"panel_constraints": (LensId.PRIOR_ART, LensId.ADVERSARIAL)}
    )
    client = FakeLlmClient([emit_panel(_full_rationales(included=_DEEP_DIVE_PANEL))])
    proposal = await propose_panel(constrained, client=client)

    assert proposal.panel == (LensId.PRIOR_ART, LensId.ADVERSARIAL)


async def test_panel_constraints_still_produce_reasoning_for_all_nine(brief: Brief) -> None:
    # Even with constraints, reasoning is produced for all nine so the user can
    # see why the constrained set excludes the others.
    constrained = brief.model_copy(update={"panel_constraints": (LensId.PRIOR_ART,)})
    client = FakeLlmClient(
        [emit_panel(_full_rationales(included={LensId.PRIOR_ART}))]
    )
    proposal = await propose_panel(constrained, client=client)

    assert {r.lens_id for r in proposal.rationales} == set(LensId)


# --- mode-aware defaults & identical routing logic --------------------------


async def test_mode_aware_default_sizes_are_distinct() -> None:
    # Story 83: exploratory defaults narrower (~4), deep_dive wider (~8).
    assert EXPLORATORY_DEFAULT_SIZE < DEEP_DIVE_DEFAULT_SIZE


async def test_mode_shifts_prompt_target_size_only(brief: Brief) -> None:
    """Routing logic is identical across modes; only the prompt's target size shifts.

    We assert that running the router on a deep_dive brief and on an exploratory
    brief (same content, different mode) hits the same single code path: one LLM
    call, the same advertised tool, the same overall structure. The only
    observable difference is the target size mentioned in the system prompt.
    """
    deep_brief = brief  # default Mode.DEEP_DIVE
    exploratory_brief = brief.model_copy(update={"mode": Mode.EXPLORATORY})

    deep_client = FakeLlmClient([emit_panel(_full_rationales(included=_DEEP_DIVE_PANEL))])
    exp_client = FakeLlmClient([emit_panel(_full_rationales(included=_EXPLORATORY_PANEL))])

    await propose_panel(deep_brief, client=deep_client)
    await propose_panel(exploratory_brief, client=exp_client)

    assert len(deep_client.requests) == 1
    assert len(exp_client.requests) == 1

    deep_req = deep_client.requests[0]
    exp_req = exp_client.requests[0]

    # Same tool advertised, same single tool — only the system prompt differs.
    deep_tools = {t.name for t in deep_req.tools}
    exp_tools = {t.name for t in exp_req.tools}
    assert deep_tools == exp_tools == {"emit_panel"}
    assert str(DEEP_DIVE_DEFAULT_SIZE) in deep_req.system
    assert str(EXPLORATORY_DEFAULT_SIZE) in exp_req.system


# --- DispatchEvent on panel edit (story 85) ---------------------------------


def test_panel_edit_creates_new_dispatch_event_without_brief_version_bump(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    proposed = (
        LensId.PRIOR_ART,
        LensId.ADVERSARIAL,
        LensId.ARCHITECTURE,
        LensId.FIRST_PRINCIPLES,
    )
    edited = (
        LensId.PRIOR_ART,
        LensId.ADVERSARIAL,
        LensId.MECHANISTIC_INTERPRETABILITY,
    )

    initial = make_dispatch_event(brief, proposed, id_generator=ids, now=now)
    later = make_dispatch_event(brief, edited, id_generator=ids, now=now)

    # Both events point at the SAME brief version — a panel edit is NOT a brief
    # edit, so it never bumps the brief.
    assert initial.brief_version == brief.version
    assert later.brief_version == brief.version
    assert initial.id != later.id  # distinct events
    assert initial.panel != later.panel
    # The brief object itself is untouched.
    assert brief.version == 1


def test_panel_constraints_edit_during_intake_is_brief_edit(brief: Brief) -> None:
    # Story 86: panel_constraints edited DURING intake is a brief edit. This
    # path goes through revise_brief (existing module), which we test here to
    # contrast with the dispatch-time path above — a constraints edit DOES bump
    # brief version.
    revised = revise_brief(brief, panel_constraints=(LensId.PRIOR_ART, LensId.ADVERSARIAL))
    assert revised.version == brief.version + 1
    assert revised.panel_constraints == (LensId.PRIOR_ART, LensId.ADVERSARIAL)


# --- schema enforcement ------------------------------------------------------


async def test_missing_rationale_for_a_lens_is_rejected(brief: Brief) -> None:
    # If the LLM omits one lens, the router rejects the envelope.
    incomplete = _full_rationales(included=_DEEP_DIVE_PANEL)[:-1]  # drop last lens
    client = FakeLlmClient([emit_panel(incomplete), emit_panel(incomplete)])
    with pytest.raises(ValueError):
        await propose_panel(brief, client=client)
