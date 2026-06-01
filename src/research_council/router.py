"""PanelRouter — hybrid lens-panel routing (Layer 3.2, stories 81-86).

Proposes a panel with one-sentence inclusion/exclusion reasoning for ALL nine
lenses (story 81). Reasoning is per-brief-specific (story 82) because it comes
from the LLM's emit_panel envelope, not from a hardcoded template. Mode-aware
defaults shift the prompt's target panel size but the routing logic is identical
across modes (stories 83, 84).

If the brief carries ``panel_constraints``, those override the LLM's inclusion
set (story 86's contrast: pre-intake constraints are brief edits and pre-empt
hybrid routing). Reasoning is still produced for all nine lenses so Screen 2
can show why the constrained set excludes the others.

A Screen-2 panel edit creates a new DispatchEvent without bumping brief version
(story 85): see :func:`make_dispatch_event`. The brief content is unchanged,
only the dispatch configuration changes.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import Field, ValidationError

from .enums import LensId, Mode
from .ids import IdGenerator, new_dispatch_event_id
from .models import Brief, DispatchEvent, FrozenModel
from .runtime.llm import LlmClient, LlmRequest, Message, TextBlock, ToolSpec

# Mode-aware defaults: only the prompt's target size shifts. Routing logic is
# identical across modes (story 84). The LLM is free to deviate; the user
# confirms or edits during Screen 2.
EXPLORATORY_DEFAULT_SIZE = 4
DEEP_DIVE_DEFAULT_SIZE = 8

EMIT_PANEL_TOOL = "emit_panel"
DEFAULT_MODEL = "claude-opus-4-8"
MAX_EMIT_ATTEMPTS = 2  # two-strike, matching run_lens


class LensRationale(FrozenModel):
    """One-sentence per-lens routing rationale (story 82). Produced for every
    lens, included and excluded alike."""

    lens_id: LensId
    included: bool
    reasoning: str


class PanelProposalDraft(FrozenModel):
    """What the LLM emits via ``emit_panel``: a flat list of per-lens rationales.

    Draft envelopes use ``list`` (mutable) per the project's draft-vs-persisted
    convention; the persisted :class:`PanelProposal` uses tuples.
    """

    rationales: list[LensRationale] = Field(default_factory=list)


class PanelProposal(FrozenModel):
    """The router's output (story 81). ``panel`` is the inclusion set the user
    sees on Screen 2; ``rationales`` carries one-sentence reasoning for every
    lens (included and excluded) so the user can spot wrong calls."""

    panel: tuple[LensId, ...]
    rationales: tuple[LensRationale, ...]


# --- Hybrid routing ---------------------------------------------------------


def _emit_panel_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rationales": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lens_id": {"type": "string", "enum": [lens.value for lens in LensId]},
                        "included": {"type": "boolean"},
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "One sentence, specific to THIS brief — not boilerplate. "
                                "Exclusion reasoning matters as much as inclusion."
                            ),
                        },
                    },
                    "required": ["lens_id", "included", "reasoning"],
                },
            }
        },
        "required": ["rationales"],
    }


def _target_panel_size(mode: Mode) -> int:
    return EXPLORATORY_DEFAULT_SIZE if mode is Mode.EXPLORATORY else DEEP_DIVE_DEFAULT_SIZE


def build_router_system_prompt(mode: Mode) -> str:
    """The router system prompt. Mode shifts the target size only; the rest of
    the instructions are identical across modes (story 84)."""
    target = _target_panel_size(mode)
    return (
        "You are the PanelRouter for a research deliberation council.\n"
        "Given a brief, propose which of the nine methodological lenses should "
        "analyze it. Produce ONE-SENTENCE reasoning for EACH of the nine "
        "lenses — both included and excluded.\n"
        "- Reasoning must be specific to THIS brief, not boilerplate.\n"
        "- Exclusion reasoning matters as much as inclusion reasoning: the user "
        "needs to see who you did not pick and why, so they can spot wrong calls.\n"
        f"This brief is in {mode.value} mode. Target panel size: ~{target} lenses "
        "(soft target; deviate if the brief warrants it).\n"
        "Call emit_panel exactly once with rationales for all nine lenses."
    )


def build_router_user_prompt(brief: Brief) -> str:
    """Render the brief for the router. Includes ``panel_constraints`` if set,
    so the LLM's reasoning is coherent with the user's pre-specified panel."""
    lines = [
        "## Brief",
        f"mode: {brief.mode.value}",
        f"problem_statement: {brief.problem_statement}",
        f"proposed_solution: {brief.proposed_solution or '(none)'}",
        f"researcher_context: {brief.researcher_context}",
        f"success_criteria_for_deliberation: {brief.success_criteria_for_deliberation}",
        f"scope_and_non_scope: {brief.scope_and_non_scope}",
    ]
    if brief.panel_constraints is not None:
        constrained = ", ".join(lens.value for lens in brief.panel_constraints)
        lines.append(
            f"panel_constraints (user pre-specified, must be included): {constrained}"
        )
    lines.append(
        "\nPropose the panel and emit_panel with rationales for all nine lenses."
    )
    return "\n".join(lines)


def _validate_full_coverage(rationales: list[LensRationale]) -> None:
    """All nine lenses must have exactly one rationale (story 81)."""
    counts = Counter(r.lens_id for r in rationales)
    missing = sorted(lens.value for lens in set(LensId) - counts.keys())
    if missing:
        raise ValueError(f"missing rationale for lens(es): {missing}")
    duplicates = sorted(lens.value for lens, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate rationale(s) for lens(es): {duplicates}")


def _materialize(
    draft: PanelProposalDraft, panel_constraints: tuple[LensId, ...] | None
) -> PanelProposal:
    rationales = tuple(draft.rationales)
    if panel_constraints is not None:
        # Story 86 acceptance: constraints override hybrid routing. The LLM
        # reasoning TEXT is preserved as narration, but the panel is exactly the
        # user-pre-specified set — and each rationale's `included` flag is
        # reconciled to match, so a lens the LLM picked but the constraints drop
        # reads as excluded on Screen 2 (with its reasoning intact).
        panel = tuple(panel_constraints)
        constrained = set(panel_constraints)
        rationales = tuple(
            r.model_copy(update={"included": r.lens_id in constrained}) for r in rationales
        )
    else:
        panel = tuple(r.lens_id for r in rationales if r.included)
    return PanelProposal(panel=panel, rationales=rationales)


async def propose_panel(
    brief: Brief,
    *,
    client: LlmClient,
    model: str = DEFAULT_MODEL,
    caller_ref: str = "panel_router",
) -> PanelProposal:
    """Run hybrid routing and return a panel proposal.

    The LLM produces one-sentence reasoning for every lens (story 81); the
    router validates full coverage of the nine lenses. If validation fails the
    router gives the LLM one retry with the error, matching the two-strike
    pattern used by ``run_lens``. If the brief carries ``panel_constraints``,
    those override the LLM's inclusion set.
    """
    system = build_router_system_prompt(brief.mode)
    tools = [
        ToolSpec(
            name=EMIT_PANEL_TOOL,
            description=(
                "Emit the panel proposal. Call exactly once with rationales for all nine lenses."
            ),
            input_schema=_emit_panel_schema(),
        )
    ]
    messages: list[Message] = [
        Message(role="user", content=[TextBlock(build_router_user_prompt(brief))])
    ]

    last_error = ""
    for _ in range(MAX_EMIT_ATTEMPTS):
        response = await client.create_message(
            LlmRequest(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                caller_ref=caller_ref,
                tool_choice="any",
            )
        )
        messages.append(Message(role="assistant", content=response.content))

        emit = next((b for b in response.tool_uses if b.name == EMIT_PANEL_TOOL), None)
        if emit is None:
            last_error = "no emit_panel tool call"
            messages.append(
                Message(
                    role="user",
                    content=[TextBlock("Call the emit_panel tool with all nine rationales.")],
                )
            )
            continue

        try:
            draft = PanelProposalDraft.model_validate(emit.input)
            _validate_full_coverage(draft.rationales)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            messages.append(
                Message(
                    role="user",
                    content=[
                        TextBlock(
                            "emit_panel was invalid. Fix and resubmit. Error:\n" + last_error
                        )
                    ],
                )
            )
            continue

        return _materialize(draft, brief.panel_constraints)

    raise ValueError(f"PanelRouter could not produce a valid proposal: {last_error}")


# --- DispatchEvent ----------------------------------------------------------


def make_dispatch_event(
    brief: Brief,
    panel: tuple[LensId, ...] | list[LensId],
    *,
    id_generator: IdGenerator,
    now: datetime,
) -> DispatchEvent:
    """Create a DispatchEvent for ``brief`` with ``panel`` (story 85).

    A panel edit at dispatch time produces a new DispatchEvent — never a new
    brief version. The brief content is unchanged; only the dispatch
    configuration (which lenses see this brief) changes. By contrast,
    ``panel_constraints`` edited DURING intake goes through ``revise_brief``
    and DOES bump the brief version (story 86).
    """
    return DispatchEvent(
        id=new_dispatch_event_id(id_generator),
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=tuple(panel),
        timestamp=now,
    )
