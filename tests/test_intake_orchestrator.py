"""IntakeOrchestrator — the master agent's intake conversation (Layer 3.1).

Covers mode declaration, the 9-field elicitation order, draft-vs-ask switch at
turn 3, the never-draft-success-criteria invariant, mode-specific termination,
the three decline gates, two-screen confirmation, and edit-after-confirm bumping
the brief version.
"""

from __future__ import annotations

import pytest

from research_council.domain.intake import (
    INTAKE_FIELD_ORDER,
    DeclineCategory,
    IntakeOrchestrator,
    PromptKind,
)
from research_council.enums import LensId, Mode, SessionStatus
from research_council.ids import SequentialIdGenerator, new_session_id
from research_council.models import Session
from research_council.store import InMemorySessionStore

# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def orchestrator(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> IntakeOrchestrator:
    session = Session(id=new_session_id(ids), status=SessionStatus.INTAKE)
    store.save_session(session)
    return IntakeOrchestrator(session_id=session.id, store=store)


def _drive_to_deep_dive_full_brief(orch: IntakeOrchestrator) -> None:
    """Walk through all 9 deep_dive fields with placeholder content."""
    orch.declare_mode(Mode.DEEP_DIVE)
    orch.submit_field("problem_statement", "Long-context degradation.")
    orch.submit_field("proposed_solution", "Filter agent.")
    orch.submit_field("researcher_context", "AI engineer.")
    orch.submit_field("background_claims", [])
    orch.submit_field("success_criteria_for_deliberation", "Surface unknown gaps.")
    orch.submit_field("scope_and_non_scope", "In: arch. Out: pretraining.")
    orch.submit_field("panel_constraints", None)
    orch.submit_field("prior_panel_consultations", [])


# --- Mode declaration (story 66) -------------------------------------------


def test_first_prompt_asks_for_mode(orchestrator: IntakeOrchestrator) -> None:
    prompt = orchestrator.next_prompt()
    assert prompt.kind is PromptKind.ASK_MODE


def test_cannot_submit_a_field_before_mode_is_declared(
    orchestrator: IntakeOrchestrator,
) -> None:
    with pytest.raises(ValueError, match="mode"):
        orchestrator.submit_field("problem_statement", "X")


def test_declare_mode_advances_to_first_field_prompt(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    prompt = orchestrator.next_prompt()
    assert prompt.field == "problem_statement"


def test_mode_cannot_be_re_declared(orchestrator: IntakeOrchestrator) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    with pytest.raises(ValueError, match="mode"):
        orchestrator.declare_mode(Mode.EXPLORATORY)


# --- Field elicitation order (story 69) ------------------------------------


def test_field_order_matches_prd(orchestrator: IntakeOrchestrator) -> None:
    assert INTAKE_FIELD_ORDER == (
        "problem_statement",
        "proposed_solution",
        "researcher_context",
        "background_claims",
        "success_criteria_for_deliberation",
        "scope_and_non_scope",
        "panel_constraints",
        "prior_panel_consultations",
    )


def test_fields_elicited_in_order(orchestrator: IntakeOrchestrator) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    seen: list[str] = []
    for field in INTAKE_FIELD_ORDER:
        prompt = orchestrator.next_prompt()
        seen.append(prompt.field or "")
        orchestrator.submit_field(field, "placeholder" if field == "problem_statement" else None)
    assert seen == list(INTAKE_FIELD_ORDER)


def test_submitting_out_of_order_field_is_rejected(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    with pytest.raises(ValueError, match="order"):
        orchestrator.submit_field("scope_and_non_scope", "anything")


# --- Draft-vs-ask logic (stories 72, 73, 74) -------------------------------


def test_first_three_turns_ask_open_questions_no_drafts(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    # Turn 1: problem_statement
    p1 = orchestrator.next_prompt()
    assert p1.kind is PromptKind.ASK_OPEN
    orchestrator.submit_field("problem_statement", "X")
    # Turn 2: proposed_solution
    p2 = orchestrator.next_prompt()
    assert p2.kind is PromptKind.ASK_OPEN
    orchestrator.submit_field("proposed_solution", "Y")
    # Turn 3: researcher_context
    p3 = orchestrator.next_prompt()
    assert p3.kind is PromptKind.ASK_OPEN
    orchestrator.submit_field("researcher_context", "Z")


def test_turn_four_and_after_proposes_drafts_for_draftable_fields(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    orchestrator.submit_field("problem_statement", "X")
    orchestrator.submit_field("proposed_solution", "Y")
    orchestrator.submit_field("researcher_context", "Z")
    # Turn 4: background_claims is draftable
    p4 = orchestrator.next_prompt()
    assert p4.kind is PromptKind.DRAFT
    assert p4.field == "background_claims"
    assert p4.draft is not None


def test_success_criteria_is_never_drafted_even_after_turn_three(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    orchestrator.submit_field("problem_statement", "X")
    orchestrator.submit_field("proposed_solution", "Y")
    orchestrator.submit_field("researcher_context", "Z")
    orchestrator.submit_field("background_claims", [])
    # Turn 5: success_criteria — must remain an open question forever
    p5 = orchestrator.next_prompt()
    assert p5.field == "success_criteria_for_deliberation"
    assert p5.kind is PromptKind.ASK_OPEN
    assert p5.draft is None


# --- Termination conditions (stories 70, 71, 68) ---------------------------


def test_deep_dive_termination_requires_all_nine_fields(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    orchestrator.submit_field("problem_statement", "X")
    assert orchestrator.can_terminate() is False


def test_deep_dive_termination_requires_two_turn_stability(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    # All 9 present (mode + 8 fields), but the latest edit was the current turn.
    # Two intake turns of no edits are required.
    assert orchestrator.can_terminate() is False
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is False
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is True


def test_deep_dive_self_test_is_a_recommendation_not_a_gate(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    # User can dispatch even before the self-test has been surfaced;
    # dispatch is always available once can_terminate() is True.
    brief = orchestrator.dispatch()
    assert brief.mode is Mode.DEEP_DIVE


def test_deep_dive_self_test_surfaces_three_questions(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    questions = orchestrator.self_test_questions()
    assert len(questions) == 3


def test_exploratory_minimum_four_fields_terminates(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.EXPLORATORY)
    orchestrator.submit_field("problem_statement", "p")
    orchestrator.submit_field("proposed_solution", None)
    orchestrator.submit_field("researcher_context", "rc")  # one of solution|context
    # In exploratory we can skip optional fields not in the 4-min set.
    orchestrator.submit_field("background_claims", [])
    orchestrator.submit_field("success_criteria_for_deliberation", "succ")
    orchestrator.submit_field("scope_and_non_scope", "scope")
    orchestrator.submit_field("panel_constraints", None)
    orchestrator.submit_field("prior_panel_consultations", [])
    assert orchestrator.can_terminate() is False  # need 2-turn stability
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is True


def test_exploratory_with_solution_but_no_context_terminates(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.EXPLORATORY)
    orchestrator.submit_field("problem_statement", "p")
    orchestrator.submit_field("proposed_solution", "sol")
    orchestrator.submit_field("researcher_context", None)  # alternative satisfied by solution
    orchestrator.submit_field("background_claims", [])
    orchestrator.submit_field("success_criteria_for_deliberation", "succ")
    orchestrator.submit_field("scope_and_non_scope", "scope")
    orchestrator.submit_field("panel_constraints", None)
    orchestrator.submit_field("prior_panel_consultations", [])
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is True


def test_exploratory_missing_both_solution_and_context_blocks_terminate(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.EXPLORATORY)
    orchestrator.submit_field("problem_statement", "p")
    orchestrator.submit_field("proposed_solution", None)
    orchestrator.submit_field("researcher_context", None)
    orchestrator.submit_field("background_claims", [])
    orchestrator.submit_field("success_criteria_for_deliberation", "succ")
    orchestrator.submit_field("scope_and_non_scope", "scope")
    orchestrator.submit_field("panel_constraints", None)
    orchestrator.submit_field("prior_panel_consultations", [])
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is False


def test_exploratory_does_not_offer_self_test(
    orchestrator: IntakeOrchestrator,
) -> None:
    orchestrator.declare_mode(Mode.EXPLORATORY)
    orchestrator.submit_field("problem_statement", "p")
    orchestrator.submit_field("proposed_solution", None)
    orchestrator.submit_field("researcher_context", "rc")
    orchestrator.submit_field("background_claims", [])
    orchestrator.submit_field("success_criteria_for_deliberation", "succ")
    orchestrator.submit_field("scope_and_non_scope", "scope")
    orchestrator.submit_field("panel_constraints", None)
    orchestrator.submit_field("prior_panel_consultations", [])
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    assert orchestrator.self_test_questions() == ()


# --- Decline gates (story 78) -----------------------------------------------


def test_decline_not_a_research_question(orchestrator: IntakeOrchestrator) -> None:
    orchestrator.declare_mode(Mode.DEEP_DIVE)
    orchestrator.decline(DeclineCategory.NOT_A_RESEARCH_QUESTION, "best pizza in SF")
    prompt = orchestrator.next_prompt()
    assert prompt.kind is PromptKind.DECLINE
    assert prompt.decline_category is DeclineCategory.NOT_A_RESEARCH_QUESTION
    with pytest.raises(ValueError, match="decline"):
        orchestrator.dispatch()


def test_decline_insufficient_specificity_after_intake(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.decline(
        DeclineCategory.INSUFFICIENT_SPECIFICITY,
        "Even after intake the brief lacks a testable claim.",
    )
    prompt = orchestrator.next_prompt()
    assert prompt.decline_category is DeclineCategory.INSUFFICIENT_SPECIFICITY


def test_decline_when_all_claims_failed_verification(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    # 4 unverified, 0 verified -> auto-decline per story 78 (c).
    orchestrator.record_verification_summary(unverified=4, verified=0)
    prompt = orchestrator.next_prompt()
    assert prompt.kind is PromptKind.DECLINE
    assert prompt.decline_category is DeclineCategory.ALL_CLAIMS_FAILED_VERIFICATION


def test_three_unverified_does_not_trigger_auto_decline(
    orchestrator: IntakeOrchestrator,
) -> None:
    """The PRD says ">3 unverified, 0 verified". Exactly 3 must NOT decline."""
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.record_verification_summary(unverified=3, verified=0)
    assert orchestrator.decline_state is None


def test_any_verified_claim_suppresses_auto_decline(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.record_verification_summary(unverified=10, verified=1)
    assert orchestrator.decline_state is None


# --- Two-screen confirmation (stories 75, 76) ------------------------------


def test_screen1_shows_all_fields_in_schema_order_no_verification(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    screen1 = orchestrator.build_screen1()
    assert tuple(screen1.fields.keys()) == INTAKE_FIELD_ORDER + ("mode",)
    assert getattr(screen1, "verification_results", None) is None


def test_screen2_includes_self_test_verification_panel(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    orchestrator.record_verification_summary(unverified=1, verified=1)
    orchestrator.propose_panel(
        included={LensId.PRIOR_ART: "prior art is load-bearing here"},
        excluded={LensId.MECHANISTIC_INTERPRETABILITY: "mechanism not at issue"},
    )
    screen2 = orchestrator.build_screen2()
    assert len(screen2.self_test_questions) == 3
    assert screen2.verification_summary is not None
    assert LensId.PRIOR_ART in screen2.panel_included
    assert LensId.MECHANISTIC_INTERPRETABILITY in screen2.panel_excluded


def test_dispatch_persists_confirmed_brief_v1(
    orchestrator: IntakeOrchestrator, store: InMemorySessionStore
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    brief = orchestrator.dispatch()
    assert brief.version == 1
    assert brief.confirmed is True
    persisted = store.get_brief(brief.session_id, 1)
    assert persisted is not None
    assert persisted.confirmed is True


def test_edit_after_confirm_produces_a_new_brief_version(
    orchestrator: IntakeOrchestrator, store: InMemorySessionStore
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    v1 = orchestrator.dispatch()
    v2 = orchestrator.edit_confirmed_brief(problem_statement="Sharper statement.")
    assert v2.version == 2
    assert v2.confirmed is True
    assert v2.problem_statement == "Sharper statement."
    # v1 untouched
    v1_stored = store.get_brief(v1.session_id, 1)
    assert v1_stored is not None
    assert v1_stored.problem_statement == "Long-context degradation."


def test_edit_after_confirm_returns_screen2_for_re_review(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    orchestrator.dispatch()
    orchestrator.edit_confirmed_brief(problem_statement="Sharper.")
    # After an edit-after-confirm, the next prompt the user sees is Screen 2
    # for the new brief version (story 76).
    prompt = orchestrator.next_prompt()
    assert prompt.kind is PromptKind.SCREEN2


def test_dispatch_cannot_be_called_twice(
    orchestrator: IntakeOrchestrator, store: InMemorySessionStore
) -> None:
    """Story 76: once confirmed, the only path to mutate the brief is
    edit_confirmed_brief (which bumps version). A second dispatch() would
    silently overwrite v1 with whatever the orchestrator currently holds —
    breaking the post-confirmation immutability invariant."""
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    orchestrator.dispatch()
    # Simulate the master mutating field state after confirmation.
    orchestrator.edit_field("problem_statement", "would-overwrite-v1")
    with pytest.raises(ValueError, match="already confirmed"):
        orchestrator.dispatch()
    # v1 in the store is untouched.
    v1 = store.get_brief(orchestrator.session_id, 1)
    assert v1 is not None
    assert v1.problem_statement == "Long-context degradation."


# --- Brief edits during intake (story 86) ----------------------------------


def test_editing_a_field_during_intake_records_a_turn(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    # All fields present but two-turn stability not yet satisfied.
    orchestrator.idle_turn()
    # An edit DURING intake resets the stability clock.
    orchestrator.edit_field("problem_statement", "edited")
    assert orchestrator.can_terminate() is False
    orchestrator.idle_turn()
    orchestrator.idle_turn()
    assert orchestrator.can_terminate() is True


def test_panel_constraints_edit_during_intake_is_a_brief_edit(
    orchestrator: IntakeOrchestrator,
) -> None:
    _drive_to_deep_dive_full_brief(orchestrator)
    orchestrator.idle_turn()
    orchestrator.edit_field("panel_constraints", (LensId.PRIOR_ART,))
    assert orchestrator.can_terminate() is False
