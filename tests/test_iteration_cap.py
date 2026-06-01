"""Runaway-iteration cap (story 161): a per-Session hard cap of 10
challenge+refinement cycles, surfaced as a *diagnostic signal* rather than
silently enforced. The cap doubles as a hint that the brief itself is the
problem — so it never blocks; it annotates."""

from __future__ import annotations

from research_council.domain.iteration_cap import (
    MAX_SESSION_ITERATION_CYCLES,
    annotate_session_with_iteration_signal,
    assess_iteration_cap,
)
from research_council.domain.session_ops import apply_transition
from research_council.enums import SessionStatus
from research_council.ids import SequentialIdGenerator, new_challenge_id, new_session_id
from research_council.models import Challenge, Finding, Session


def _challenge(ids: SequentialIdGenerator, session: Session, finding: Finding) -> Challenge:
    return Challenge(
        id=new_challenge_id(ids),
        session_id=session.id,
        challenged_finding_ref=finding.id,
        challenge_text="Re-examine this claim.",
    )


def test_below_cap_produces_no_signal(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    s = session.model_copy(update={"current_brief_version": 3})  # 2 refinements
    signal = assess_iteration_cap(
        session=s, challenges=[_challenge(ids, s, finding) for _ in range(4)]
    )
    assert signal.total_cycles == 6  # 4 challenges + 2 refinements
    assert not signal.cap_reached
    assert signal.message is None


def test_cap_reached_at_ten_surfaces_diagnostic_message(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    s = session.model_copy(update={"current_brief_version": 4})  # 3 refinements
    signal = assess_iteration_cap(
        session=s, challenges=[_challenge(ids, s, finding) for _ in range(7)]
    )
    assert signal.total_cycles == MAX_SESSION_ITERATION_CYCLES
    assert signal.cap_reached
    assert signal.message is not None
    assert "the brief itself" in signal.message


def test_refinement_count_derives_from_brief_version(
    session: Session,
) -> None:
    # v1 confirmed brief is not a refinement; every version beyond it is.
    s = session.model_copy(update={"current_brief_version": 1})
    assert assess_iteration_cap(session=s, challenges=[]).refinement_cycles == 0
    s = session.model_copy(update={"current_brief_version": None})
    assert assess_iteration_cap(session=s, challenges=[]).refinement_cycles == 0


def test_challenges_for_other_sessions_are_not_counted(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    other = Session(id=new_session_id(ids))
    mine = _challenge(ids, session, finding)
    theirs = Challenge(
        id=new_challenge_id(ids),
        session_id=other.id,
        challenged_finding_ref=finding.id,
        challenge_text="not mine",
    )
    signal = assess_iteration_cap(session=session, challenges=[mine, theirs])
    assert signal.challenge_cycles == 1


def test_signal_annotates_session_notes_without_blocking(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    s = session.model_copy(update={"current_brief_version": 11})  # 10 refinements
    signal = assess_iteration_cap(session=s, challenges=[])
    assert signal.cap_reached

    annotated = annotate_session_with_iteration_signal(s, signal)
    assert annotated.notes[-1] == signal.message
    # Surfaced, never enforced: the session machine is untouched, transitions
    # remain legal after the cap is hit.
    moved = apply_transition(annotated, SessionStatus.CONSULTING)
    assert moved.status is SessionStatus.CONSULTING


def test_annotation_is_idempotent_and_noop_below_cap(
    session: Session,
) -> None:
    below = session.model_copy(update={"current_brief_version": 2})
    signal_below = assess_iteration_cap(session=below, challenges=[])
    assert annotate_session_with_iteration_signal(below, signal_below) is below

    at_cap = session.model_copy(update={"current_brief_version": 11})
    signal = assess_iteration_cap(session=at_cap, challenges=[])
    once = annotate_session_with_iteration_signal(at_cap, signal)
    twice = annotate_session_with_iteration_signal(once, signal)
    assert once.notes == twice.notes  # no duplicate note


def test_assess_does_not_mutate_session(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    s = session.model_copy(update={"current_brief_version": 11})
    assess_iteration_cap(session=s, challenges=[_challenge(ids, s, finding)])
    assert s.notes == ()
