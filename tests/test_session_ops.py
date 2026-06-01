"""Session status machine (story 8): only legal transitions permitted."""

from __future__ import annotations

import pytest

from research_council.domain.session_ops import (
    IllegalTransitionError,
    apply_transition,
    is_legal_transition,
)
from research_council.enums import SessionStatus
from research_council.models import Session


def test_full_happy_path_transitions(session: Session) -> None:
    s = apply_transition(session, SessionStatus.CONSULTING)
    s = apply_transition(s, SessionStatus.SYNTHESIZED)
    s = apply_transition(s, SessionStatus.ITERATING)
    s = apply_transition(s, SessionStatus.CONSULTING)  # re-dispatch
    s = apply_transition(s, SessionStatus.SYNTHESIZED)  # re-synthesize
    s = apply_transition(s, SessionStatus.CLOSED)
    assert s.status is SessionStatus.CLOSED


def test_can_abandon_from_any_nonterminal_state(session: Session) -> None:
    assert is_legal_transition(SessionStatus.INTAKE, SessionStatus.CLOSED)
    assert is_legal_transition(SessionStatus.CONSULTING, SessionStatus.CLOSED)
    assert is_legal_transition(SessionStatus.SYNTHESIZED, SessionStatus.CLOSED)
    assert is_legal_transition(SessionStatus.ITERATING, SessionStatus.CLOSED)


def test_illegal_skip_is_rejected(session: Session) -> None:
    with pytest.raises(IllegalTransitionError):
        apply_transition(session, SessionStatus.SYNTHESIZED)  # intake -> synthesized


def test_closed_is_terminal() -> None:
    assert not is_legal_transition(SessionStatus.CLOSED, SessionStatus.INTAKE)
    assert not is_legal_transition(SessionStatus.CLOSED, SessionStatus.CONSULTING)


def test_apply_transition_does_not_mutate_input(session: Session) -> None:
    apply_transition(session, SessionStatus.CONSULTING)
    assert session.status is SessionStatus.INTAKE
