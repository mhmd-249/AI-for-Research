"""Session status machine (story 8). Only legal transitions are permitted.

Kept as pure functions over a :class:`Session` (no store dependency): the caller
reads, applies, and persists. ``closed`` is terminal; a session may be closed
from any non-terminal state (abandon at any point).
"""

from __future__ import annotations

from ..enums import SessionStatus
from ..models import Session

LEGAL_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.INTAKE: frozenset({SessionStatus.CONSULTING, SessionStatus.CLOSED}),
    SessionStatus.CONSULTING: frozenset({SessionStatus.SYNTHESIZED, SessionStatus.CLOSED}),
    SessionStatus.SYNTHESIZED: frozenset({SessionStatus.ITERATING, SessionStatus.CLOSED}),
    SessionStatus.ITERATING: frozenset(
        {SessionStatus.CONSULTING, SessionStatus.SYNTHESIZED, SessionStatus.CLOSED}
    ),
    SessionStatus.CLOSED: frozenset(),
}


class IllegalTransitionError(ValueError):
    """Raised when a session status transition is not permitted."""

    def __init__(self, current: SessionStatus, target: SessionStatus) -> None:
        super().__init__(f"illegal session transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


def is_legal_transition(current: SessionStatus, target: SessionStatus) -> bool:
    return target in LEGAL_TRANSITIONS[current]


def assert_legal_transition(current: SessionStatus, target: SessionStatus) -> None:
    if not is_legal_transition(current, target):
        raise IllegalTransitionError(current, target)


def apply_transition(session: Session, target: SessionStatus) -> Session:
    """Return ``session`` advanced to ``target``, or raise on an illegal edge."""
    assert_legal_transition(session.status, target)
    return session.model_copy(update={"status": target})
