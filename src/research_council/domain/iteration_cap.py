"""Runaway-iteration cap (story 161) — the fourth cross-cutting safeguard.

A per-Session hard cap of **10 challenge+refinement cycles**, surfaced as a
*diagnostic signal* rather than silently enforced: hitting it is a hint that the
problem is upstream of the tool ("the council may be telling you the brief
itself is the problem"), not a nuisance limit. So this module never raises and
never blocks a transition — it assesses, and a caller may surface the signal
into :attr:`Session.notes`.

Kept as pure functions over typed models (no store), matching
:mod:`.session_ops`: the caller reads the persisted graph, assesses, and
persists the annotated Session.

A "cycle" is one challenge OR one refinement:

* **Challenge cycles** — the count of :class:`Challenge` records for the Session.
* **Refinement cycles** — brief versions beyond the initial confirmed v1. Every
  post-confirmation brief edit (intake re-edit or the refinement loop) mints a
  new version (story 76/118), so ``current_brief_version - 1`` is the number of
  refinements the Session has accumulated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..models import Challenge, Session

# The cap (story 161). Arbitrary by design — refine on data; the point is the
# signal, not the precise number (PRD open question on the right value).
MAX_SESSION_ITERATION_CYCLES = 10

# story 161's diagnostic wording: the cap is a signal that the brief is upstream
# of the churn, not a limit being hit.
_SIGNAL_TEMPLATE = (
    "You've run {cycles} challenge+refinement cycles on this Session — the "
    "council may be telling you the brief itself is the problem."
)


@dataclass(frozen=True)
class IterationCapSignal:
    """The assessment of a Session's iteration churn. ``message`` is the
    diagnostic to surface, present only once the cap is reached — below the cap
    there is nothing to say."""

    challenge_cycles: int
    refinement_cycles: int
    cap: int

    @property
    def total_cycles(self) -> int:
        return self.challenge_cycles + self.refinement_cycles

    @property
    def cap_reached(self) -> bool:
        return self.total_cycles >= self.cap

    @property
    def message(self) -> str | None:
        if not self.cap_reached:
            return None
        return _SIGNAL_TEMPLATE.format(cycles=self.total_cycles)


def assess_iteration_cap(
    *,
    session: Session,
    challenges: Iterable[Challenge],
    cap: int = MAX_SESSION_ITERATION_CYCLES,
) -> IterationCapSignal:
    """Count this Session's challenge+refinement cycles and assess against the
    cap. Pure — never mutates ``session`` and never raises."""
    refinement_cycles = max(0, (session.current_brief_version or 1) - 1)
    challenge_cycles = sum(1 for c in challenges if c.session_id == session.id)
    return IterationCapSignal(
        challenge_cycles=challenge_cycles,
        refinement_cycles=refinement_cycles,
        cap=cap,
    )


def annotate_session_with_iteration_signal(
    session: Session, signal: IterationCapSignal
) -> Session:
    """Surface the diagnostic into ``session.notes`` (story 161) — a signal,
    never a block. Returns the Session unchanged below the cap, and is idempotent
    at the cap (the note is appended at most once). The session status machine is
    deliberately untouched: no transition is forbidden by hitting the cap."""
    message = signal.message
    if message is None or message in session.notes:
        return session
    return session.model_copy(update={"notes": (*session.notes, message)})
