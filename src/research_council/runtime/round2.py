"""Round 2 controller (Layer 3.4, stories 95-101).

Cross-pollination over anonymized Round 1 outputs.

What the controller does:

* **Participant selection.** Adversarial and prior-art are mandatory (story 95)
  — included regardless of their Round 1 outcome. Tier-2 lenses opt in via the
  story-96 heuristic: their Round 1 ``disagreements_with_my_own_framing`` must
  contain an entry that explicitly references "would change if other lenses..."
  (case-insensitive). Generic self-doubt is not enough.
* **Anonymization.** Every succeeded Round 1 run is rewritten as a
  :class:`AnonymizedPeer` labelled ``Lens A``, ``Lens B``, ... in the order the
  caller supplies (the contract: Round 1 completion order). The adversarial
  lens's own Round 1 output appears as one of these labels — unmarked — so
  fresh engagement isn't pre-spoiled (story 97). Peers'
  ``disagreements_with_my_own_framing`` is dropped entirely (story 98); that
  field is for synthesis, not cross-pollination.
* **Skip.** If fewer than 3 Round 1 runs succeeded, Round 2 is skipped entirely
  and synthesis runs on Round 1 (story 101). The result still carries a
  synthesis-input summary explaining the skip.
* **Failure handling.** Identical to Round 1 — both timeouts, two-strike
  validation inside the task, schema_invalid surfaced. The difference is the
  >70% threshold (story 100): instead of halting (Round 1's behavior), the
  controller emits a ``failure_note`` so the synthesis layer can run on Round 1
  only with a note. Round 2 failure never halts synthesis.

Like Round 1, the controller is fed prepared per-lens tasks via a
``Round2TaskBuilder`` so the caller can close over ``run_lens`` with each
participant's ``LensRunInput`` (the cross_pollination peers are computed by the
controller and handed to the builder)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from ..enums import LensId, LensRunStatus
from ..ids import IdGenerator, SessionId, new_dispatch_event_id
from ..models import DispatchEvent, LensRun
from ..store.interface import SessionStore
from .inputs import AnonymizedPeer, PeerOutput, anonymize_peers
from .round1 import (
    DEFAULT_FAILURE_HALT_THRESHOLD,
    DEFAULT_HARD_TIMEOUT_SECONDS,
    DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS,
    LensTask,
    _names_with_status,
    _run_with_timeouts,
    _utc_now,
)
from .run_lens import persist_outcome

MANDATORY_PARTICIPANTS: Final[tuple[LensId, ...]] = (
    LensId.ADVERSARIAL,
    LensId.PRIOR_ART,
)

# Story 96: the explicit textual signal that a Round 1 lens wants to see peers
# before standing by its analysis. Generic self-doubt does NOT trip this — the
# whole point is "narrow opt-in, not full panel re-run."
OPT_IN_HEURISTIC_PHRASE: Final[str] = "would change if other lens"

# Story 101: the minimum succeeded Round 1 count for cross-pollination to be
# meaningful at all. Below this, Round 2 is skipped wholesale.
MIN_ROUND_1_SUCCESSES_FOR_ROUND_2: Final[int] = 3

Round2TaskBuilder = Callable[[LensId, list[AnonymizedPeer]], LensTask]


@dataclass(frozen=True)
class Round2Result:
    """The controller's output for one Round 2 dispatch.

    Three terminal shapes the synthesis layer must distinguish:

    * **Skipped** (story 101): ``skipped=True``, no dispatch event, empty
      ``runs`` / ``participants``. Synthesis runs on Round 1 with the skip
      noted via ``synthesis_input_summary``.
    * **Dispatched, mostly failed** (story 100): ``failure_note`` is set; the
      runs are still present and persisted. Synthesis runs on Round 1 only
      with the note, but is NOT halted.
    * **Dispatched, healthy**: ``failure_note`` is None; ``runs`` are folded
      into synthesis alongside Round 1."""

    dispatch_event: DispatchEvent | None
    runs: tuple[LensRun, ...]
    participants: tuple[LensId, ...]
    skipped: bool
    skip_reason: str | None
    failure_note: str | None
    synthesis_input_summary: str

    @property
    def succeeded(self) -> tuple[LensRun, ...]:
        return tuple(r for r in self.runs if r.status is LensRunStatus.SUCCEEDED)


def _is_opt_in_disagreement(text: str) -> bool:
    return OPT_IN_HEURISTIC_PHRASE in text.lower()


def _should_opt_in(run: LensRun) -> bool:
    """Tier-2 opt-in test: a succeeded, non-mandatory lens whose disagreements
    explicitly reference 'would change if other lenses...'. Failed runs are
    ineligible — they have no valid output to cross-pollinate from anyway."""
    if run.status is not LensRunStatus.SUCCEEDED:
        return False
    if run.lens_id in MANDATORY_PARTICIPANTS:
        return False
    return any(_is_opt_in_disagreement(d) for d in run.disagreements_with_my_own_framing)


def _select_participants(round1_runs: tuple[LensRun, ...]) -> tuple[LensId, ...]:
    """Mandatory + opt-in tier-2, preserving the natural mandatory ordering and
    appending opt-ins in the supplied run order."""
    selected: list[LensId] = list(MANDATORY_PARTICIPANTS)
    for run in round1_runs:
        if _should_opt_in(run) and run.lens_id not in selected:
            selected.append(run.lens_id)
    return tuple(selected)


def _build_anonymized_peers(
    round1_runs_in_completion_order: tuple[LensRun, ...],
    store: SessionStore,
) -> list[AnonymizedPeer]:
    """Build the cross_pollination input that every Round 2 participant sees.

    Only succeeded Round 1 runs become peers (a schema_invalid run has no
    typed findings to share). The labelling order is exactly the input
    order — the caller's contract is to pass runs in Round 1 completion
    order (story 97)."""
    peer_outputs: list[PeerOutput] = []
    for run in round1_runs_in_completion_order:
        if run.status is not LensRunStatus.SUCCEEDED:
            continue
        findings = store.list_findings_for_lens_run(run.id)
        peer_outputs.append(
            PeerOutput(
                lens_id=run.lens_id,
                findings=findings,
                brief_summary=run.brief_summary,
                open_questions=list(run.open_questions),
            )
        )
    return anonymize_peers(peer_outputs)


def _summarize_for_synthesis(
    runs: tuple[LensRun, ...], failure_note: str | None
) -> str:
    """Round 2 synthesis-input summary. Names schema_invalid / timed_out /
    refused lenses explicitly, then appends the failure note if set."""
    valid = sum(1 for r in runs if r.status is LensRunStatus.SUCCEEDED)
    parts = [
        f"Round 2: {len(runs)} lenses dispatched",
        f"{valid} produced valid output",
    ]

    schema_invalid = _names_with_status(runs, LensRunStatus.SCHEMA_INVALID)
    if schema_invalid:
        parts.append(
            f"{len(schema_invalid)} ({', '.join(schema_invalid)}) failed schema "
            "validation after retry — excluded from synthesis"
        )

    timed_out = _names_with_status(runs, LensRunStatus.TIMEOUT)
    if timed_out:
        parts.append(
            f"{len(timed_out)} ({', '.join(timed_out)}) timed out "
            "— excluded from synthesis"
        )

    refused = _names_with_status(runs, LensRunStatus.REFUSED)
    if refused:
        parts.append(
            f"{len(refused)} ({', '.join(refused)}) refused "
            "— excluded from synthesis"
        )

    if failure_note is not None:
        parts.append(failure_note)

    return "; ".join(parts) + "."


def _maybe_failure_note(
    runs: tuple[LensRun, ...], threshold: float
) -> str | None:
    """The Round 2 analogue of Round 1's halt check (story 100). Same >70%
    rule, different consequence: synthesis runs on Round 1 only with a note,
    rather than halting."""
    if not runs:
        return None
    non_succeeded = sum(1 for r in runs if r.status is not LensRunStatus.SUCCEEDED)
    if non_succeeded / len(runs) > threshold:
        return "Round 2 cross-pollination failed"
    return None


async def run_round_2(
    *,
    session_id: SessionId,
    brief_version: int,
    round1_runs_in_completion_order: tuple[LensRun, ...],
    store: SessionStore,
    task_builder: Round2TaskBuilder,
    id_generator: IdGenerator,
    now: Callable[[], datetime] = _utc_now,
    hard_timeout_seconds: float = DEFAULT_HARD_TIMEOUT_SECONDS,
    no_progress_timeout_seconds: float = DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS,
    failure_note_threshold: float = DEFAULT_FAILURE_HALT_THRESHOLD,
) -> Round2Result:
    """Dispatch Round 2 cross-pollination, or skip if Round 1 lacks consensus."""
    succeeded_r1 = tuple(
        r
        for r in round1_runs_in_completion_order
        if r.status is LensRunStatus.SUCCEEDED
    )

    if len(succeeded_r1) < MIN_ROUND_1_SUCCESSES_FOR_ROUND_2:
        skip_reason = (
            f"Round 2 skipped: only {len(succeeded_r1)} successful Round 1 runs "
            f"(need ≥{MIN_ROUND_1_SUCCESSES_FOR_ROUND_2} for meaningful "
            "cross-pollination)."
        )
        return Round2Result(
            dispatch_event=None,
            runs=(),
            participants=(),
            skipped=True,
            skip_reason=skip_reason,
            failure_note=None,
            synthesis_input_summary=skip_reason,
        )

    peers = _build_anonymized_peers(round1_runs_in_completion_order, store)
    participants = _select_participants(round1_runs_in_completion_order)

    dispatch_event_id = new_dispatch_event_id(id_generator)

    async def _run_one(lens_id: LensId) -> LensRun:
        task = task_builder(lens_id, peers)
        outcome = await _run_with_timeouts(
            task, hard_timeout_seconds, no_progress_timeout_seconds
        )
        return persist_outcome(
            store,
            outcome=outcome,
            lens_id=lens_id,
            session_id=session_id,
            brief_version=brief_version,
            round=2,
            id_generator=id_generator,
            dispatch_event_id=dispatch_event_id,
        )

    runs = tuple(await asyncio.gather(*(_run_one(lid) for lid in participants)))

    dispatch_event = DispatchEvent(
        id=dispatch_event_id,
        session_id=session_id,
        brief_version=brief_version,
        panel=participants,
        timestamp=now(),
        lens_run_ids=tuple(run.id for run in runs),
    )
    store.save_dispatch_event(dispatch_event)

    failure_note = _maybe_failure_note(runs, failure_note_threshold)
    summary = _summarize_for_synthesis(runs, failure_note)

    return Round2Result(
        dispatch_event=dispatch_event,
        runs=runs,
        participants=participants,
        skipped=False,
        skip_reason=None,
        failure_note=failure_note,
        synthesis_input_summary=summary,
    )
