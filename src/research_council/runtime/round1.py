"""Round 1 controller (Layer 3.3, stories 87-94).

Parallel dispatch of selected lenses with two timeouts, schema_invalid surfacing
for synthesis, and the >70% halt — all enforced in code (the controller is
intentionally testable via simulated outcomes).

What the controller is and is not:

* It dispatches lens *tasks* (caller-prepared callables) in parallel, persists
  every terminal outcome, and decides whether to halt before Round 2.
* It does NOT throttle: the verifier service handles rate limits internally
  (story 88). It also does NOT auto-retry timed-out runs (story 90) — a timeout
  is treated as an investigate-it signal, not a transient blip.
* It does NOT call into a lens. The caller wires up the LensTask (typically
  closing over ``run_lens`` with the right inputs); the controller wraps each
  task in the timeout watchdog and persists the outcome.

A ``LensTask`` is ``(progress) -> Awaitable[LensRunOutcome]``. ``progress`` is a
heartbeat callback the task is expected to invoke whenever it makes forward
progress (e.g., on each LLM tool-use turn). Tasks that never heartbeat will trip
the no-progress watchdog; tasks that hang past the hard ceiling trip the hard
watchdog. Both timeout kinds persist ``status: timeout``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..enums import LensId, LensRunStatus
from ..ids import IdGenerator, SessionId, new_dispatch_event_id
from ..models import DispatchEvent, LensRun
from ..store.interface import SessionStore
from .run_lens import LensRunOutcome, LensRunTimeout, persist_outcome

ProgressCallback = Callable[[], None]
# Coroutine (not Awaitable) so callers can pass the result directly to
# ``asyncio.create_task`` without an extra wrapper.
LensTask = Callable[[ProgressCallback], Coroutine[Any, Any, LensRunOutcome]]

# Production defaults from the PRD (story 89). Tests override with small values.
DEFAULT_HARD_TIMEOUT_SECONDS = 600.0
DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS = 180.0
# Strictly-greater halt threshold (story 94).
DEFAULT_FAILURE_HALT_THRESHOLD = 0.70


@dataclass(frozen=True)
class Round1Result:
    """The controller's output for one Round 1 dispatch.

    ``runs`` is in panel order. ``halted`` is True when the non-succeeded ratio
    strictly exceeds ``failure_halt_threshold``; the synthesis layer must respect
    this before advancing to Round 2.
    """

    dispatch_event: DispatchEvent
    runs: tuple[LensRun, ...]
    halted: bool
    halt_message: str | None
    synthesis_input_summary: str

    @property
    def succeeded(self) -> tuple[LensRun, ...]:
        return tuple(r for r in self.runs if r.status is LensRunStatus.SUCCEEDED)

    @property
    def non_succeeded(self) -> tuple[LensRun, ...]:
        return tuple(r for r in self.runs if r.status is not LensRunStatus.SUCCEEDED)


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _run_with_timeouts(
    task: LensTask,
    hard_timeout_seconds: float,
    no_progress_timeout_seconds: float,
) -> LensRunOutcome:
    """Run a single lens task under both timeouts. On either timeout the task is
    cancelled and a ``LensRunTimeout`` is returned. The task is invoked exactly
    once — no auto-retry (story 90)."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    last_progress = start

    def heartbeat() -> None:
        nonlocal last_progress
        last_progress = loop.time()

    main_task: asyncio.Task[LensRunOutcome] = asyncio.create_task(task(heartbeat))

    # Watchdog tick: fine enough to honor sub-second test timeouts, never busier
    # than 1 kHz. In production (10-min / 3-min) this is ~3 s per check. The wait
    # below returns as soon as the task finishes, so the tick is only a ceiling
    # on the check interval, not a floor on overall latency.
    tick = max(0.001, min(hard_timeout_seconds, no_progress_timeout_seconds) / 60)

    timed_out = False
    while True:
        done, _ = await asyncio.wait({main_task}, timeout=tick)
        if done:
            break
        now = loop.time()
        if (
            now - start >= hard_timeout_seconds
            or now - last_progress >= no_progress_timeout_seconds
        ):
            timed_out = True
            main_task.cancel()
            break

    if timed_out:
        # Drain the cancellation; swallow whatever the task raised (cancelled or
        # otherwise) — it's a timeout regardless.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await main_task
        return LensRunTimeout()

    return main_task.result()


def _names_with_status(
    runs: tuple[LensRun, ...], status: LensRunStatus
) -> list[str]:
    return sorted(r.lens_id.value for r in runs if r.status is status)


def _summarize_for_synthesis(runs: tuple[LensRun, ...]) -> str:
    """Synthesis-input summary (story 91). Names schema_invalid lenses explicitly
    and notes that raw output is preserved; also surfaces timeouts and refusals
    so the synthesizer can see why the count is short."""
    valid = sum(1 for r in runs if r.status is LensRunStatus.SUCCEEDED)
    parts = [f"{len(runs)} lenses dispatched", f"{valid} produced valid output"]

    schema_invalid = _names_with_status(runs, LensRunStatus.SCHEMA_INVALID)
    if schema_invalid:
        parts.append(
            f"{len(schema_invalid)} ({', '.join(schema_invalid)}) failed schema "
            "validation after retry — excluded from synthesis. "
            "Raw output preserved for debugging."
        )

    timed_out = _names_with_status(runs, LensRunStatus.TIMEOUT)
    if timed_out:
        parts.append(
            f"{len(timed_out)} ({', '.join(timed_out)}) timed out "
            "— excluded from synthesis."
        )

    refused = _names_with_status(runs, LensRunStatus.REFUSED)
    if refused:
        parts.append(
            f"{len(refused)} ({', '.join(refused)}) refused "
            "— excluded from synthesis."
        )

    return "; ".join(parts) + "."


def _maybe_halt(
    runs: tuple[LensRun, ...], threshold: float
) -> tuple[bool, str | None]:
    if not runs:
        return False, None
    non_succeeded = sum(1 for r in runs if r.status is not LensRunStatus.SUCCEEDED)
    if non_succeeded / len(runs) > threshold:
        return True, (
            f"{non_succeeded} of {len(runs)} lenses failed; this is unusual. "
            "Investigate before proceeding?"
        )
    return False, None


async def run_round_1(
    *,
    session_id: SessionId,
    brief_version: int,
    panel: tuple[LensId, ...],
    tasks: dict[LensId, LensTask],
    store: SessionStore,
    id_generator: IdGenerator,
    now: Callable[[], datetime] = _utc_now,
    hard_timeout_seconds: float = DEFAULT_HARD_TIMEOUT_SECONDS,
    no_progress_timeout_seconds: float = DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS,
    failure_halt_threshold: float = DEFAULT_FAILURE_HALT_THRESHOLD,
) -> Round1Result:
    """Dispatch the panel in parallel, persist every terminal outcome, decide on
    the halt, and assemble the synthesis-input summary."""
    if set(tasks.keys()) != set(panel):
        missing = sorted(lid.value for lid in set(panel) - set(tasks.keys()))
        extra = sorted(lid.value for lid in set(tasks.keys()) - set(panel))
        raise ValueError(
            f"tasks must cover the panel exactly; missing={missing} extra={extra}"
        )

    dispatch_event_id = new_dispatch_event_id(id_generator)

    async def _run_one(lens_id: LensId) -> LensRun:
        outcome = await _run_with_timeouts(
            tasks[lens_id], hard_timeout_seconds, no_progress_timeout_seconds
        )
        return persist_outcome(
            store,
            outcome=outcome,
            lens_id=lens_id,
            session_id=session_id,
            brief_version=brief_version,
            round=1,
            id_generator=id_generator,
            dispatch_event_id=dispatch_event_id,
        )

    runs = tuple(await asyncio.gather(*(_run_one(lens_id) for lens_id in panel)))

    dispatch_event = DispatchEvent(
        id=dispatch_event_id,
        session_id=session_id,
        brief_version=brief_version,
        panel=panel,
        timestamp=now(),
        lens_run_ids=tuple(run.id for run in runs),
    )
    store.save_dispatch_event(dispatch_event)

    halted, halt_message = _maybe_halt(runs, failure_halt_threshold)
    summary = _summarize_for_synthesis(runs)

    return Round1Result(
        dispatch_event=dispatch_event,
        runs=runs,
        halted=halted,
        halt_message=halt_message,
        synthesis_input_summary=summary,
    )
