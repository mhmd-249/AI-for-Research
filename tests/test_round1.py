"""Round 1 controller (Layer 3.3, stories 87-94).

Tests the parallel dispatch, two timeouts, schema_invalid surfacing in synthesis
input, "all terminal" completion, and the >70% non-succeeded halt — all driven
by simulated lens outcomes per the issue acceptance criteria.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from research_council.enums import (
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    VerificationStatus,
)
from research_council.ids import SequentialIdGenerator, new_finding_id
from research_council.lenses import get_lens_config
from research_council.models import Brief, Finding
from research_council.runtime import (
    LensRunInput,
    LensRunOutcome,
    LensRunRefused,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    LensRunTimeout,
    LensTask,
    ProgressCallback,
    Round1Result,
    run_lens,
    run_round_1,
)
from research_council.runtime.llm import FakeLlmClient, LlmResponse, ToolUseBlock
from research_council.store import InMemorySessionStore

# --- helpers ----------------------------------------------------------------


def _finding(text: str, ids: SequentialIdGenerator) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=ClaimType.MECHANISM_HYPOTHESIS,
        confidence=Confidence.SUPPORTING,
        failure_modes_if_wrong="...",
        verification_status=VerificationStatus.UNVERIFIED,
        round=1,
    )


def _succeeded(ids: SequentialIdGenerator, text: str = "x") -> LensRunSucceeded:
    return LensRunSucceeded(
        findings=[_finding(text, ids)],
        open_questions=["q?"],
        disagreements_with_my_own_framing=["self-doubt"],
        brief_summary="summary",
    )


def task_returning(outcome: LensRunOutcome) -> LensTask:
    """A lens task that heartbeats once and returns the given outcome."""

    async def _task(progress: ProgressCallback) -> LensRunOutcome:
        progress()
        return outcome

    return _task


def task_silent_for(duration: float, outcome: LensRunOutcome) -> LensTask:
    """A lens task that sleeps without ever calling progress."""

    async def _task(progress: ProgressCallback) -> LensRunOutcome:  # noqa: ARG001
        await asyncio.sleep(duration)
        return outcome

    return _task


VALID_FIRST_PRINCIPLES: dict[str, Any] = {
    "findings": [
        {
            "claim_text": "Selective gating, not a separate filter agent.",
            "claim_type": "mechanism_hypothesis",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": "If gating cannot be learned, this fails.",
        }
    ],
    "open_questions": ["Can the gate be learned unsupervised?"],
    "disagreements_with_my_own_framing": ["I may discount engineering constraints."],
    "brief_summary": "Filter-agent proposal critique.",
}


def emit(payload: dict[str, Any]) -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id="u1", name="emit_output", input=payload)])


# --- parallel dispatch (story 87) -------------------------------------------


async def test_dispatches_all_lenses_in_parallel(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Four tasks that meet at a barrier. If the controller ran them serially the
    # barrier would deadlock and the test would time out.
    panel = (
        LensId.ADVERSARIAL,
        LensId.FIRST_PRINCIPLES,
        LensId.ARCHITECTURE,
        LensId.PRIOR_ART,
    )
    barrier = asyncio.Barrier(len(panel))

    def at_barrier(outcome: LensRunOutcome) -> LensTask:
        async def _task(progress: ProgressCallback) -> LensRunOutcome:
            progress()
            await barrier.wait()
            return outcome

        return _task

    tasks = {lens: at_barrier(_succeeded(ids)) for lens in panel}

    result = await asyncio.wait_for(
        run_round_1(
            session_id=brief.session_id,
            brief_version=brief.version,
            panel=panel,
            tasks=tasks,
            store=store,
            id_generator=ids,
            hard_timeout_seconds=2.0,
            no_progress_timeout_seconds=2.0,
        ),
        timeout=2.0,
    )
    assert len(result.runs) == len(panel)
    assert all(run.status is LensRunStatus.SUCCEEDED for run in result.runs)


# --- success and persistence ------------------------------------------------


async def test_all_succeed_persists_runs_findings_and_dispatch_event(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    panel = (LensId.ADVERSARIAL, LensId.FIRST_PRINCIPLES, LensId.ARCHITECTURE)
    tasks = {lens: task_returning(_succeeded(ids)) for lens in panel}

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
    )

    assert not result.halted
    assert result.halt_message is None
    assert {run.lens_id for run in result.runs} == set(panel)
    assert all(run.status is LensRunStatus.SUCCEEDED for run in result.runs)

    # DispatchEvent persisted with all lens_run_ids.
    persisted = store.get_dispatch_event(result.dispatch_event.id)
    assert persisted is not None
    assert set(persisted.lens_run_ids) == {run.id for run in result.runs}
    assert persisted.panel == panel
    assert persisted.brief_version == brief.version

    # LensRuns are saved against the same dispatch event.
    for run in result.runs:
        assert run.dispatch_event_id == result.dispatch_event.id
        # Findings persisted via the run.
        assert len(store.list_findings_for_lens_run(run.id)) == 1


# --- two-strike retry inside the task (integration with run_lens) -----------


async def test_invalid_once_then_valid_succeeds_for_single_lens(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Use the real run_lens to verify two-strike behaves end-to-end inside a task.
    bad = {**VALID_FIRST_PRINCIPLES, "disagreements_with_my_own_framing": []}
    client = FakeLlmClient([emit(bad), emit(VALID_FIRST_PRINCIPLES)])

    inputs = LensRunInput(
        brief=brief, lens_config=get_lens_config(LensId.FIRST_PRINCIPLES), round=1
    )

    async def _task(progress: ProgressCallback) -> LensRunOutcome:
        progress()
        return await run_lens(inputs=inputs, client=client, id_generator=ids)

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=(LensId.FIRST_PRINCIPLES,),
        tasks={LensId.FIRST_PRINCIPLES: _task},
        store=store,
        id_generator=ids,
    )
    assert result.runs[0].status is LensRunStatus.SUCCEEDED
    assert len(client.requests) == 2  # one retry consumed
    assert not result.halted


# --- schema_invalid surfaced explicitly in synthesis input (story 91) -------


async def test_schema_invalid_surfaced_in_synthesis_input(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    panel = (
        LensId.ADVERSARIAL,
        LensId.FIRST_PRINCIPLES,
        LensId.ARCHITECTURE,
    )
    tasks = {
        LensId.ADVERSARIAL: task_returning(_succeeded(ids)),
        LensId.FIRST_PRINCIPLES: task_returning(_succeeded(ids)),
        LensId.ARCHITECTURE: task_returning(
            LensRunSchemaInvalid(raw_output="{'bad': true}", validation_error="oops")
        ),
    }

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
    )

    # 3 dispatched; 2 valid; 1 (architecture) explicitly named with schema_invalid.
    summary = result.synthesis_input_summary
    assert "3" in summary  # dispatched
    assert "2" in summary  # valid
    assert "architecture" in summary
    assert "schema validation" in summary
    assert "raw output preserved" in summary.lower()

    # Raw output preserved on the persisted run, not lost.
    arch_run = next(r for r in result.runs if r.lens_id is LensId.ARCHITECTURE)
    assert arch_run.status is LensRunStatus.SCHEMA_INVALID
    assert arch_run.raw_output == "{'bad': true}"


# --- timeouts (story 89, 90) -------------------------------------------------


async def test_hard_timeout_kills_run_and_persists_timeout(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Hard timeout fires before no-progress (and before the task would ever return).
    panel = (LensId.FIRST_PRINCIPLES,)
    call_counter = {"n": 0}

    async def _task(progress: ProgressCallback) -> LensRunOutcome:
        call_counter["n"] += 1
        progress()  # one heartbeat so no-progress doesn't fire first
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks={LensId.FIRST_PRINCIPLES: _task},
        store=store,
        id_generator=ids,
        hard_timeout_seconds=0.05,
        no_progress_timeout_seconds=10.0,
    )
    assert call_counter["n"] == 1  # NO auto-retry (story 90).
    assert result.runs[0].status is LensRunStatus.TIMEOUT


async def test_no_progress_timeout_kills_silent_run(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # No-progress fires while the task is still alive and hard timeout is generous.
    panel = (LensId.FIRST_PRINCIPLES,)
    tasks = {LensId.FIRST_PRINCIPLES: task_silent_for(2.0, _succeeded(ids))}

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
        hard_timeout_seconds=10.0,
        no_progress_timeout_seconds=0.05,
    )
    assert result.runs[0].status is LensRunStatus.TIMEOUT


async def test_heartbeating_lens_avoids_no_progress_timeout(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # The lens stays alive longer than the no-progress window but heartbeats
    # often enough that the watchdog never trips.
    succeeded = _succeeded(ids)

    async def _task(progress: ProgressCallback) -> LensRunOutcome:
        for _ in range(5):
            progress()
            await asyncio.sleep(0.02)
        return succeeded

    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=(LensId.FIRST_PRINCIPLES,),
        tasks={LensId.FIRST_PRINCIPLES: _task},
        store=store,
        id_generator=ids,
        hard_timeout_seconds=10.0,
        no_progress_timeout_seconds=0.05,
    )
    assert result.runs[0].status is LensRunStatus.SUCCEEDED


# --- refusal as a legitimate terminal state ---------------------------------


async def test_refusal_persisted_with_reason(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    refused = LensRunRefused(reason="Outside my frame.")
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=(LensId.FIRST_PRINCIPLES,),
        tasks={LensId.FIRST_PRINCIPLES: task_returning(refused)},
        store=store,
        id_generator=ids,
    )
    run = result.runs[0]
    assert run.status is LensRunStatus.REFUSED
    assert run.refusal_reason == "Outside my frame."


# --- "no partial proceed" (story 93) ----------------------------------------


async def test_completion_waits_for_all_terminal_states(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # One fast lens, one slow lens. The controller must not return before the
    # slow lens reaches a terminal state — there is no partial proceed.
    panel = (LensId.ADVERSARIAL, LensId.FIRST_PRINCIPLES)

    async def slow_task(progress: ProgressCallback) -> LensRunOutcome:
        progress()
        await asyncio.sleep(0.1)
        progress()
        return _succeeded(ids)

    tasks = {
        LensId.ADVERSARIAL: task_returning(_succeeded(ids)),
        LensId.FIRST_PRINCIPLES: slow_task,
    }
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
        hard_timeout_seconds=2.0,
        no_progress_timeout_seconds=2.0,
    )
    assert len(result.runs) == 2
    assert {run.lens_id for run in result.runs} == set(panel)


# --- >70% non-succeeded triggers the halt (story 94) -------------------------


async def test_over_70_percent_failure_triggers_halt(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # 3 of 4 fail = 75%, > 70% -> halt.
    panel = (
        LensId.ADVERSARIAL,
        LensId.FIRST_PRINCIPLES,
        LensId.ARCHITECTURE,
        LensId.PRIOR_ART,
    )
    tasks = {
        LensId.ADVERSARIAL: task_returning(_succeeded(ids)),
        LensId.FIRST_PRINCIPLES: task_returning(
            LensRunSchemaInvalid(raw_output="x", validation_error="bad")
        ),
        LensId.ARCHITECTURE: task_returning(LensRunRefused(reason="not in frame")),
        LensId.PRIOR_ART: task_returning(LensRunTimeout()),
    }
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
    )
    assert result.halted
    assert result.halt_message is not None
    assert "3 of 4" in result.halt_message
    assert "investigate" in result.halt_message.lower()


async def test_at_threshold_does_not_halt(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # The halt is strictly > threshold (story 94). With 3 of 5 non-succeeded
    # (60%) and the default 0.70 threshold, no halt.
    panel = (
        LensId.ADVERSARIAL,
        LensId.FIRST_PRINCIPLES,
        LensId.ARCHITECTURE,
        LensId.PRIOR_ART,
        LensId.EMPIRICAL_BENCHMARKING,
    )
    tasks = {
        LensId.ADVERSARIAL: task_returning(_succeeded(ids)),
        LensId.FIRST_PRINCIPLES: task_returning(_succeeded(ids)),
        LensId.ARCHITECTURE: task_returning(
            LensRunSchemaInvalid(raw_output="x", validation_error="bad")
        ),
        LensId.PRIOR_ART: task_returning(LensRunRefused(reason="not in frame")),
        LensId.EMPIRICAL_BENCHMARKING: task_returning(LensRunTimeout()),
    }
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
    )
    assert not result.halted
    assert result.halt_message is None


async def test_threshold_strict_greater_than(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # With a custom threshold equal to the actual failure ratio, no halt
    # (equality is not "more than").
    panel = (LensId.ADVERSARIAL, LensId.FIRST_PRINCIPLES)
    tasks = {
        LensId.ADVERSARIAL: task_returning(_succeeded(ids)),
        LensId.FIRST_PRINCIPLES: task_returning(LensRunRefused(reason="x")),
    }
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=panel,
        tasks=tasks,
        store=store,
        id_generator=ids,
        failure_halt_threshold=0.5,  # exactly 1/2 = 50%
    )
    assert not result.halted


# --- input validation -------------------------------------------------------


async def test_panel_must_match_task_keys(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    with pytest.raises(ValueError, match="task"):
        await run_round_1(
            session_id=brief.session_id,
            brief_version=brief.version,
            panel=(LensId.ADVERSARIAL, LensId.FIRST_PRINCIPLES),
            tasks={LensId.ADVERSARIAL: task_returning(_succeeded(ids))},  # missing one
            store=store,
            id_generator=ids,
        )


# --- typing / re-export sanity check ----------------------------------------


def test_round1_result_is_re_exported_and_typed() -> None:
    # The synthesis layer consumes Round1Result; the controller must export it.
    assert Round1Result.__name__ == "Round1Result"
    # LensRunTimeout is the new outcome variant the controller introduces.
    t = LensRunTimeout()
    assert t.status == "timeout"


async def test_clock_injectable_for_dispatch_event_timestamp(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    fixed = datetime(2026, 6, 1, tzinfo=UTC)
    result = await run_round_1(
        session_id=brief.session_id,
        brief_version=brief.version,
        panel=(LensId.FIRST_PRINCIPLES,),
        tasks={LensId.FIRST_PRINCIPLES: task_returning(_succeeded(ids))},
        store=store,
        id_generator=ids,
        now=lambda: fixed,
    )
    assert result.dispatch_event.timestamp == fixed
