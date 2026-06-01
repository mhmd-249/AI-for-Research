"""Round 2 controller (Layer 3.4, stories 95-101).

Cross-pollination over anonymized Round 1 outputs. Tests cover the five issue
acceptance criteria: mandatory + opt-in participant selection, completion-order
anonymization with the adversarial lens unmarked among its peers, peers'
``disagreements_with_my_own_framing`` withheld, the <3-success skip, and the
>70% failure note that never halts synthesis.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from research_council.enums import (
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_finding_id,
    new_lens_run_id,
)
from research_council.models import Brief, Finding, LensRun
from research_council.runtime import (
    AnonymizedPeer,
    LensRunOutcome,
    LensRunRefused,
    LensRunSchemaInvalid,
    LensRunSucceeded,
    LensRunTimeout,
    LensTask,
    ProgressCallback,
    Round2Result,
    run_round_2,
)
from research_council.runtime.round2 import MANDATORY_PARTICIPANTS
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


def _succeeded_r1_run(
    lens_id: LensId,
    brief: Brief,
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    *,
    brief_summary: str = "summary",
    open_questions: tuple[str, ...] = ("q?",),
    disagreements: tuple[str, ...] = ("doubt",),
    finding_text: str = "claim",
) -> LensRun:
    finding = _finding(finding_text, ids)
    store.save_finding(finding)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=lens_id,
        session_id=brief.session_id,
        brief_version=brief.version,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(finding.id,),
        open_questions=open_questions,
        disagreements_with_my_own_framing=disagreements,
        brief_summary=brief_summary,
    )
    store.save_lens_run(run)
    return run


def _succeeded_outcome(
    ids: SequentialIdGenerator, text: str = "r2 claim"
) -> LensRunSucceeded:
    return LensRunSucceeded(
        findings=[_finding(text, ids)],
        open_questions=["r2 q?"],
        disagreements_with_my_own_framing=["r2 doubt"],
        brief_summary="r2 summary",
    )


def _task_returning(outcome: LensRunOutcome) -> LensTask:
    async def _t(progress: ProgressCallback) -> LensRunOutcome:
        progress()
        return outcome

    return _t


class CapturingBuilder:
    """A task_builder that records every (lens_id, peers) call. Returns the
    pre-arranged outcome for each lens_id; raises if an un-arranged lens_id is
    asked for so a participant-selection bug surfaces loudly."""

    def __init__(self, outcomes: dict[LensId, LensRunOutcome]) -> None:
        self.calls: list[tuple[LensId, list[AnonymizedPeer]]] = []
        self._outcomes = outcomes

    def __call__(self, lens_id: LensId, peers: list[AnonymizedPeer]) -> LensTask:
        self.calls.append((lens_id, peers))
        if lens_id not in self._outcomes:
            raise AssertionError(f"task_builder called for unexpected lens {lens_id}")
        return _task_returning(self._outcomes[lens_id])


def _three_non_mandatory(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> tuple[LensRun, ...]:
    """Three succeeded R1 runs that are NOT mandatory participants and have no
    opt-in phrase."""
    return (
        _succeeded_r1_run(LensId.ARCHITECTURE, brief, store, ids, brief_summary="arch"),
        _succeeded_r1_run(
            LensId.FIRST_PRINCIPLES, brief, store, ids, brief_summary="fp"
        ),
        _succeeded_r1_run(
            LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids, brief_summary="mi"
        ),
    )


# --- mandatory participants (story 95) --------------------------------------


async def test_mandatory_participants_are_adversarial_and_prior_art(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = _three_non_mandatory(brief, store, ids)
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )

    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    assert not result.skipped
    assert set(result.participants) == {LensId.ADVERSARIAL, LensId.PRIOR_ART}


async def test_mandatory_participate_even_when_they_had_no_round_1_success(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Adversarial and prior-art did NOT succeed in Round 1; three other lenses did.
    # Both mandatory still participate in Round 2 — they always do.
    r1 = _three_non_mandatory(brief, store, ids)
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )

    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    assert set(result.participants) == {LensId.ADVERSARIAL, LensId.PRIOR_ART}


# --- opt-in heuristic (story 96) --------------------------------------------


async def test_opt_in_lens_with_other_lenses_phrase_participates(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = (
        _succeeded_r1_run(
            LensId.ARCHITECTURE,
            brief,
            store,
            ids,
            disagreements=(
                "I would change if other lenses said the gating cannot be learned.",
            ),
        ),
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
        _succeeded_r1_run(LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids),
    )
    builder = CapturingBuilder(
        {
            LensId.ADVERSARIAL: _succeeded_outcome(ids),
            LensId.PRIOR_ART: _succeeded_outcome(ids),
            LensId.ARCHITECTURE: _succeeded_outcome(ids),
        }
    )

    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    assert set(result.participants) == {
        LensId.ADVERSARIAL,
        LensId.PRIOR_ART,
        LensId.ARCHITECTURE,
    }


async def test_opt_in_phrase_is_case_insensitive(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = (
        _succeeded_r1_run(
            LensId.ARCHITECTURE,
            brief,
            store,
            ids,
            disagreements=("This Would CHANGE If Other Lenses Said the gate works.",),
        ),
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
        _succeeded_r1_run(LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids),
    )
    builder = CapturingBuilder(
        {
            LensId.ADVERSARIAL: _succeeded_outcome(ids),
            LensId.PRIOR_ART: _succeeded_outcome(ids),
            LensId.ARCHITECTURE: _succeeded_outcome(ids),
        }
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert LensId.ARCHITECTURE in result.participants


async def test_tier2_without_opt_in_phrase_is_excluded(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Generic self-doubt is not enough; the heuristic requires the explicit
    # "would change if other lenses..." reference (story 96).
    r1 = (
        _succeeded_r1_run(
            LensId.ARCHITECTURE,
            brief,
            store,
            ids,
            disagreements=("I might be wrong about the design space.",),
        ),
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
        _succeeded_r1_run(LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids),
    )
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )

    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    assert LensId.ARCHITECTURE not in result.participants


async def test_opt_in_only_counts_succeeded_round_1_runs(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Even if a tier-2 lens emitted the opt-in phrase, a failed Round 1 run is
    # not a basis for opting in (we'd have no findings to anonymize for it
    # anyway, and the lens's R1 status is non-succeeded).
    failed_arch = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.ARCHITECTURE,
        session_id=brief.session_id,
        brief_version=brief.version,
        round=1,
        status=LensRunStatus.SCHEMA_INVALID,
        raw_output="{}",
    )
    store.save_lens_run(failed_arch)
    r1 = (
        failed_arch,
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
        _succeeded_r1_run(LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids),
        _succeeded_r1_run(LensId.INFORMATION_THEORETIC, brief, store, ids),
    )
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )

    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert LensId.ARCHITECTURE not in result.participants


# --- anonymization in completion order (story 97) ---------------------------


async def test_peers_are_anonymized_in_completion_order(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Three succeeded R1 runs in a specific completion order.
    r1 = (
        _succeeded_r1_run(LensId.ARCHITECTURE, brief, store, ids, brief_summary="A_BS"),
        _succeeded_r1_run(
            LensId.FIRST_PRINCIPLES, brief, store, ids, brief_summary="B_BS"
        ),
        _succeeded_r1_run(
            LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids, brief_summary="C_BS"
        ),
    )
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )
    await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    # Every participant sees the same anonymized peer list, in completion order.
    for _lens_id, peers in builder.calls:
        assert [p.label for p in peers] == ["Lens A", "Lens B", "Lens C"]
        assert [p.brief_summary for p in peers] == ["A_BS", "B_BS", "C_BS"]


async def test_adversarials_own_round1_appears_unmarked_among_peers(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Adversarial is one of the succeeded R1 runs. In its Round 2 cross_pollination,
    # its own output appears as a plain "Lens X" with no distinguishing marker
    # (story 97).
    r1 = (
        _succeeded_r1_run(LensId.ARCHITECTURE, brief, store, ids, brief_summary="ARCH"),
        _succeeded_r1_run(LensId.ADVERSARIAL, brief, store, ids, brief_summary="ADV"),
        _succeeded_r1_run(
            LensId.FIRST_PRINCIPLES, brief, store, ids, brief_summary="FP"
        ),
    )
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )
    await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    adv_call = next(c for c in builder.calls if c[0] is LensId.ADVERSARIAL)
    _, peers = adv_call

    # The adversarial's own R1 (brief_summary="ADV") sits in the second position
    # (Lens B) — the same neutral label as everyone else.
    self_as_peer = next(p for p in peers if p.brief_summary == "ADV")
    assert self_as_peer.label == "Lens B"
    # No marker betrays which peer is "yourself".
    for p in peers:
        label = p.label.lower()
        assert "self" not in label
        assert "you" not in label
        assert "adversarial" not in label


# --- disagreements field withheld (story 98) --------------------------------


async def test_peers_disagreements_field_is_withheld(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # A Round 1 lens emitted a secret self-doubt; it must NOT appear anywhere
    # in the AnonymizedPeer view passed to Round 2 participants.
    secret = "secret self-doubt that must not leak to peers"
    r1 = (
        _succeeded_r1_run(
            LensId.ARCHITECTURE, brief, store, ids, disagreements=(secret,)
        ),
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
        _succeeded_r1_run(LensId.MECHANISTIC_INTERPRETABILITY, brief, store, ids),
    )
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )

    await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )

    for _lens_id, peers in builder.calls:
        for p in peers:
            # The AnonymizedPeer type literally has no disagreements field.
            assert not hasattr(p, "disagreements_with_my_own_framing")
            # And the secret string is nowhere in the peer view.
            blob = (
                p.brief_summary
                + " ".join(p.open_questions)
                + " ".join(f.claim_text for f in p.findings)
            )
            assert secret not in blob


# --- round and responding_to (story 99) -------------------------------------


async def test_round_2_lens_runs_persist_with_round_2(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = _three_non_mandatory(brief, store, ids)
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert len(result.runs) == 2
    for run in result.runs:
        assert run.round == 2
        persisted = store.get_lens_run(run.id)
        assert persisted is not None
        assert persisted.round == 2


# --- skip when <3 successful Round 1 runs (story 101) -----------------------


async def test_skip_when_fewer_than_three_round_1_successes(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = (
        _succeeded_r1_run(LensId.ARCHITECTURE, brief, store, ids),
        _succeeded_r1_run(LensId.FIRST_PRINCIPLES, brief, store, ids),
    )
    # The builder should never be called when we skip.
    builder = CapturingBuilder({})
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert result.skipped
    assert result.runs == ()
    assert result.participants == ()
    assert result.dispatch_event is None
    assert builder.calls == []
    assert "Round 2 skipped" in result.synthesis_input_summary
    assert "2" in result.synthesis_input_summary  # the actual success count


async def test_skip_when_only_failed_round_1_runs(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Zero R1 successes: definitely <3.
    r1 = (
        LensRun(
            id=new_lens_run_id(ids),
            lens_id=LensId.ARCHITECTURE,
            session_id=brief.session_id,
            brief_version=brief.version,
            round=1,
            status=LensRunStatus.TIMEOUT,
        ),
    )
    builder = CapturingBuilder({})
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert result.skipped


# --- >70% failure note, no halt (story 100) ---------------------------------


async def test_over_70_percent_failure_emits_note_does_not_halt(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = _three_non_mandatory(brief, store, ids)
    # Both mandatory R2 participants fail (2/2 = 100% > 70%).
    builder = CapturingBuilder(
        {
            LensId.ADVERSARIAL: LensRunSchemaInvalid(
                raw_output="{}", validation_error="bad"
            ),
            LensId.PRIOR_ART: LensRunTimeout(),
        }
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    # The controller still returns normally — synthesis is not halted.
    assert result.failure_note is not None
    assert "cross-pollination failed" in result.failure_note.lower()
    assert len(result.runs) == 2
    # And the runs are persisted with their actual failure statuses.
    statuses = {r.status for r in result.runs}
    assert statuses == {LensRunStatus.SCHEMA_INVALID, LensRunStatus.TIMEOUT}


async def test_below_threshold_failure_has_no_note(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # 1 of 2 fails = 50%, not > 70%; no note.
    r1 = _three_non_mandatory(brief, store, ids)
    builder = CapturingBuilder(
        {
            LensId.ADVERSARIAL: _succeeded_outcome(ids),
            LensId.PRIOR_ART: LensRunRefused(reason="not in frame"),
        }
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert result.failure_note is None


# --- dispatch + persistence --------------------------------------------------


async def test_dispatch_event_for_round_2_is_persisted(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = _three_non_mandatory(brief, store, ids)
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
    )
    assert result.dispatch_event is not None
    persisted = store.get_dispatch_event(result.dispatch_event.id)
    assert persisted is not None
    assert set(persisted.panel) == {LensId.ADVERSARIAL, LensId.PRIOR_ART}
    assert set(persisted.lens_run_ids) == {r.id for r in result.runs}


async def test_dispatches_round_2_in_parallel(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Two tasks that meet at a barrier; serial dispatch would deadlock.
    r1 = _three_non_mandatory(brief, store, ids)
    barrier = asyncio.Barrier(2)

    def at_barrier(outcome: LensRunOutcome) -> LensTask:
        async def _t(progress: ProgressCallback) -> LensRunOutcome:
            progress()
            await barrier.wait()
            return outcome

        return _t

    captured: dict[LensId, LensTask] = {
        LensId.ADVERSARIAL: at_barrier(_succeeded_outcome(ids)),
        LensId.PRIOR_ART: at_barrier(_succeeded_outcome(ids)),
    }

    def builder(lens_id: LensId, peers: list[AnonymizedPeer]) -> LensTask:  # noqa: ARG001
        return captured[lens_id]

    result = await asyncio.wait_for(
        run_round_2(
            session_id=brief.session_id,
            brief_version=brief.version,
            round1_runs_in_completion_order=r1,
            store=store,
            task_builder=builder,
            id_generator=ids,
            hard_timeout_seconds=2.0,
            no_progress_timeout_seconds=2.0,
        ),
        timeout=2.0,
    )
    assert all(r.status is LensRunStatus.SUCCEEDED for r in result.runs)


async def test_clock_injectable_for_dispatch_event_timestamp(
    brief: Brief, store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    r1 = _three_non_mandatory(brief, store, ids)
    fixed = datetime(2026, 6, 1, tzinfo=UTC)
    builder = CapturingBuilder(
        {lid: _succeeded_outcome(ids) for lid in MANDATORY_PARTICIPANTS}
    )
    result = await run_round_2(
        session_id=brief.session_id,
        brief_version=brief.version,
        round1_runs_in_completion_order=r1,
        store=store,
        task_builder=builder,
        id_generator=ids,
        now=lambda: fixed,
    )
    assert result.dispatch_event is not None
    assert result.dispatch_event.timestamp == fixed


# --- typing / re-export sanity check ----------------------------------------


def test_round2_result_is_re_exported_and_typed() -> None:
    assert Round2Result.__name__ == "Round2Result"
