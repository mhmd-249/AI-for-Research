"""SqliteSessionStore-specific behavior: persistence across re-open (the
between-rounds resume acceptance, PRD story 146), global cross-session source
dedup (story 145), and persistence of the ``supersedes`` /
``reused_from_version`` markers and the Session status."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from research_council.domain.source_ops import canonical_source_id
from research_council.enums import (
    ChangeReason,
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    Mode,
    SessionStatus,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_dispatch_event_id,
    new_finding_id,
    new_lens_run_id,
    new_session_id,
)
from research_council.models import (
    Brief,
    DispatchEvent,
    Finding,
    LensRun,
    Session,
    Source,
)
from research_council.store import SqliteSessionStore


def test_resumes_session_state_between_rounds(tmp_path: Path) -> None:
    # The between-rounds resume contract (PRD story 146): once Round 1 reaches
    # the controller's terminal state, the session + brief + dispatch + every
    # successful LensRun + findings persist; reopening the store reloads them
    # at that round boundary so the controller can dispatch Round 2.
    db_path = tmp_path / "session.db"
    ids = SequentialIdGenerator()
    session = Session(
        id=new_session_id(ids), status=SessionStatus.CONSULTING, current_brief_version=1
    )
    brief = Brief(
        session_id=session.id,
        version=1,
        mode=Mode.DEEP_DIVE,
        problem_statement="P",
        researcher_context="R",
        success_criteria_for_deliberation="S",
        scope_and_non_scope="Sc",
        confirmed=True,
    )
    dispatch = DispatchEvent(
        id=new_dispatch_event_id(ids),
        session_id=session.id,
        brief_version=1,
        panel=(LensId.PRIOR_ART, LensId.ADVERSARIAL),
        timestamp=datetime(2026, 6, 1, tzinfo=UTC),
    )
    finding = Finding(
        id=new_finding_id(ids),
        claim_text="claim",
        claim_type=ClaimType.GAP,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="fm",
        verification_status=VerificationStatus.UNVERIFIED,
        round=1,
    )
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.PRIOR_ART,
        session_id=session.id,
        brief_version=1,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        dispatch_event_id=dispatch.id,
        finding_ids=(finding.id,),
        disagreements_with_my_own_framing=("self-doubt",),
        brief_summary="summary",
    )

    writer = SqliteSessionStore(db_path)
    try:
        writer.save_session(session)
        writer.save_brief(brief)
        writer.save_dispatch_event(dispatch)
        writer.save_finding(finding)
        writer.save_lens_run(run)
    finally:
        writer.close()

    # Resume: a fresh store instance against the same file recovers everything.
    reader = SqliteSessionStore(db_path)
    try:
        assert reader.get_session(session.id) == session
        assert reader.get_latest_brief(session.id) == brief
        assert reader.list_dispatch_events(session.id) == [dispatch]
        assert reader.get_lens_run(run.id) == run
        assert reader.list_findings_for_lens_run(run.id) == [finding]
        assert reader.list_findings_for_session(session.id) == [finding]
    finally:
        reader.close()


def test_session_status_transitions_persist(tmp_path: Path) -> None:
    # save_session(...) is the persistence step the session-status state machine
    # writes through; the SQLite store must reflect the most-recent status.
    db_path = tmp_path / "status.db"
    ids = SequentialIdGenerator()
    initial = Session(id=new_session_id(ids), status=SessionStatus.INTAKE)

    writer = SqliteSessionStore(db_path)
    try:
        writer.save_session(initial)
        writer.save_session(initial.model_copy(update={"status": SessionStatus.CONSULTING}))
        writer.save_session(initial.model_copy(update={"status": SessionStatus.SYNTHESIZED}))
    finally:
        writer.close()

    reader = SqliteSessionStore(db_path)
    try:
        reloaded = reader.get_session(initial.id)
        assert reloaded is not None
        assert reloaded.status == SessionStatus.SYNTHESIZED
    finally:
        reader.close()


def test_supersedes_pointer_persists_across_reopen(tmp_path: Path) -> None:
    # A challenge revision creates a new Finding with `supersedes` pointing at
    # the original (PRD story 110). Both findings must survive a reopen so the
    # audit / verdict / trace-walk readers see the literal stored set.
    db_path = tmp_path / "supersedes.db"
    ids = SequentialIdGenerator()
    original = Finding(
        id=new_finding_id(ids),
        claim_text="original",
        claim_type=ClaimType.MECHANISM_HYPOTHESIS,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="fm",
    )
    revised = original.model_copy(
        update={
            "id": new_finding_id(ids),
            "claim_text": "revised",
            "supersedes": original.id,
            "change_reason": ChangeReason.REVISED_NEW_EVIDENCE,
        }
    )

    writer = SqliteSessionStore(db_path)
    try:
        writer.save_finding(original)
        writer.save_finding(revised)
    finally:
        writer.close()

    reader = SqliteSessionStore(db_path)
    try:
        reloaded_original = reader.get_finding(original.id)
        reloaded_revised = reader.get_finding(revised.id)
        assert reloaded_original == original
        assert reloaded_revised == revised
        assert reloaded_revised is not None
        assert reloaded_revised.supersedes == original.id
        assert reloaded_revised.change_reason is ChangeReason.REVISED_NEW_EVIDENCE
    finally:
        reader.close()


def test_reused_from_version_marker_persists(tmp_path: Path) -> None:
    # A refinement-driven reuse re-points an existing LensRun at the new brief
    # version with `reused_from_version` set (PRD story 121). The marker must
    # round-trip across reopen so the synthesizer can render reused vs. fresh.
    db_path = tmp_path / "reused.db"
    ids = SequentialIdGenerator()
    session_id = new_session_id(ids)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.ARCHITECTURE,
        session_id=session_id,
        brief_version=2,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        reused_from_version=1,
        disagreements_with_my_own_framing=("self-doubt",),
        brief_summary="reused from v1",
    )

    writer = SqliteSessionStore(db_path)
    try:
        writer.save_lens_run(run)
    finally:
        writer.close()

    reader = SqliteSessionStore(db_path)
    try:
        reloaded = reader.get_lens_run(run.id)
        assert reloaded == run
        assert reloaded is not None
        assert reloaded.reused_from_version == 1
    finally:
        reader.close()


def test_sources_table_dedups_across_sessions(tmp_path: Path) -> None:
    # PRD story 145: the Sources table is GLOBAL — the same paper cited by a
    # Finding in session A and a Finding in session B resolves to one row,
    # not two per-session copies. The same connection sees the same source
    # regardless of which session's runs reference it.
    db_path = tmp_path / "sources.db"
    cid = canonical_source_id(arxiv_id="2307.03172")
    source = Source(
        canonical_id=cid, title="Lost in the Middle", arxiv_id="2307.03172", year=2023
    )

    store_a = SqliteSessionStore(db_path)
    try:
        # Session A inserts the source.
        store_a.upsert_source(source)
    finally:
        store_a.close()

    store_b = SqliteSessionStore(db_path)
    try:
        # Session B (later, after a resume) attempts to upsert the same paper
        # with a v2 arxiv id and a different title — the canonical id collapses
        # both into the originally-stored row (first-writer-wins).
        dup = Source(
            canonical_id=cid,
            title="DIFFERENT TITLE SAME PAPER",
            arxiv_id="2307.03172v2",
            year=2023,
        )
        returned = store_b.upsert_source(dup)
        assert returned == source  # the existing row wins
        assert store_b.get_source(cid) == source
    finally:
        store_b.close()


def test_in_memory_sqlite_isolated_per_instance() -> None:
    # Sanity check: a ":memory:" SqliteSessionStore is a usable transient store
    # (handy for tests/spikes) but doesn't share state with a sibling instance.
    a = SqliteSessionStore(":memory:")
    b = SqliteSessionStore(":memory:")
    try:
        ids = SequentialIdGenerator()
        s = Session(id=new_session_id(ids))
        a.save_session(s)
        assert a.get_session(s.id) == s
        assert b.get_session(s.id) is None
    finally:
        a.close()
        b.close()
