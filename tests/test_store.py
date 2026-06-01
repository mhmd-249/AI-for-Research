"""InMemorySessionStore: round-trips every domain object, dedups sources, and
isolates stored state from caller references."""

from __future__ import annotations

from datetime import datetime

from research_council.domain.source_ops import canonical_source_id
from research_council.enums import (
    Decision,
    LensId,
    LensRunStatus,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_challenge_id,
    new_dispatch_event_id,
    new_finding_id,
    new_lens_run_id,
    new_synthesis_id,
    new_trace_record_id,
    new_verdict_id,
    new_verification_result_id,
)
from research_council.models import (
    AgreementCluster,
    Brief,
    Challenge,
    ConfidenceBreakdown,
    DispatchEvent,
    Finding,
    LensRun,
    SelfUseLog,
    Session,
    Source,
    Synthesis,
    TraceRecord,
    Verdict,
    VerdictRef,
    VerificationResult,
)
from research_council.store import InMemorySessionStore


def test_session_round_trip(store: InMemorySessionStore, session: Session) -> None:
    store.save_session(session)
    assert store.get_session(session.id) == session


def test_brief_versions_round_trip(store: InMemorySessionStore, brief: Brief) -> None:
    v2 = brief.model_copy(update={"version": 2})
    store.save_brief(brief)
    store.save_brief(v2)

    assert store.get_brief(brief.session_id, 1) == brief
    assert store.get_latest_brief(brief.session_id) == v2
    assert [b.version for b in store.list_brief_versions(brief.session_id)] == [1, 2]


def test_finding_round_trip(store: InMemorySessionStore, finding: Finding) -> None:
    store.save_finding(finding)
    assert store.get_finding(finding.id) == finding


def test_lens_run_and_findings_join(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    store.save_finding(finding)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=1,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(finding.id,),
        disagreements_with_my_own_framing=("maybe I'm wrong",),
        brief_summary="summary",
    )
    store.save_lens_run(run)

    assert store.get_lens_run(run.id) == run
    assert store.list_findings_for_lens_run(run.id) == [finding]
    assert store.list_lens_runs_for_brief(session.id, 1, round=1) == [run]
    assert store.list_lens_runs_for_lens(session.id, LensId.FIRST_PRINCIPLES, 1) == [run]


def test_list_findings_for_session_includes_superseded(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    # A challenge revision creates a new Finding pointing back via supersedes;
    # the session-level query is literal and must return BOTH the original and
    # its successor (winners-resolution is a domain-layer concern, not the store's).
    revised = finding.model_copy(update={"id": new_finding_id(ids), "supersedes": finding.id})
    store.save_finding(finding)
    store.save_finding(revised)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=1,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(finding.id, revised.id),
    )
    store.save_lens_run(run)

    found = store.list_findings_for_session(session.id)
    assert {f.id for f in found} == {finding.id, revised.id}


def test_source_dedup_in_store(store: InMemorySessionStore) -> None:
    cid = canonical_source_id(arxiv_id="2307.03172")
    first = Source(canonical_id=cid, title="Lost in the Middle", arxiv_id="2307.03172")
    dup = Source(canonical_id=cid, title="DIFFERENT TITLE SAME PAPER", arxiv_id="2307.03172v2")

    stored = store.upsert_source(first)
    again = store.upsert_source(dup)

    assert stored == first
    assert again == first  # first-writer-wins; dup deduped to the existing source
    assert store.get_source(cid) == first


def test_dispatch_event_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, now: datetime
) -> None:
    event = DispatchEvent(
        id=new_dispatch_event_id(ids),
        session_id=session.id,
        brief_version=1,
        panel=(LensId.PRIOR_ART, LensId.ADVERSARIAL),
        timestamp=now,
    )
    store.save_dispatch_event(event)
    assert store.get_dispatch_event(event.id) == event
    assert store.list_dispatch_events(session.id) == [event]


def test_verification_result_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, finding: Finding
) -> None:
    result = VerificationResult(
        id=new_verification_result_id(ids),
        finding_id=finding.id,
        status=VerificationStatus.VERIFIED,
        claim_checked=finding.claim_text,
        verifier_version="v1",
    )
    store.save_verification_result(result)
    assert store.list_verification_results_for_finding(finding.id) == [result]


def test_challenge_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    challenge = Challenge(
        id=new_challenge_id(ids),
        session_id=session.id,
        challenged_finding_ref=finding.id,
        challenge_text="I think this is wrong because...",
    )
    store.save_challenge(challenge)
    assert store.get_challenge(challenge.id) == challenge
    assert store.list_challenges(session.id) == [challenge]


def test_synthesis_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, finding: Finding
) -> None:
    synthesis = Synthesis(
        id=new_synthesis_id(ids),
        session_id=session.id,
        brief_version=1,
        agreement_clusters=(
            AgreementCluster(
                finding_refs=(finding.id,),
                count=1,
                confidence_breakdown=ConfidenceBreakdown(load_bearing=1),
            ),
        ),
    )
    store.save_synthesis(synthesis)
    assert store.get_latest_synthesis(session.id) == synthesis


def test_verdict_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session
) -> None:
    verdict = Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.ABANDONED,
        outcome_summary="Killed by the circularity gap.",
        predictions_validated=(
            VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity"),
        ),
    )
    store.save_verdict(verdict)
    assert store.get_verdict(session.id) == verdict


def test_self_use_log_round_trip(store: InMemorySessionStore, session: Session) -> None:
    log = SelfUseLog(
        session_ref=session.id,
        time_to_dispatch_minutes=12.0,
        total_user_attention_minutes=45.0,
        acted_on_output=True,
        trusted_output=True,
    )
    store.save_self_use_log(log)
    assert store.get_self_use_log(session.id) == log


def test_trace_round_trip(
    store: InMemorySessionStore, ids: SequentialIdGenerator, now: datetime
) -> None:
    record = TraceRecord(
        id=new_trace_record_id(ids),
        caller_ref="lensrun_0001",
        prompt="system + user",
        completion="tool_use emit_output",
        model="claude-opus-4-8",
        timestamp=now,
    )
    store.append_trace(record)
    assert store.list_traces("lensrun_0001") == [record]
    assert store.list_traces("nobody") == []


def test_store_isolates_caller_from_stored_state(
    store: InMemorySessionStore, session: Session
) -> None:
    with_notes = session.model_copy(update={"notes": ("one",)})
    store.save_session(with_notes)

    first = store.get_session(session.id)
    second = store.get_session(session.id)
    assert first is not None and second is not None

    # The store deep-copies on the way out: each read is a distinct object, so a
    # caller can never reach into stored state. (The persisted collections are
    # tuples now, immutable by construction — the isolation is belt-and-braces.)
    assert first is not second
    assert first.notes == ("one",)
    assert store.get_session(session.id).notes == ("one",)  # type: ignore[union-attr]
