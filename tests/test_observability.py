"""TraceStore and observability: backward pointer-graph walk for synthesis
claims and partial replay (single LensRun / verifier check). Story 162-165."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from research_council.enums import (
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    Mode,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_finding_id,
    new_lens_run_id,
    new_session_id,
    new_synthesis_id,
    new_trace_record_id,
    new_verification_result_id,
)
from research_council.models import (
    AgreementCluster,
    Brief,
    ConfidenceBreakdown,
    Finding,
    LensRun,
    Session,
    Source,
    SourceRef,
    Synthesis,
    TraceRecord,
    VerificationResult,
)
from research_council.observability import (
    FindingTrace,
    reconstruct_lens_run_inputs,
    replay_lens_run,
    replay_verifier_check,
    walk_finding,
    walk_findings,
    walk_synthesis_claim,
)
from research_council.runtime import LensRunSucceeded
from research_council.runtime.llm import FakeLlmClient, LlmResponse, TextBlock, ToolUseBlock
from research_council.store import InMemorySessionStore

VALID_FIRST_PRINCIPLES: dict[str, Any] = {
    "findings": [
        {
            "claim_text": "The minimal mechanism is selective gating, not a separate filter agent.",
            "claim_type": "mechanism_hypothesis",
            "confidence": "load_bearing",
            "failure_modes_if_wrong": "If gating cannot be learned end-to-end, this fails.",
        }
    ],
    "disagreements_with_my_own_framing": ["I may be discounting engineering constraints."],
    "brief_summary": "The proposal adds a filter agent to prune the context window.",
}


def emit(payload: dict[str, Any], use_id: str = "u1") -> LlmResponse:
    return LlmResponse(content=[ToolUseBlock(id=use_id, name="emit_output", input=payload)])


# --- helpers ----------------------------------------------------------------


def _seed_session_with_brief(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> tuple[Session, Brief]:
    session = Session(id=new_session_id(ids), title="ECF")
    store.save_session(session)
    brief = Brief(
        session_id=session.id,
        version=1,
        mode=Mode.DEEP_DIVE,
        problem_statement="Long-context models under-attend to mid-context information.",
        researcher_context="AI engineer.",
        success_criteria_for_deliberation="Surface gaps I haven't found.",
        scope_and_non_scope="In scope: architecture.",
        proposed_solution="Active Context Inhibition via filter agent.",
        confirmed=True,
    )
    store.save_brief(brief)
    return session, brief


def _seed_finding_with_run_and_traces(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    brief: Brief,
    source: Source,
) -> tuple[Finding, LensRun, VerificationResult, list[TraceRecord]]:
    store.upsert_source(source)
    run_id = new_lens_run_id(ids)
    finding = Finding(
        id=new_finding_id(ids),
        claim_text="The filter agent has a circularity problem.",
        claim_type=ClaimType.GAP,
        confidence=Confidence.LOAD_BEARING,
        sources=(SourceRef(canonical_id=source.canonical_id),),
        failure_modes_if_wrong="If the filter is cheap, the circularity argument weakens.",
        verification_status=VerificationStatus.UNVERIFIED,
        round=1,
    )
    store.save_finding(finding)
    run = LensRun(
        id=run_id,
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=brief.version,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(finding.id,),
        brief_summary="summary",
        disagreements_with_my_own_framing=("maybe I'm wrong",),
    )
    store.save_lens_run(run)
    verification = VerificationResult(
        id=new_verification_result_id(ids),
        finding_id=finding.id,
        status=VerificationStatus.VERIFIED,
        claim_checked=finding.claim_text,
        source_canonical_id=source.canonical_id,
        quoted_passage="circularity passage",
        verifier_version="v1",
    )
    store.save_verification_result(verification)
    traces = [
        TraceRecord(
            id=new_trace_record_id(ids),
            caller_ref=run.id,
            prompt="SYSTEM: ... USER: brief...",
            completion="emit_output(...)",
            model="claude-opus-4-8",
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
    ]
    for trace in traces:
        store.append_trace(trace)
    return finding, run, verification, traces


# --- trace-view: backward graph walk (story 162) ----------------------------


def test_walk_finding_reconstructs_full_backward_chain(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    finding, run, verification, traces = _seed_finding_with_run_and_traces(
        store, ids, session, brief, source
    )

    trace = walk_finding(store, store, finding.id)

    assert trace is not None
    assert trace.finding == finding
    assert trace.emitting_run == run
    assert trace.brief == brief
    assert trace.verifications == [verification]
    assert trace.sources == [source]
    assert trace.traces == traces


def test_walk_finding_returns_none_for_unknown_id(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    assert walk_finding(store, store, new_finding_id(ids)) is None


def test_walk_finding_handles_finding_without_emitting_run(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Background-claim Findings (carried on a Brief) have no emitting LensRun.
    orphan = Finding(
        id=new_finding_id(ids),
        claim_text="Background claim.",
        claim_type=ClaimType.PRIOR_ART,
        confidence=Confidence.SUPPORTING,
        failure_modes_if_wrong="...",
    )
    store.save_finding(orphan)

    trace = walk_finding(store, store, orphan.id)

    assert trace is not None
    assert trace.finding == orphan
    assert trace.emitting_run is None
    assert trace.brief is None
    assert trace.verifications == []
    assert trace.sources == []
    assert trace.traces == []


def test_walk_synthesis_claim_walks_all_finding_refs(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    finding, _run, _v, _traces = _seed_finding_with_run_and_traces(
        store, ids, session, brief, source
    )

    cluster = AgreementCluster(
        finding_refs=(finding.id,),
        count=1,
        confidence_breakdown=ConfidenceBreakdown(load_bearing=1),
    )
    synthesis = Synthesis(
        id=new_synthesis_id(ids),
        session_id=session.id,
        brief_version=brief.version,
        agreement_clusters=(cluster,),
    )
    store.save_synthesis(synthesis)

    chain = walk_synthesis_claim(store, store, cluster.finding_refs)

    assert len(chain) == 1
    assert isinstance(chain[0], FindingTrace)
    assert chain[0].finding == finding
    assert chain[0].brief == brief
    assert chain[0].verifications and chain[0].verifications[0].source_canonical_id == (
        source.canonical_id
    )


def test_walk_findings_filters_missing_ids(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    finding, _run, _v, _traces = _seed_finding_with_run_and_traces(
        store, ids, session, brief, source
    )
    missing = new_finding_id(ids)

    chain = walk_findings(store, store, [finding.id, missing])

    assert len(chain) == 1
    assert chain[0].finding == finding


# --- TraceStore: append-only, unbounded (stories 163-164) -------------------


def test_trace_table_is_append_only_and_unbounded(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # Append many trace records; none are evicted (no retention logic for v0).
    moment = datetime(2026, 6, 1, tzinfo=UTC)
    for _ in range(100):
        store.append_trace(
            TraceRecord(
                id=new_trace_record_id(ids),
                caller_ref="lensrun_42",
                prompt="p",
                completion="c",
                model="claude-opus-4-8",
                timestamp=moment,
            )
        )
    assert len(store.list_traces("lensrun_42")) == 100


# --- partial replay: single LensRun (story 165) -----------------------------


def test_reconstruct_lens_run_inputs_round_1(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    _f, run, _v, _t = _seed_finding_with_run_and_traces(store, ids, session, brief, source)

    reconstructed = reconstruct_lens_run_inputs(store, run.id)

    assert reconstructed is not None
    assert reconstructed.run == run
    assert reconstructed.inputs.brief == brief
    assert reconstructed.inputs.lens_config.id is LensId.FIRST_PRINCIPLES
    assert reconstructed.inputs.round == 1
    assert reconstructed.inputs.prior_self == []
    assert reconstructed.inputs.cross_pollination == []


def test_reconstruct_lens_run_inputs_round_2_populates_prior_self_and_peers(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, brief = _seed_session_with_brief(store, ids)

    # A Round 1 run by FIRST_PRINCIPLES whose findings become prior_self.
    fp_round1_finding = Finding(
        id=new_finding_id(ids),
        claim_text="r1 fp finding",
        claim_type=ClaimType.GAP,
        confidence=Confidence.SUPPORTING,
        failure_modes_if_wrong="...",
        round=1,
    )
    store.save_finding(fp_round1_finding)
    fp_round1 = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=brief.version,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(fp_round1_finding.id,),
        brief_summary="fp r1",
        disagreements_with_my_own_framing=("d",),
    )
    store.save_lens_run(fp_round1)

    # A Round 1 run by ADVERSARIAL whose findings become cross-pollination.
    adv_round1_finding = Finding(
        id=new_finding_id(ids),
        claim_text="r1 adv finding",
        claim_type=ClaimType.FAILURE_MODE,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="...",
        round=1,
    )
    store.save_finding(adv_round1_finding)
    adv_round1 = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.ADVERSARIAL,
        session_id=session.id,
        brief_version=brief.version,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(adv_round1_finding.id,),
        brief_summary="adv r1",
        disagreements_with_my_own_framing=("d",),
        open_questions=("q?",),
    )
    store.save_lens_run(adv_round1)

    # The Round 2 run we want to replay.
    fp_round2 = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=brief.version,
        round=2,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(),
        brief_summary="fp r2",
        disagreements_with_my_own_framing=("d",),
    )
    store.save_lens_run(fp_round2)

    reconstructed = reconstruct_lens_run_inputs(store, fp_round2.id)

    assert reconstructed is not None
    assert reconstructed.inputs.round == 2
    # prior_self: this lens's prior findings, verification status stripped.
    assert len(reconstructed.inputs.prior_self) == 1
    assert reconstructed.inputs.prior_self[0].claim_text == "r1 fp finding"
    assert (
        reconstructed.inputs.prior_self[0].verification_status is VerificationStatus.UNVERIFIED
    )
    # cross-pollination: peer ADVERSARIAL's Round 1, anonymized.
    assert len(reconstructed.inputs.cross_pollination) == 1
    peer = reconstructed.inputs.cross_pollination[0]
    assert peer.label == "Lens A"  # neutral, lens identity stripped
    assert peer.findings[0].claim_text == "r1 adv finding"
    assert peer.brief_summary == "adv r1"


def test_reconstruct_lens_run_inputs_missing_run_returns_none(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    assert reconstruct_lens_run_inputs(store, new_lens_run_id(ids)) is None


def test_reconstruct_lens_run_inputs_missing_brief_returns_none(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session = Session(id=new_session_id(ids))
    store.save_session(session)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session.id,
        brief_version=1,
        round=1,
    )
    store.save_lens_run(run)
    assert reconstruct_lens_run_inputs(store, run.id) is None


async def test_replay_lens_run_redispatches_against_stored_brief(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    _f, run, _v, _t = _seed_finding_with_run_and_traces(store, ids, session, brief, source)

    client = FakeLlmClient([emit(VALID_FIRST_PRINCIPLES)])
    outcome = await replay_lens_run(
        store, run.id, client=client, id_generator=ids
    )

    assert isinstance(outcome, LensRunSucceeded)
    # The replay used the stored brief: that brief's problem_statement appears
    # in the prompt the FakeLlmClient saw.
    assert len(client.requests) == 1
    first_block = client.requests[0].messages[0].content[0]
    assert isinstance(first_block, TextBlock)
    assert brief.problem_statement in first_block.text
    # The replay used a non-tracing FakeLlmClient; the only TraceRecord under
    # this run's id is the one we seeded — no new ones were appended.
    assert len(store.list_traces(run.id)) == 1


async def test_replay_lens_run_unknown_id_returns_none(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    outcome = await replay_lens_run(
        store, new_lens_run_id(ids), client=FakeLlmClient([]), id_generator=ids
    )
    assert outcome is None


# --- partial replay: single verifier check (story 165) ----------------------


class _StubVerifier:
    """Captures the (finding, source) it was handed and returns a scripted result."""

    def __init__(
        self,
        result_factory: Callable[[Finding, Source | None], VerificationResult],
    ) -> None:
        self.result_factory = result_factory
        self.calls: list[tuple[Finding, Source | None]] = []

    async def verify(
        self, finding: Finding, source: Source | None
    ) -> VerificationResult:
        self.calls.append((finding, source))
        return self.result_factory(finding, source)


async def test_replay_verifier_check_runs_verifier_with_finding_and_source(
    store: InMemorySessionStore, ids: SequentialIdGenerator, source: Source
) -> None:
    session, brief = _seed_session_with_brief(store, ids)
    finding, _run, _v, _t = _seed_finding_with_run_and_traces(
        store, ids, session, brief, source
    )

    def _make(f: Finding, s: Source | None) -> VerificationResult:
        return VerificationResult(
            id=new_verification_result_id(ids),
            finding_id=f.id,
            status=VerificationStatus.VERIFIED,
            claim_checked=f.claim_text,
            source_canonical_id=s.canonical_id if s else None,
            verifier_version="v2",
        )

    verifier = _StubVerifier(_make)
    result = await replay_verifier_check(store, finding.id, verifier=verifier)

    assert result is not None
    assert result.finding_id == finding.id
    assert result.verifier_version == "v2"
    assert len(verifier.calls) == 1
    seen_finding, seen_source = verifier.calls[0]
    assert seen_finding == finding
    assert seen_source == source


async def test_replay_verifier_check_finding_without_source(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    finding = Finding(
        id=new_finding_id(ids),
        claim_text="A claim with no source.",
        claim_type=ClaimType.MECHANISM_HYPOTHESIS,
        confidence=Confidence.SUPPORTING,
        failure_modes_if_wrong="...",
    )
    store.save_finding(finding)

    def _make(f: Finding, s: Source | None) -> VerificationResult:
        return VerificationResult(
            id=new_verification_result_id(ids),
            finding_id=f.id,
            status=VerificationStatus.SOURCE_NOT_FOUND,
            claim_checked=f.claim_text,
            verifier_version="v1",
        )

    verifier = _StubVerifier(_make)
    result = await replay_verifier_check(store, finding.id, verifier=verifier)

    assert result is not None
    assert verifier.calls[0][1] is None


async def test_replay_verifier_check_missing_finding_returns_none(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    def _make(f: Finding, s: Source | None) -> VerificationResult:
        return VerificationResult(
            id=new_verification_result_id(ids),
            finding_id=f.id,
            status=VerificationStatus.VERIFIED,
            claim_checked="",
            verifier_version="v1",
        )

    verifier = _StubVerifier(_make)
    result = await replay_verifier_check(
        store, new_finding_id(ids), verifier=verifier
    )
    assert result is None
    assert verifier.calls == []
