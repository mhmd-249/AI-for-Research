"""ChallengeOrchestrator — scoped single-Finding challenge loop (stories 102-116).

Covers the six issue acceptance criteria with a fake scoped-challenge runner and
either a fake or a real re-synthesizer:

* only the challenged Finding changes; the lens is not re-run wholesale;
* the stripped envelope is enforced (no brief_summary / disagreements);
* the supersedes chain hides the old Finding and the new one wins at render;
* sibling-impact flags surface as synthesis tensions;
* change_reason is captured, revised_reconsidered survives, reaffirmed is held;
* no brief-version bump, and re-synthesis fires on an accepted response.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from research_council.enums import (
    ChangeReason,
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    Mode,
    SessionStatus,
    TensionSource,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    SessionId,
    new_challenge_id,
    new_finding_id,
    new_lens_run_id,
    new_session_id,
    new_synthesis_id,
)
from research_council.models import (
    Brief,
    Challenge,
    Finding,
    FindingDraft,
    LensRun,
    ScopedChallengeOutputDraft,
    Session,
    Synthesis,
)
from research_council.runtime.challenge import (
    ChallengeResult,
    ScopedChallengeAccepted,
    ScopedChallengeInput,
    ScopedChallengeOutcome,
    ScopedChallengeRefused,
    make_resynthesizer,
    run_challenge,
)
from research_council.store import InMemorySessionStore
from research_council.synthesis import (
    AdjudicationVerdict,
    ComputedStructure,
    NarrationResult,
    RenderedSynthesis,
    SynthesisResult,
)

# --- re-synthesis fakes (mirror the clusterer/synthesizer test fakes) --------


class FakeEmbedder:
    """Unmapped text gets a unique one-hot vector (orthogonal → never merges)."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = dict(table)
        self._fallback_index = 1000

    def embed(self, text: str) -> list[float]:
        if text in self._table:
            return self._table[text]
        idx = self._fallback_index
        self._fallback_index += 1
        vec = [0.0] * (idx + 1)
        vec[idx] = 1.0
        return vec


class FakeAdjudicator:
    def __init__(self, *, default: AdjudicationVerdict = AdjudicationVerdict.DIFFERENT) -> None:
        self._default = default

    def adjudicate(self, finding_a: Finding, finding_b: Finding) -> AdjudicationVerdict:
        return self._default


class EchoNarrator:
    """An honest narrator: echoes the structure it was handed, adds no prose."""

    def narrate(self, structure: ComputedStructure) -> NarrationResult:
        return NarrationResult(
            agreement_cluster_refs=tuple(c.finding_refs for c in structure.agreement_clusters),
            declared_gap_refs=tuple(g.finding_refs for g in structure.declared_gaps),
        )


# --- builders ---------------------------------------------------------------


def _finding(
    ids: SequentialIdGenerator,
    text: str,
    *,
    claim_type: ClaimType = ClaimType.GAP,
    confidence: Confidence = Confidence.LOAD_BEARING,
) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=claim_type,
        confidence=confidence,
        failure_modes_if_wrong="...",
        verification_status=VerificationStatus.UNVERIFIED,
        round=1,
    )


def _draft(
    text: str = "On reflection, the circularity is partly resolvable.",
    *,
    claim_type: ClaimType = ClaimType.GAP,
    confidence: Confidence = Confidence.SUPPORTING,
) -> FindingDraft:
    return FindingDraft(
        claim_text=text,
        claim_type=claim_type,
        confidence=confidence,
        failure_modes_if_wrong="...",
    )


class FakeRunner:
    """Records the scoped input it was handed and returns a pre-arranged outcome."""

    def __init__(self, outcome: ScopedChallengeOutcome) -> None:
        self._outcome = outcome
        self.seen: ScopedChallengeInput | None = None

    async def run(self, inputs: ScopedChallengeInput) -> ScopedChallengeOutcome:
        self.seen = inputs
        return self._outcome


class CountingResynthesizer:
    """A re-synthesis seam that records how many times it fired."""

    def __init__(self, session_id: SessionId, ids: SequentialIdGenerator) -> None:
        self.calls = 0
        self._session_id = session_id
        self._ids = ids

    def __call__(self) -> SynthesisResult:
        self.calls += 1
        synthesis = Synthesis(
            id=new_synthesis_id(self._ids),
            session_id=self._session_id,
            brief_version=1,
        )
        rendered = RenderedSynthesis(
            primary=(),
            minor_speculative=(),
            emergent_gaps=(),
            agreements=(),
            anomalies=(),
            conditional_recommendations=(),
        )
        return SynthesisResult(
            synthesis=synthesis, rendered=rendered, narration_fell_back=False
        )


def _seed_session(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> tuple[Session, Brief]:
    session_id = new_session_id(ids)
    session = Session(
        id=session_id,
        status=SessionStatus.SYNTHESIZED,
        title="ECF",
        current_brief_version=1,
    )
    store.save_session(session)
    brief = Brief(
        session_id=session_id,
        version=1,
        mode=Mode.DEEP_DIVE,
        problem_statement="Long-context under-attention.",
        researcher_context="AI engineer.",
        success_criteria_for_deliberation="Surface gaps.",
        scope_and_non_scope="In: architecture.",
        proposed_solution="A filter agent.",
    )
    store.save_brief(brief)
    return session, brief


def _seed_lens_run(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    *,
    session: Session,
    lens_id: LensId,
    findings: tuple[Finding, ...],
) -> LensRun:
    for f in findings:
        store.save_finding(f)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=lens_id,
        session_id=session.id,
        brief_version=1,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=tuple(f.id for f in findings),
        brief_summary="s",
        disagreements_with_my_own_framing=("d",),
    )
    store.save_lens_run(run)
    return run


def _challenge(
    ids: SequentialIdGenerator, session: Session, finding: Finding
) -> Challenge:
    return Challenge(
        id=new_challenge_id(ids),
        session_id=session.id,
        challenged_finding_ref=finding.id,
        challenge_text="I think the circularity dissolves if the filter is cheap.",
    )


# --- the stripped envelope is enforced (story 107) --------------------------


def test_stripped_envelope_rejects_brief_summary_and_disagreements() -> None:
    with pytest.raises(ValidationError):
        ScopedChallengeOutputDraft.model_validate(
            {
                "finding": _draft().model_dump(),
                "change_reason": ChangeReason.REAFFIRMED_AGAINST_CHALLENGE.value,
                "brief_summary": "a sneaky full re-run",
            }
        )
    with pytest.raises(ValidationError):
        ScopedChallengeOutputDraft.model_validate(
            {
                "finding": _draft().model_dump(),
                "change_reason": ChangeReason.UNCHANGED.value,
                "disagreements_with_my_own_framing": ["x"],
            }
        )


# --- supersedes chain: old hidden, new wins (stories 110, 111) --------------


def test_revision_supersedes_old_and_new_wins(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    runner = FakeRunner(
        ScopedChallengeAccepted(
            ScopedChallengeOutputDraft(
                finding=_draft(),
                change_reason=ChangeReason.REVISED_NEW_EVIDENCE,
            )
        )
    )
    resynth = CountingResynthesizer(session.id, SequentialIdGenerator())

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=runner,
            resynthesize=resynth,
            id_generator=ids,
        )
    )

    assert result.accepted is True
    assert result.response_finding is not None
    # The new Finding points back at the challenged one; both persisted.
    assert result.response_finding.supersedes == challenged.id
    assert result.response_finding.change_reason is ChangeReason.REVISED_NEW_EVIDENCE
    assert store.get_finding(challenged.id) is not None  # old kept (audit)
    # The scoped run references only the new Finding.
    assert result.scoped_run.finding_ids == (result.response_finding.id,)
    assert result.scoped_run.round is None


# --- only the challenged Finding can change (stories 105, 167) ---------------


def test_only_challenged_finding_changes_siblings_untouched(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    sibling = _finding(ids, "Latency is acceptable.", claim_type=ClaimType.MECHANISM_HYPOTHESIS)
    _seed_lens_run(
        store,
        ids,
        session=session,
        lens_id=LensId.FIRST_PRINCIPLES,
        findings=(challenged, sibling),
    )
    runner = FakeRunner(
        ScopedChallengeAccepted(
            ScopedChallengeOutputDraft(
                finding=_draft(), change_reason=ChangeReason.REVISED_RECONSIDERED
            )
        )
    )

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=runner,
            resynthesize=CountingResynthesizer(session.id, SequentialIdGenerator()),
            id_generator=ids,
        )
    )

    # The sibling is neither superseded nor altered.
    assert store.get_finding(sibling.id) == sibling
    assert result.response_finding is not None
    assert result.response_finding.supersedes == challenged.id
    # The scoped input carried the lens's prior self (context), never peers.
    assert runner.seen is not None
    prior_ids = {f.id for f in runner.seen.prior_self}
    assert challenged.id in prior_ids and sibling.id in prior_ids
    assert "circularity dissolves" in runner.seen.challenge_text


# --- change_reason captured; reaffirm is first-class (stories 108, 109, 114) -


def test_reaffirmed_against_challenge_is_supported(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    runner = FakeRunner(
        ScopedChallengeAccepted(
            ScopedChallengeOutputDraft(
                finding=_draft("The circularity problem stands; you are wrong."),
                change_reason=ChangeReason.REAFFIRMED_AGAINST_CHALLENGE,
            )
        )
    )

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=runner,
            resynthesize=CountingResynthesizer(session.id, SequentialIdGenerator()),
            id_generator=ids,
        )
    )

    assert result.change_reason is ChangeReason.REAFFIRMED_AGAINST_CHALLENGE
    assert result.response_finding is not None
    assert (
        result.response_finding.change_reason is ChangeReason.REAFFIRMED_AGAINST_CHALLENGE
    )


# --- no brief-version bump; re-synthesis fires on accept (stories 115, 116) --


def test_no_brief_bump_and_resynthesis_fires_on_accept(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    resynth = CountingResynthesizer(session.id, SequentialIdGenerator())

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=FakeRunner(
                ScopedChallengeAccepted(
                    ScopedChallengeOutputDraft(
                        finding=_draft(), change_reason=ChangeReason.REVISED_NEW_EVIDENCE
                    )
                )
            ),
            resynthesize=resynth,
            id_generator=ids,
        )
    )

    assert resynth.calls == 1
    assert result.challenge.bumps_brief_version is False
    assert result.challenge.response_ref == result.scoped_run.id
    # The new finding stays on brief version 1 — no bump (story 115).
    assert result.scoped_run.brief_version == 1
    assert store.get_brief(session.id, 2) is None


def test_refusal_does_not_change_findings_or_fire_resynthesis(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    resynth = CountingResynthesizer(session.id, SequentialIdGenerator())

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=FakeRunner(ScopedChallengeRefused("outside my frame")),
            resynthesize=resynth,
            id_generator=ids,
        )
    )

    assert result.accepted is False
    assert result.response_finding is None
    assert result.refusal_reason == "outside my frame"
    assert resynth.calls == 0
    assert result.scoped_run.status is LensRunStatus.REFUSED
    # No superseding finding was created; the original is still the only one.
    assert store.get_finding(challenged.id) == challenged


# --- end-to-end with the real synthesizer (stories 110, 112, 116) -----------


def test_end_to_end_supersession_and_sibling_flag_in_resynthesis(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "The filter agent has a circularity problem.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    runner = FakeRunner(
        ScopedChallengeAccepted(
            ScopedChallengeOutputDraft(
                finding=_draft(
                    "The circularity is partly resolvable on reflection.",
                ),
                change_reason=ChangeReason.REVISED_RECONSIDERED,
                sibling_impact_flags=[
                    "This likely undermines my earlier point about G."
                ],
            )
        )
    )
    resynth = make_resynthesizer(
        store=store,
        session_id=session.id,
        brief_version=1,
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
        narrator=EchoNarrator(),
        id_generator=ids,
    )

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=runner,
            resynthesize=resynth,
            id_generator=ids,
        )
    )

    assert result.resynthesis is not None
    syn = result.resynthesis.synthesis
    winner_ids = {fid for g in syn.declared_gaps for fid in g.finding_refs}
    # The challenged Finding is gone from synthesis; the new one wins.
    assert challenged.id not in winner_ids
    assert result.response_finding is not None
    assert result.response_finding.id in winner_ids
    # The sibling-impact flag surfaced as a tension.
    sibling = [
        t for t in syn.tensions if t.source is TensionSource.SIBLING_IMPACT_FLAG
    ]
    assert len(sibling) == 1
    assert "undermines my earlier point about G" in sibling[0].description
    # The persisted synthesis is the latest one for the session.
    assert store.get_latest_synthesis(session.id) is not None


def test_withdrawn_response_removes_claim_from_synthesis(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "No training signal exists for the filter agent.")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    runner = FakeRunner(
        ScopedChallengeAccepted(
            ScopedChallengeOutputDraft(
                finding=_draft("I withdraw the training-signal gap."),
                change_reason=ChangeReason.WITHDRAWN,
            )
        )
    )
    resynth = make_resynthesizer(
        store=store,
        session_id=session.id,
        brief_version=1,
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
        narrator=EchoNarrator(),
        id_generator=ids,
    )

    result = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=runner,
            resynthesize=resynth,
            id_generator=ids,
        )
    )

    assert result.resynthesis is not None
    syn = result.resynthesis.synthesis
    all_refs = {
        fid for g in syn.declared_gaps for fid in g.finding_refs
    } | {fid for c in syn.agreement_clusters for fid in c.finding_refs}
    # Both the challenged claim and its tombstone are absent from synthesis.
    assert challenged.id not in all_refs
    assert result.response_finding is not None
    assert result.response_finding.id not in all_refs


def test_missing_finding_raises(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    session, _ = _seed_session(store, ids)
    ghost = _finding(ids, "never persisted")
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(
            run_challenge(
                challenge=_challenge(ids, session, ghost),
                store=store,
                runner=FakeRunner(ScopedChallengeRefused("x")),
                resynthesize=CountingResynthesizer(session.id, SequentialIdGenerator()),
                id_generator=ids,
            )
        )


def test_challenge_result_is_immutable(
    store: InMemorySessionStore, ids: SequentialIdGenerator
) -> None:
    # frozen dataclass: result cannot be mutated after the orchestrator returns.
    session, _ = _seed_session(store, ids)
    challenged = _finding(ids, "x")
    _seed_lens_run(
        store, ids, session=session, lens_id=LensId.FIRST_PRINCIPLES, findings=(challenged,)
    )
    result: ChallengeResult = asyncio.run(
        run_challenge(
            challenge=_challenge(ids, session, challenged),
            store=store,
            runner=FakeRunner(ScopedChallengeRefused("nope")),
            resynthesize=CountingResynthesizer(session.id, SequentialIdGenerator()),
            id_generator=ids,
        )
    )
    with pytest.raises(AttributeError):
        result.accepted = True  # type: ignore[misc]
