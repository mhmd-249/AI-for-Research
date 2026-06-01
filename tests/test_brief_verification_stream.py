"""BriefVerificationStream — background verification of background_claims during
intake (PRD Layer 3.1, issue #7, stories 65 / 67 / 79 / 80).

Covers the four acceptance criteria:

  * background_claims verify in the background while intake continues; the
    submit call is silent (non-blocking) and the verification work overlaps
    with the intake conversation.
  * problems are surfaced only at the end of intake, with a 3-option choice
    (fix / replace / dispatch-with-unverified-tag) per problem.
  * chosen ``dispatch_unverified`` tags propagate into the dispatched brief —
    the Finding's ``verification_status`` is rewritten to ``unverified`` so
    lenses see it on the brief they receive.
  * empty background_claims makes the stream a no-op.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest

from research_council.domain.brief_verification import (
    PROBLEM_OPTIONS,
    BriefVerificationStream,
    ProblemAction,
)
from research_council.domain.source_ops import canonical_source_id
from research_council.enums import (
    ClaimType,
    Confidence,
    VerificationStatus,
)
from research_council.ids import (
    FindingId,
    SequentialIdGenerator,
    new_finding_id,
)
from research_council.models import Finding, Source
from research_council.verifier import (
    ApiDownError,
    ClaimToVerify,
    ClaimVerifier,
    EntailmentResult,
    InMemoryVerifierCache,
    Judge,
    LocalityResult,
    RetryConfig,
    SourceCandidate,
    SourceClient,
    SourceHint,
    SourceRegistry,
    TagAppropriatenessResult,
)

# --- Tiny fakes (mirrored from test_verifier.py to keep this file standalone) ----


@dataclass
class _ScriptedClient:
    name_: str
    script: list[SourceCandidate | None | Exception]
    calls: list[SourceHint] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.name_

    async def query(self, hint: SourceHint) -> SourceCandidate | None:
        self.calls.append(hint)
        if not self.script:
            return None
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _miss(name: str) -> _ScriptedClient:
    return _ScriptedClient(name_=name, script=[None])


def _source(*, title: str = "Lost in the Middle", arxiv: str = "2307.03172") -> Source:
    cid = canonical_source_id(arxiv_id=arxiv, title=title, first_author="Liu", year=2023)
    return Source(
        canonical_id=cid,
        title=title,
        authors=("Liu",),
        year=2023,
        arxiv_id=arxiv,
        has_full_text=True,
    )


@dataclass
class _FakeJudge:
    """Mirror of test_verifier.FakeJudge — covers the surface the stream tests need."""

    locality_result: LocalityResult = field(
        default_factory=lambda: LocalityResult(has_relevant_content=True)
    )
    entailment_result: EntailmentResult | None = None
    tag_result: TagAppropriatenessResult = field(
        default_factory=lambda: TagAppropriatenessResult(appropriate=True)
    )

    async def locality(
        self, *, source: Source, body: str, claim_text: str
    ) -> LocalityResult:
        return self.locality_result

    async def entailment(
        self, *, source: Source, body: str, claim_text: str
    ) -> EntailmentResult:
        if self.entailment_result is not None:
            return self.entailment_result
        return EntailmentResult(verdict="supports", quoted_passage=body)

    async def tag_appropriateness(
        self, *, claim_text: str, declared_type: ClaimType
    ) -> TagAppropriatenessResult:
        return self.tag_result


async def _noop_sleep(_: float) -> None:
    return None


def _registry(
    *, s2: SourceClient, arxiv: SourceClient | None = None
) -> SourceRegistry:
    return SourceRegistry(
        s2=s2,
        arxiv=arxiv or _miss("arxiv"),
        openalex=_miss("openalex"),
        crossref=_miss("crossref"),
    )


def _build_verifier(
    *,
    registry: SourceRegistry,
    judge: Judge | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ClaimVerifier:
    return ClaimVerifier(
        sources=registry,
        judge=judge or _FakeJudge(),
        cache=InMemoryVerifierCache(),
        verifier_version="v1.0",
        id_generator=SequentialIdGenerator(),
        retry=RetryConfig(rate_limited_attempts=3, parse_error_attempts=2,
                          backoff_seconds=0.0),
        sleep=sleep or _noop_sleep,
    )


def _claim(
    *,
    finding_id: FindingId | None = None,
    claim_text: str = "X reports 92% on benchmark Y.",
    claim_type: ClaimType = ClaimType.EMPIRICAL,
    arxiv_id: str | None = "2307.03172",
) -> ClaimToVerify:
    fid = finding_id or FindingId("finding_test")
    return ClaimToVerify(
        finding_id=fid,
        claim_text=claim_text,
        claim_type=claim_type,
        arxiv_id=arxiv_id,
    )


def _finding(
    *,
    fid: FindingId,
    claim_text: str = "X reports 92% on benchmark Y.",
    claim_type: ClaimType = ClaimType.EMPIRICAL,
) -> Finding:
    return Finding(
        id=fid,
        claim_text=claim_text,
        claim_type=claim_type,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong=(
            "If the citation doesn't say this, the load-bearing piece collapses."
        ),
        verification_status=VerificationStatus.UNVERIFIED,
    )


# --- Acceptance criterion 4: empty background_claims is a no-op ------------


async def test_empty_background_claims_makes_stream_a_noop() -> None:
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)

    assert stream.is_noop is True
    # Awaiting completion when nothing was submitted must not hang or raise.
    await stream.await_completion()
    assert stream.problems() == ()


# --- Acceptance criterion 1: background verification during intake ---------


async def test_submit_is_non_blocking_so_intake_continues() -> None:
    """submit() must schedule background work and return immediately; the
    intake conversation continues without awaiting verification."""

    # A slow source client: we hold it pending on an event so we can prove the
    # submit returned before any verification work happened.
    release = asyncio.Event()

    @dataclass
    class _SlowClient:
        name_: str
        candidate: SourceCandidate

        @property
        def name(self) -> str:
            return self.name_

        async def query(self, hint: SourceHint) -> SourceCandidate | None:
            await release.wait()
            return self.candidate

    src = _source()
    slow = _SlowClient(name_="s2", candidate=SourceCandidate(source=src, body="body text"))
    verifier = _build_verifier(registry=_registry(s2=slow))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_async")))
    # submit returned without waiting on the verifier.
    assert stream.is_noop is False
    # Verification has not produced a result yet.
    assert stream.results() == {}

    # Simulate intake doing other work, then releasing and awaiting at end of intake.
    release.set()
    await stream.await_completion()
    assert FindingId("finding_async") in stream.results()


async def test_multiple_claims_run_concurrently() -> None:
    """Two submits should run in parallel — the second doesn't wait on the first."""

    started = asyncio.Event()
    release = asyncio.Event()
    start_count = 0

    @dataclass
    class _Gated:
        name_: str
        candidate: SourceCandidate

        @property
        def name(self) -> str:
            return self.name_

        async def query(self, hint: SourceHint) -> SourceCandidate | None:
            nonlocal start_count
            start_count += 1
            if start_count >= 2:
                started.set()
            await release.wait()
            return self.candidate

    src = _source()
    s2 = _Gated(name_="s2", candidate=SourceCandidate(source=src, body="body text"))
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_a")))
    stream.submit(_claim(finding_id=FindingId("finding_b")))

    # Both queries should reach the gated point before either is released.
    await asyncio.wait_for(started.wait(), timeout=1.0)
    release.set()
    await stream.await_completion()
    assert {FindingId("finding_a"), FindingId("finding_b")} <= stream.results().keys()


# --- Acceptance criterion 2: problems surfaced with 3-option choice ---------


async def test_problem_options_are_fix_replace_dispatch_unverified() -> None:
    assert PROBLEM_OPTIONS == (
        ProblemAction.FIX,
        ProblemAction.REPLACE,
        ProblemAction.DISPATCH_UNVERIFIED,
    )


async def test_problems_surfaced_for_non_verified_results() -> None:
    """SOURCE_NOT_FOUND from a Stage-1 miss is a problem; each problem carries
    the three resolution options."""
    # All four backends return None → source_not_found
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_x")))
    await stream.await_completion()

    problems = stream.problems()
    assert len(problems) == 1
    problem = problems[0]
    assert problem.finding_id == FindingId("finding_x")
    assert problem.result.status is VerificationStatus.SOURCE_NOT_FOUND
    assert problem.options == PROBLEM_OPTIONS


async def test_verified_claims_are_not_problems() -> None:
    """A successful empirical resolution should not surface a problem."""
    src = _source()
    s2 = _ScriptedClient(
        name_="s2", script=[SourceCandidate(source=src, body="body text")]
    )
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_ok")))
    await stream.await_completion()
    assert stream.problems() == ()


async def test_partially_supported_is_not_a_problem() -> None:
    """partially_supported still goes to dispatch with that status; the user
    doesn't need to triage it the way contradicted / source_not_found does."""
    src = _source()
    s2 = _ScriptedClient(
        name_="s2", script=[SourceCandidate(source=src, body="body text")]
    )
    judge = _FakeJudge(
        entailment_result=EntailmentResult(
            verdict="partially_supports", quoted_passage="body text"
        )
    )
    verifier = _build_verifier(registry=_registry(s2=s2), judge=judge)
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_partial")))
    await stream.await_completion()
    assert stream.problems() == ()


async def test_contradicted_is_a_problem() -> None:
    """A direct contradiction is a problem the user must triage at end of intake."""
    src = _source()
    s2 = _ScriptedClient(
        name_="s2", script=[SourceCandidate(source=src, body="body text")]
    )
    judge = _FakeJudge(
        entailment_result=EntailmentResult(
            verdict="contradicts", quoted_passage="body text"
        )
    )
    verifier = _build_verifier(registry=_registry(s2=s2), judge=judge)
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_contra")))
    await stream.await_completion()

    problems = stream.problems()
    assert len(problems) == 1
    assert problems[0].result.status is VerificationStatus.CONTRADICTED


async def test_verifier_error_is_a_problem() -> None:
    """verifier_error (e.g. api_down) surfaces as a problem with 3 options."""

    @dataclass
    class _Down:
        name_: str

        @property
        def name(self) -> str:
            return self.name_

        async def query(self, hint: SourceHint) -> SourceCandidate | None:
            raise ApiDownError("backend out")

    s2 = _Down(name_="s2")
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_err")))
    await stream.await_completion()
    problems = stream.problems()
    assert len(problems) == 1
    assert problems[0].result.status is VerificationStatus.VERIFIER_ERROR


# --- Acceptance criterion 3: unverified-tag propagates into dispatch -------


async def test_dispatch_unverified_propagates_unverified_status_to_brief() -> None:
    """Story 65/79: the chosen ``dispatch_unverified`` tag carries into the
    brief's background_claims as Finding.verification_status=unverified, so
    every lens that reads the dispatched brief sees the tag."""
    # Source-not-found problem.
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)

    fid = FindingId("finding_unv")
    stream.submit(_claim(finding_id=fid))
    await stream.await_completion()
    [problem] = stream.problems()
    stream.accept_unverified(problem.finding_id)

    original = _finding(fid=fid)
    final = stream.apply_to([original])
    assert len(final) == 1
    assert final[0].verification_status is VerificationStatus.UNVERIFIED
    # Identity preserved — same FindingId, same claim text.
    assert final[0].id == fid
    assert final[0].claim_text == original.claim_text


async def test_apply_to_propagates_verified_status_for_clean_results() -> None:
    """Non-problem results write their actual status back to the Finding so the
    dispatched brief reflects what the verifier found."""
    src = _source()
    s2 = _ScriptedClient(
        name_="s2", script=[SourceCandidate(source=src, body="body text")]
    )
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    fid = FindingId("finding_ver")
    stream.submit(_claim(finding_id=fid))
    await stream.await_completion()

    original = _finding(fid=fid)
    final = stream.apply_to([original])
    assert final[0].verification_status is VerificationStatus.VERIFIED


async def test_accept_unverified_removes_the_problem_from_the_list() -> None:
    """Once the user resolves a problem with dispatch_unverified, it is no
    longer surfaced as needing triage."""
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)
    stream.submit(_claim(finding_id=FindingId("finding_p")))
    await stream.await_completion()

    assert len(stream.problems()) == 1
    stream.accept_unverified(FindingId("finding_p"))
    assert stream.problems() == ()


async def test_accept_unverified_unknown_id_rejected() -> None:
    """Refuse a resolution for a finding the stream never verified — a typo or
    stale UI state must not silently mark something as 'user accepted'."""
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)
    with pytest.raises(ValueError, match="no verification result"):
        stream.accept_unverified(FindingId("finding_ghost"))


# --- "fix" / "replace" — re-submitting a corrected claim -------------------


async def test_resubmit_after_fix_replaces_prior_result() -> None:
    """The 'fix' / 'replace' options translate to the user editing the field and
    the orchestrator re-submitting; the new verification supersedes the prior
    one keyed on the same FindingId."""
    src = _source()
    # First call: miss → source_not_found. Second call: hit → verified.
    s2 = _ScriptedClient(
        name_="s2",
        script=[None, SourceCandidate(source=src, body="body text")],
    )
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    fid = FindingId("finding_fix")
    stream.submit(_claim(finding_id=fid))
    await stream.await_completion()
    assert len(stream.problems()) == 1

    # User chooses "fix": orchestrator submits a corrected claim with the same id.
    stream.submit(
        _claim(
            finding_id=fid,
            claim_text="Sharpened claim text after user edit.",
        )
    )
    await stream.await_completion()
    assert stream.problems() == ()
    assert stream.results()[fid].status is VerificationStatus.VERIFIED


# --- Sequencing: silent during intake, surfaced only at end ----------------


async def test_results_not_available_until_await_completion() -> None:
    """During intake, partial results are not surfaced — they appear only after
    await_completion. Mirrors story 80's 'silent during intake'."""
    release = asyncio.Event()

    @dataclass
    class _Gated:
        name_: str
        candidate: SourceCandidate

        @property
        def name(self) -> str:
            return self.name_

        async def query(self, hint: SourceHint) -> SourceCandidate | None:
            await release.wait()
            return self.candidate

    src = _source()
    s2 = _Gated(name_="s2", candidate=SourceCandidate(source=src, body="body text"))
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_silent")))
    # Mid-intake: nothing has surfaced.
    assert stream.results() == {}
    assert stream.problems() == ()

    release.set()
    await stream.await_completion()
    assert FindingId("finding_silent") in stream.results()


# --- Idempotent termination ------------------------------------------------


async def test_await_completion_is_idempotent() -> None:
    """Calling await_completion twice (e.g. after a re-screen-2) must not hang
    or re-await already-finished tasks."""
    src = _source()
    s2 = _ScriptedClient(
        name_="s2", script=[SourceCandidate(source=src, body="body text")]
    )
    verifier = _build_verifier(registry=_registry(s2=s2))
    stream = BriefVerificationStream(verifier=verifier)

    stream.submit(_claim(finding_id=FindingId("finding_idem")))
    await stream.await_completion()
    await stream.await_completion()  # second call must be a no-op
    assert stream.results()[FindingId("finding_idem")].status is VerificationStatus.VERIFIED


# --- Apply-to fall-through for not-yet-verified or unrelated findings ------


async def test_apply_to_passes_through_unknown_findings(ids: SequentialIdGenerator) -> None:
    """A Finding not seen by the stream is returned unchanged. Empty stream
    against a Finding list is the identity transform."""
    verifier = _build_verifier(registry=_registry(s2=_miss("s2")))
    stream = BriefVerificationStream(verifier=verifier)
    fid = new_finding_id(ids)
    f = _finding(fid=fid)
    final = stream.apply_to([f])
    assert final == (f,)
