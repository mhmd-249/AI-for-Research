"""ClaimVerifier — the 3-stage per-claim verification pipeline (PRD Layer 2.1,
stories 31-41). Every acceptance criterion of issue #3 has a test here.

Test layering (read top-to-bottom):

  * Source-resolution tiebreak / parallel co-primary / secondary / DOI fallback
  * Stage-2 vs Stage-3 distinct failure semantics
  * Tag-appropriateness short-circuit and the mistagged-empirical catch
  * Three-level caching and verifier_version invalidation
  * Passage-hallucination structural downgrade
  * The four failure sub-reasons (rate_limited / api_down / paywalled / parse)
  * Status-reachability matrix (all seven VerificationStatus values reachable)
  * verifier_error never collapses into verified (story 37)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest

from research_council.domain.source_ops import canonical_source_id
from research_council.enums import (
    ClaimType,
    VerificationStatus,
    VerifierErrorSubReason,
)
from research_council.ids import (
    FindingId,
    SequentialIdGenerator,
)
from research_council.models import Source
from research_council.verifier import (
    ApiDownError,
    ClaimToVerify,
    ClaimVerifier,
    EntailmentResult,
    InMemoryVerifierCache,
    Judge,
    LocalityResult,
    ParseClientError,
    PaywalledError,
    RateLimitedError,
    RetryConfig,
    SourceCandidate,
    SourceClient,
    SourceClientError,
    SourceHint,
    SourceRegistry,
    TagAppropriatenessResult,
    tiebreak,
)

# --- helpers ----------------------------------------------------------------


def _source(*, title: str, arxiv: str | None = None, doi: str | None = None,
            openalex: str | None = None, has_full_text: bool = False) -> Source:
    cid = canonical_source_id(
        doi=doi, arxiv_id=arxiv, openalex_id=openalex, title=title, first_author="Liu", year=2023
    )
    return Source(
        canonical_id=cid,
        title=title,
        authors=("Liu",),
        year=2023,
        doi=doi,
        arxiv_id=arxiv,
        openalex_id=openalex,
        has_full_text=has_full_text,
    )


@dataclass
class ScriptedClient:
    """A SourceClient that returns a scripted candidate or raises a scripted
    exception per call. The script is a list; each call pops the head."""

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


def _miss(name: str = "x") -> ScriptedClient:
    return ScriptedClient(name_=name, script=[None])


def _registry(*, s2: SourceClient, arxiv: SourceClient,
              openalex: SourceClient | None = None,
              crossref: SourceClient | None = None) -> SourceRegistry:
    return SourceRegistry(
        s2=s2,
        arxiv=arxiv,
        openalex=openalex or _miss("openalex"),
        crossref=crossref or _miss("crossref"),
    )


@dataclass
class FakeJudge:
    """Records calls; returns scripted Stage 2 / Stage 3 / tag results. The
    default entailment quotes the first few words of the body, so the hallucination
    string-match always passes for tests that don't care about the quoted passage."""

    locality_result: LocalityResult = field(
        default_factory=lambda: LocalityResult(has_relevant_content=True)
    )
    entailment_result: EntailmentResult | None = None  # None ⇒ derive from body
    tag_result: TagAppropriatenessResult = field(
        default_factory=lambda: TagAppropriatenessResult(appropriate=True)
    )

    locality_calls: int = 0
    entailment_calls: int = 0
    tag_calls: int = 0
    raise_in_locality: Exception | None = None
    raise_in_entailment: Exception | None = None
    raise_in_tag: Exception | None = None

    async def locality(self, *, source: Source, body: str, claim_text: str) -> LocalityResult:
        self.locality_calls += 1
        if self.raise_in_locality:
            raise self.raise_in_locality
        return self.locality_result

    async def entailment(
        self, *, source: Source, body: str, claim_text: str
    ) -> EntailmentResult:
        self.entailment_calls += 1
        if self.raise_in_entailment:
            raise self.raise_in_entailment
        if self.entailment_result is not None:
            return self.entailment_result
        # Default: quote whatever the body actually contains.
        return EntailmentResult(verdict="supports", quoted_passage=body)

    async def tag_appropriateness(
        self, *, claim_text: str, declared_type: ClaimType
    ) -> TagAppropriatenessResult:
        self.tag_calls += 1
        if self.raise_in_tag:
            raise self.raise_in_tag
        return self.tag_result


def _claim(
    *,
    finding_id: str = "finding_0001",
    claim_text: str = "X reports 92% on benchmark Y.",
    claim_type: ClaimType = ClaimType.EMPIRICAL,
    arxiv_id: str | None = "2307.03172",
    doi: str | None = None,
    openalex_id: str | None = None,
    title: str | None = None,
) -> ClaimToVerify:
    return ClaimToVerify(
        finding_id=FindingId(finding_id),
        claim_text=claim_text,
        claim_type=claim_type,
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        title=title,
    )


def _build_verifier(
    *,
    registry: SourceRegistry,
    judge: Judge | None = None,
    cache: InMemoryVerifierCache | None = None,
    verifier_version: str = "v1.0",
    retry: RetryConfig | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ClaimVerifier:
    return ClaimVerifier(
        sources=registry,
        judge=judge or FakeJudge(),
        cache=cache or InMemoryVerifierCache(),
        verifier_version=verifier_version,
        id_generator=SequentialIdGenerator(),
        retry=retry or RetryConfig(rate_limited_attempts=3, parse_error_attempts=2,
                                   backoff_seconds=0.0),
        sleep=sleep or _noop_sleep,
    )


async def _noop_sleep(_: float) -> None:
    return None


# --- Stage 1: source resolution + tiebreak (stories 31-32) ------------------


async def test_tiebreak_prefers_s2_when_both_full_text() -> None:
    s2_src = _source(title="Lost in the Middle", arxiv="2307.03172", has_full_text=True)
    arxiv_src = _source(title="Lost in the Middle", arxiv="2307.03172", has_full_text=True)
    winner = tiebreak(
        SourceCandidate(source=s2_src, body="from s2"),
        SourceCandidate(source=arxiv_src, body="from arxiv"),
    )
    assert winner is not None and winner.body == "from s2"


async def test_tiebreak_prefers_arxiv_when_neither_full_text() -> None:
    s2_src = _source(title="T", arxiv="2307.03172", has_full_text=False)
    arxiv_src = _source(title="T", arxiv="2307.03172", has_full_text=False)
    winner = tiebreak(
        SourceCandidate(source=s2_src, body="from s2"),
        SourceCandidate(source=arxiv_src, body="from arxiv"),
    )
    assert winner is not None and winner.body == "from arxiv"


async def test_tiebreak_prefers_full_text_winner() -> None:
    s2_src = _source(title="T", arxiv="2307.03172", has_full_text=False)
    arxiv_src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    winner = tiebreak(
        SourceCandidate(source=s2_src, body="from s2"),
        SourceCandidate(source=arxiv_src, body="from arxiv"),
    )
    assert winner is not None and winner.body == "from arxiv"


async def test_co_primary_queried_in_parallel() -> None:
    """Both s2 and arxiv must be invoked on every empirical resolution attempt;
    we do not short-circuit one against the other."""
    arxiv_src = _source(title="T", arxiv="2307.03172", has_full_text=False)
    s2 = ScriptedClient(name_="s2", script=[None])
    arxiv = ScriptedClient(name_="arxiv", script=[SourceCandidate(arxiv_src, "body")])
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIED
    assert len(s2.calls) == 1
    assert len(arxiv.calls) == 1


async def test_openalex_secondary_when_co_primary_miss() -> None:
    openalex_src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = _miss("s2")
    arxiv = _miss("arxiv")
    oa = ScriptedClient(name_="openalex",
                        script=[SourceCandidate(openalex_src, "openalex body")])
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv, openalex=oa))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIED
    assert len(oa.calls) == 1


async def test_crossref_doi_fallback() -> None:
    crossref_src = _source(title="T", doi="10.1/x", has_full_text=False)
    s2 = _miss("s2")
    arxiv = _miss("arxiv")
    oa = _miss("openalex")
    cr = ScriptedClient(name_="crossref",
                        script=[SourceCandidate(crossref_src, "crossref body")])
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv, openalex=oa, crossref=cr))

    result = await verifier.verify(_claim(arxiv_id=None, doi="10.1/x"))
    assert result.status is VerificationStatus.VERIFIED
    assert len(cr.calls) == 1


async def test_crossref_skipped_when_no_doi() -> None:
    """Crossref is the DOI fallback specifically; with no DOI hint, it is not
    consulted (avoids burning DOI-API quota on title-only resolutions)."""
    s2 = _miss("s2")
    arxiv = _miss("arxiv")
    oa = _miss("openalex")
    cr = ScriptedClient(name_="crossref", script=[])
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv, openalex=oa, crossref=cr))

    result = await verifier.verify(_claim())  # no DOI
    assert result.status is VerificationStatus.SOURCE_NOT_FOUND
    assert len(cr.calls) == 0


async def test_all_apis_miss_yields_source_not_found() -> None:
    verifier = _build_verifier(registry=_registry(s2=_miss(), arxiv=_miss()))
    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.SOURCE_NOT_FOUND
    assert result.source_canonical_id is None  # Stage 1 fail: no resolved id
    assert result.error_sub_reason is None  # not an error: a real verdict


# --- Stage 2 vs Stage 3 distinct failure semantics (story 34) ---------------


async def test_stage_2_fail_distinct_from_stage_1_fail() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body about something else")])
    arxiv = _miss("arxiv")
    judge = FakeJudge(locality_result=LocalityResult(
        has_relevant_content=False, note="paper is about unrelated topic"
    ))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    # Stage 2 fail: source resolved (canonical_id set), but no bearing.
    assert result.status is VerificationStatus.SOURCE_NOT_FOUND
    assert result.source_canonical_id is not None  # the distinguishing field
    assert result.error_sub_reason is None
    assert "no passages bearing on the claim" in (result.evidence or "")
    # Stage 3 never ran.
    assert judge.entailment_calls == 0


async def test_stage_3_contradicts_yields_contradicted() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    body = "The benchmark showed 12%, not 92%."
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, body)])
    arxiv = _miss("arxiv")
    judge = FakeJudge(entailment_result=EntailmentResult(
        verdict="contradicts",
        quoted_passage="The benchmark showed 12%, not 92%.",
        reasoning="The paper reports a different number.",
    ))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.CONTRADICTED
    assert result.quoted_passage == "The benchmark showed 12%, not 92%."


async def test_stage_3_partial_yields_partially_supported() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    body = "Results reach 88% on a different benchmark."
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, body)])
    arxiv = _miss("arxiv")
    judge = FakeJudge(entailment_result=EntailmentResult(
        verdict="partially_supports",
        quoted_passage="Results reach 88% on a different benchmark.",
    ))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.PARTIALLY_SUPPORTED


# --- Tag-appropriateness short-circuit (story 35) ---------------------------


@pytest.mark.parametrize(
    "ct", [ClaimType.MECHANISM_HYPOTHESIS, ClaimType.GAP, ClaimType.FAILURE_MODE]
)
async def test_unverifiable_by_design_runs_only_tag_check(ct: ClaimType) -> None:
    s2 = ScriptedClient(name_="s2", script=[])  # would error if called
    arxiv = ScriptedClient(name_="arxiv", script=[])
    judge = FakeJudge()
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(
        _claim(claim_type=ct, claim_text="I think the cause is feature suppression.")
    )

    assert result.status is VerificationStatus.UNVERIFIABLE_BY_DESIGN
    assert judge.tag_calls == 1
    assert judge.locality_calls == 0
    assert judge.entailment_calls == 0
    assert len(s2.calls) == 0
    assert len(arxiv.calls) == 0


async def test_mistagged_empirical_as_hypothesis_is_caught() -> None:
    judge = FakeJudge(tag_result=TagAppropriatenessResult(
        appropriate=False,
        reasoning="claim cites a numeric benchmark result — this is empirical, not hypothesis",
    ))
    verifier = _build_verifier(registry=_registry(s2=_miss(), arxiv=_miss()), judge=judge)

    result = await verifier.verify(
        _claim(
            claim_type=ClaimType.MECHANISM_HYPOTHESIS,
            claim_text="Paper X reports 92% on benchmark Y.",
        )
    )

    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PARSE_ERROR
    assert "does not match" in (result.evidence or "")


# --- Three-level cache (story 36) -------------------------------------------


async def test_source_cache_hit_skips_api_calls() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body")])
    arxiv = _miss("arxiv")
    cache = InMemoryVerifierCache()
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), cache=cache)

    first = await verifier.verify(_claim())
    assert first.status is VerificationStatus.VERIFIED
    assert len(s2.calls) == 1

    # Second run: same claim hits cached source — s2 NOT consulted.
    second = await verifier.verify(_claim())
    assert second.status is VerificationStatus.VERIFIED
    assert len(s2.calls) == 1  # unchanged


async def test_locality_cache_hit_skips_locality_judge() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[
        SourceCandidate(src, "body"), SourceCandidate(src, "body"),
    ])
    arxiv = _miss("arxiv")
    judge = FakeJudge()
    cache = InMemoryVerifierCache()
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge, cache=cache)

    await verifier.verify(_claim())
    # Second run with same claim: source and locality both hit cache — locality
    # judge stays at one call.
    await verifier.verify(_claim())
    assert judge.locality_calls == 1


async def test_entailment_cache_keyed_on_claim_type() -> None:
    """Same (source, claim_text) but different claim_type ⇒ entailment runs again.
    This is the (source, claim, claim_type) discrimination required by story 36."""
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[
        SourceCandidate(src, "body"), SourceCandidate(src, "body"),
    ])
    arxiv = _miss("arxiv")
    judge = FakeJudge()
    cache = InMemoryVerifierCache()
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge, cache=cache)

    await verifier.verify(_claim(claim_type=ClaimType.EMPIRICAL))
    assert judge.entailment_calls == 1

    # Same claim text, different claim_type → entailment slot does not hit.
    await verifier.verify(_claim(claim_type=ClaimType.PRIOR_ART))
    assert judge.entailment_calls == 2


async def test_verifier_version_bump_invalidates_cache() -> None:
    """A prompt/model upgrade is signaled by bumping verifier_version; every
    cache layer must miss for the new version (story 36)."""
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[
        SourceCandidate(src, "body"), SourceCandidate(src, "body"),
    ])
    arxiv = _miss("arxiv")
    judge = FakeJudge()
    cache = InMemoryVerifierCache()

    v1 = _build_verifier(
        registry=_registry(s2=s2, arxiv=arxiv), judge=judge, cache=cache, verifier_version="v1"
    )
    await v1.verify(_claim())
    assert judge.locality_calls == 1
    assert judge.entailment_calls == 1

    # Bump the version: a fresh verifier sharing the same cache must miss every layer.
    v2 = _build_verifier(
        registry=_registry(s2=s2, arxiv=arxiv), judge=judge, cache=cache, verifier_version="v2"
    )
    await v2.verify(_claim())
    assert judge.locality_calls == 2
    assert judge.entailment_calls == 2


# --- Passage hallucination structural downgrade (story 40) ------------------


async def test_passage_hallucination_downgrades_to_verifier_error() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    body = "The paper reports a different result on a different benchmark."
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, body)])
    arxiv = _miss("arxiv")
    judge = FakeJudge(entailment_result=EntailmentResult(
        verdict="supports",
        quoted_passage="The benchmark showed 92% accuracy.",  # fabricated
        reasoning="quoted passage was hallucinated",
    ))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PASSAGE_HALLUCINATED


async def test_passage_present_modulo_whitespace_passes() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    body = "We achieve 92% accuracy on benchmark Y, surpassing prior work."
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, body)])
    arxiv = _miss("arxiv")
    judge = FakeJudge(entailment_result=EntailmentResult(
        verdict="supports", quoted_passage="We achieve\n92% accuracy on benchmark Y",
    ))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIED


# --- Failure sub-reasons (story 38) -----------------------------------------


async def test_rate_limited_retries_then_persists_sub_reason() -> None:
    """3 attempts, then ``rate_limited`` persists. The third call still raises;
    the verifier does not silently swallow it."""
    s2 = ScriptedClient(name_="s2", script=[
        RateLimitedError("attempt 1"),
        RateLimitedError("attempt 2"),
        RateLimitedError("attempt 3"),
    ])
    arxiv = _miss("arxiv")
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    verifier = _build_verifier(
        registry=_registry(s2=s2, arxiv=arxiv),
        retry=RetryConfig(rate_limited_attempts=3, parse_error_attempts=2,
                          backoff_seconds=0.25),
        sleep=fake_sleep,
    )
    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.RATE_LIMITED
    assert len(s2.calls) == 3  # exactly 3 attempts
    assert len(sleeps) == 2  # backoff fired between attempts 1→2 and 2→3


async def test_paper_paywalled_no_abstract_is_permanent() -> None:
    s2 = ScriptedClient(name_="s2", script=[PaywalledError("no body")])
    arxiv = _miss("arxiv")
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PAPER_PAYWALLED_NO_ABSTRACT
    assert len(s2.calls) == 1  # no retry


async def test_api_down_persists_sub_reason_immediately() -> None:
    s2 = ScriptedClient(name_="s2", script=[ApiDownError("503")])
    arxiv = _miss("arxiv")
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.API_DOWN
    assert len(s2.calls) == 1


async def test_parse_error_retries_once_then_persists() -> None:
    s2 = ScriptedClient(name_="s2", script=[
        ParseClientError("malformed 1"),
        ParseClientError("malformed 2"),
    ])
    arxiv = _miss("arxiv")
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PARSE_ERROR
    assert len(s2.calls) == 2  # one retry


async def test_unknown_source_client_error_surfaces_as_parse_error() -> None:
    """An unfamiliar SourceClientError subclass must not silently degrade to
    a successful resolution (story 37). It surfaces as parse_error."""
    class WeirdError(SourceClientError):
        pass

    s2 = ScriptedClient(name_="s2", script=[WeirdError("???")])
    arxiv = _miss("arxiv")
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv))

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PARSE_ERROR


async def test_judge_raising_in_locality_yields_verifier_error() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body")])
    arxiv = _miss("arxiv")
    judge = FakeJudge(raise_in_locality=RuntimeError("LLM died"))
    verifier = _build_verifier(registry=_registry(s2=s2, arxiv=arxiv), judge=judge)

    result = await verifier.verify(_claim())
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert result.error_sub_reason is VerifierErrorSubReason.PARSE_ERROR


# --- Status-reachability matrix (acceptance criterion: all 7 statuses) ------


async def test_all_seven_statuses_reachable() -> None:
    """Every one of the seven VerificationStatus values should be reachable
    through some verifier path. UNVERIFIED is the default Finding state (the
    verifier never produces it — by definition it persists a *result*)."""
    # UNVERIFIED: the default Finding state, not a verifier output.
    reachable = {VerificationStatus.UNVERIFIED}

    # VERIFIED
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body says X")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(entailment_result=EntailmentResult(
            verdict="supports", quoted_passage="body says X")),
    )
    reachable.add((await v.verify(_claim())).status)

    # PARTIALLY_SUPPORTED
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body says X")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(entailment_result=EntailmentResult(
            verdict="partially_supports", quoted_passage="body says X")),
    )
    reachable.add((await v.verify(_claim())).status)

    # CONTRADICTED
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body says X")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(entailment_result=EntailmentResult(
            verdict="contradicts", quoted_passage="body says X")),
    )
    reachable.add((await v.verify(_claim())).status)

    # SOURCE_NOT_FOUND (Stage 1 fail)
    v = _build_verifier(registry=_registry(s2=_miss(), arxiv=_miss()))
    reachable.add((await v.verify(_claim())).status)

    # UNVERIFIABLE_BY_DESIGN
    v = _build_verifier(registry=_registry(s2=_miss(), arxiv=_miss()))
    reachable.add((await v.verify(_claim(claim_type=ClaimType.MECHANISM_HYPOTHESIS))).status)

    # VERIFIER_ERROR
    s2 = ScriptedClient(name_="s2", script=[ApiDownError("down")])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    reachable.add((await v.verify(_claim())).status)

    assert reachable == set(VerificationStatus)


async def test_all_four_failure_sub_reasons_reachable() -> None:
    seen: set[VerifierErrorSubReason] = set()

    s2 = ScriptedClient(name_="s2", script=[
        RateLimitedError("1"), RateLimitedError("2"), RateLimitedError("3"),
    ])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    seen.add((await v.verify(_claim())).error_sub_reason)  # type: ignore[arg-type]

    s2 = ScriptedClient(name_="s2", script=[PaywalledError("p")])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    seen.add((await v.verify(_claim())).error_sub_reason)  # type: ignore[arg-type]

    s2 = ScriptedClient(name_="s2", script=[ApiDownError("a")])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    seen.add((await v.verify(_claim())).error_sub_reason)  # type: ignore[arg-type]

    s2 = ScriptedClient(name_="s2", script=[
        ParseClientError("1"), ParseClientError("2"),
    ])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    seen.add((await v.verify(_claim())).error_sub_reason)  # type: ignore[arg-type]

    expected: set[VerifierErrorSubReason] = {
        VerifierErrorSubReason.RATE_LIMITED,
        VerifierErrorSubReason.PAPER_PAYWALLED_NO_ABSTRACT,
        VerifierErrorSubReason.API_DOWN,
        VerifierErrorSubReason.PARSE_ERROR,
    }
    assert expected <= seen


# --- story 37: verifier_error never collapses into verified -----------------


async def test_verifier_error_carries_into_result_never_verified() -> None:
    """Even when the verifier hits an error path, the persisted result must NOT
    show ``verified``. This is the single anti-credibility-laundering invariant
    (story 37) — exercise every error path and assert."""
    error_results: list[VerificationStatus] = []

    # paywalled
    s2 = ScriptedClient(name_="s2", script=[PaywalledError("p")])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    error_results.append((await v.verify(_claim())).status)

    # api down
    s2 = ScriptedClient(name_="s2", script=[ApiDownError("a")])
    v = _build_verifier(registry=_registry(s2=s2, arxiv=_miss()))
    error_results.append((await v.verify(_claim())).status)

    # judge crash
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(raise_in_entailment=RuntimeError("died")),
    )
    error_results.append((await v.verify(_claim())).status)

    # passage hallucinated
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "real body content")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(entailment_result=EntailmentResult(
            verdict="supports", quoted_passage="totally not in body")),
    )
    error_results.append((await v.verify(_claim())).status)

    assert all(s is VerificationStatus.VERIFIER_ERROR for s in error_results)
    assert VerificationStatus.VERIFIED not in error_results


# --- VerificationResult shape --------------------------------------------


async def test_result_carries_verifier_version_and_finding_id() -> None:
    src = _source(title="T", arxiv="2307.03172", has_full_text=True)
    s2 = ScriptedClient(name_="s2", script=[SourceCandidate(src, "body says X")])
    v = _build_verifier(
        registry=_registry(s2=s2, arxiv=_miss()),
        judge=FakeJudge(entailment_result=EntailmentResult(
            verdict="supports", quoted_passage="body says X")),
        verifier_version="v9.9",
    )
    result = await v.verify(_claim(finding_id="finding_4242"))
    assert result.verifier_version == "v9.9"
    assert result.finding_id == FindingId("finding_4242")
    assert result.id.startswith("verification_")
    assert result.source_canonical_id is not None
    assert isinstance(result.source_canonical_id, str)


# --- claim_text_hash --------------------------------------------------------


def test_claim_text_hash_collapses_whitespace_and_case() -> None:
    from research_council.verifier import claim_text_hash

    a = claim_text_hash("X reports 92% on benchmark Y.")
    b = claim_text_hash("x  REPORTS\t92% on benchmark y.")
    c = claim_text_hash("X reports 88% on benchmark Y.")
    assert a == b
    assert a != c
