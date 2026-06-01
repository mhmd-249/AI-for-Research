"""The 3-stage ClaimVerifier — PRD Layer 2.1, stories 31-41.

``ClaimVerifier.verify`` runs:

  1. **Tag-appropriateness shortcut** (story 35) — for ``mechanism_hypothesis`` /
     ``gap`` / ``failure_mode`` claims (``unverifiable_by_design``), no source
     lookup at all; a single LLM call confirms the claim text matches its
     declared tag, so lenses cannot dodge verification by mistagging empirical
     claims as hypotheses. A mistagged claim is downgraded to ``verifier_error``.
  2. **Stage 1 — source resolution** (stories 31-32). Semantic Scholar and
     arXiv are queried in parallel; OpenAlex is secondary; Crossref is the DOI
     fallback. The co-primary tiebreak from :func:`sources.tiebreak` decides.
  3. **Stage 2 — locality**. Does the source bear on the claim? ``False`` is a
     distinct outcome from Stage 3 contradiction (``source_not_found`` with a
     resolved source id) — "paper doesn't address this" vs. "paper addresses
     it but the claim summarized it wrong."
  4. **Stage 3 — entailment**. The judge returns ``supports`` /
     ``partially_supports`` / ``contradicts`` plus a verbatim quoted passage.
     Before persisting, a string-match check confirms the passage actually
     appears in the source body — a hallucinated quote downgrades the result
     to ``verifier_error`` / ``passage_hallucinated`` (story 40).

Failure handling is explicit, never silent: each :class:`SourceClientError`
subclass maps to a documented sub-reason. ``rate_limited`` retries with
backoff (3 attempts); ``parse_error`` retries once; ``api_down`` and
``paper_paywalled_no_abstract`` persist immediately. ``verifier_error`` is
NEVER converted to ``verified`` (story 37).

Three caches (Source / Locality / Entailment) sit between the stages, each
keyed with ``verifier_version`` so a prompt or model upgrade invalidates
everything cleanly (story 36).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..domain.source_ops import canonical_source_id
from ..enums import ClaimType, VerificationStatus, VerifierErrorSubReason
from ..ids import (
    FindingId,
    IdGenerator,
    SourceId,
    new_verification_result_id,
)
from ..models import VerificationResult
from .cache import VerifierCache, claim_text_hash
from .judge import EntailmentResult, Judge
from .sources import (
    ApiDownError,
    ParseClientError,
    PaywalledError,
    RateLimitedError,
    SourceCandidate,
    SourceClient,
    SourceClientError,
    SourceHint,
    SourceRegistry,
    tiebreak,
)

UNVERIFIABLE_BY_DESIGN_TYPES: frozenset[ClaimType] = frozenset(
    {ClaimType.MECHANISM_HYPOTHESIS, ClaimType.GAP, ClaimType.FAILURE_MODE}
)


@dataclass(frozen=True)
class ClaimToVerify:
    """What the verifier receives per claim. Hints (doi / arxiv_id / openalex_id
    / title+author+year) drive Stage 1 resolution; for ``unverifiable_by_design``
    claim types they are ignored."""

    finding_id: FindingId
    claim_text: str
    claim_type: ClaimType
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    first_author: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class RetryConfig:
    """Tunable backoff for the explicit retry policy (story 38)."""

    rate_limited_attempts: int = 3
    parse_error_attempts: int = 2  # original + 1 retry
    backoff_seconds: float = 0.5


Sleep = Callable[[float], Awaitable[None]]


@dataclass
class _ResolutionOutcome:
    """Internal Stage 1 outcome: either a winning candidate or an explicit failure."""

    candidate: SourceCandidate | None = None
    sub_reason: VerifierErrorSubReason | None = None
    evidence: str = ""
    sources_tried: list[str] = field(default_factory=list)


class ClaimVerifier:
    def __init__(
        self,
        *,
        sources: SourceRegistry,
        judge: Judge,
        cache: VerifierCache,
        verifier_version: str,
        id_generator: IdGenerator,
        retry: RetryConfig | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self._sources = sources
        self._judge = judge
        self._cache = cache
        self._verifier_version = verifier_version
        self._ids = id_generator
        self._retry = retry or RetryConfig()
        self._sleep: Sleep = sleep or asyncio.sleep

    @property
    def verifier_version(self) -> str:
        return self._verifier_version

    # -- public surface ----------------------------------------------------

    async def verify(self, claim: ClaimToVerify) -> VerificationResult:
        # Tag-appropriateness short-circuit (story 35): no source lookup at all.
        if claim.claim_type in UNVERIFIABLE_BY_DESIGN_TYPES:
            return await self._verify_unverifiable_by_design(claim)
        return await self._verify_with_source(claim)

    # -- unverifiable-by-design path --------------------------------------

    async def _verify_unverifiable_by_design(self, claim: ClaimToVerify) -> VerificationResult:
        try:
            tag = await self._judge.tag_appropriateness(
                claim_text=claim.claim_text, declared_type=claim.claim_type
            )
        except Exception as exc:  # noqa: BLE001 — judge errors map to parse_error
            return self._error_result(
                claim,
                source_canonical_id=None,
                sub_reason=VerifierErrorSubReason.PARSE_ERROR,
                evidence=f"tag-appropriateness judge raised: {exc}",
            )

        if tag.appropriate:
            return self._build_result(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE_BY_DESIGN,
                source_canonical_id=None,
                quoted_passage=None,
                evidence=tag.reasoning or "claim text matches its declared tag",
                error_sub_reason=None,
            )
        # Mistagged: lens tried to dodge verification by mistagging an empirical
        # claim. Story 35 — caught and surfaced as verifier_error (parse_error
        # is the closest sub-reason: a structurally-invalid claim envelope).
        return self._error_result(
            claim,
            source_canonical_id=None,
            sub_reason=VerifierErrorSubReason.PARSE_ERROR,
            evidence=(
                "claim text does not match its declared unverifiable-by-design tag: "
                f"{tag.reasoning}"
            ),
        )

    # -- source-bearing path ----------------------------------------------

    async def _verify_with_source(self, claim: ClaimToVerify) -> VerificationResult:
        hint = SourceHint(
            doi=claim.doi,
            arxiv_id=claim.arxiv_id,
            openalex_id=claim.openalex_id,
            title=claim.title,
            first_author=claim.first_author,
            year=claim.year,
        )

        # Stage 1: source resolution -------------------------------------
        cached_source = self._maybe_cached_source(hint)
        if cached_source is not None:
            outcome = _ResolutionOutcome(candidate=cached_source, sources_tried=["cache"])
        else:
            outcome = await self._resolve_source(hint)
            if outcome.candidate is not None:
                self._cache.put_source(
                    canonical_id=outcome.candidate.source.canonical_id,
                    verifier_version=self._verifier_version,
                    value=outcome.candidate,
                )

        if outcome.candidate is None:
            if outcome.sub_reason is not None:
                return self._error_result(
                    claim,
                    source_canonical_id=None,
                    sub_reason=outcome.sub_reason,
                    evidence=outcome.evidence,
                )
            tried = ", ".join(outcome.sources_tried) or "none"
            return self._build_result(
                claim=claim,
                status=VerificationStatus.SOURCE_NOT_FOUND,
                source_canonical_id=None,
                quoted_passage=None,
                evidence=f"no source resolved (tried: {tried})",
                error_sub_reason=None,
            )

        candidate = outcome.candidate
        canonical = candidate.source.canonical_id
        chash = claim_text_hash(claim.claim_text)

        # Stage 2: locality ----------------------------------------------
        locality = self._cache.get_locality(
            canonical_id=canonical, claim_hash=chash, verifier_version=self._verifier_version
        )
        if locality is None:
            try:
                locality = await self._judge.locality(
                    source=candidate.source, body=candidate.body, claim_text=claim.claim_text
                )
            except Exception as exc:  # noqa: BLE001 — judge errors map to parse_error
                return self._error_result(
                    claim,
                    source_canonical_id=canonical,
                    sub_reason=VerifierErrorSubReason.PARSE_ERROR,
                    evidence=f"locality judge raised: {exc}",
                )
            self._cache.put_locality(
                canonical_id=canonical,
                claim_hash=chash,
                verifier_version=self._verifier_version,
                value=locality,
            )

        if not locality.has_relevant_content:
            # Stage 2 fail: paper resolved but doesn't address the claim. Distinct
            # from Stage 1's "citation doesn't resolve" by virtue of carrying a
            # source_canonical_id on the result (story 34).
            return self._build_result(
                claim=claim,
                status=VerificationStatus.SOURCE_NOT_FOUND,
                source_canonical_id=canonical,
                quoted_passage=None,
                evidence=(
                    "paper resolved but no passages bearing on the claim: "
                    f"{locality.note}"
                ).strip(),
                error_sub_reason=None,
            )

        # Stage 3: entailment --------------------------------------------
        entailment = self._cache.get_entailment(
            canonical_id=canonical,
            claim_hash=chash,
            claim_type=claim.claim_type,
            verifier_version=self._verifier_version,
        )
        if entailment is None:
            try:
                entailment = await self._judge.entailment(
                    source=candidate.source, body=candidate.body, claim_text=claim.claim_text
                )
            except Exception as exc:  # noqa: BLE001 — judge errors map to parse_error
                return self._error_result(
                    claim,
                    source_canonical_id=canonical,
                    sub_reason=VerifierErrorSubReason.PARSE_ERROR,
                    evidence=f"entailment judge raised: {exc}",
                )
            self._cache.put_entailment(
                canonical_id=canonical,
                claim_hash=chash,
                claim_type=claim.claim_type,
                verifier_version=self._verifier_version,
                value=entailment,
            )

        # Story 40: structural hallucination detection — quoted passage MUST
        # appear verbatim in the body. If not, downgrade.
        if not _passage_in_body(entailment.quoted_passage, candidate.body):
            return self._error_result(
                claim,
                source_canonical_id=canonical,
                sub_reason=VerifierErrorSubReason.PASSAGE_HALLUCINATED,
                evidence=(
                    "Stage 3 quoted passage not found in source body: "
                    f"{entailment.quoted_passage!r}"
                ),
            )

        status = _entailment_to_status(entailment)
        return self._build_result(
            claim=claim,
            status=status,
            source_canonical_id=canonical,
            quoted_passage=entailment.quoted_passage,
            evidence=entailment.reasoning,
            error_sub_reason=None,
        )

    # -- Stage 1 internals ------------------------------------------------

    def _maybe_cached_source(self, hint: SourceHint) -> SourceCandidate | None:
        canonical = _canonical_id_from_hint(hint)
        if canonical is None:
            return None
        return self._cache.get_source(
            canonical_id=canonical, verifier_version=self._verifier_version
        )

    async def _resolve_source(self, hint: SourceHint) -> _ResolutionOutcome:
        outcome = _ResolutionOutcome()

        # Co-primary: S2 + arXiv in parallel.
        s2_task = self._call_with_retries(self._sources.s2, hint, outcome)
        arxiv_task = self._call_with_retries(self._sources.arxiv, hint, outcome)
        outcome.sources_tried.extend([self._sources.s2.name, self._sources.arxiv.name])
        s2_hit, arxiv_hit = await asyncio.gather(s2_task, arxiv_task)

        winner = tiebreak(s2_hit, arxiv_hit)
        if winner is not None:
            outcome.candidate = winner
            return outcome

        # If a co-primary surfaced a non-retryable terminal error
        # (api_down / paywalled), stop here.
        if outcome.sub_reason is not None:
            return outcome

        # Secondary: OpenAlex.
        outcome.sources_tried.append(self._sources.openalex.name)
        oa_hit = await self._call_with_retries(self._sources.openalex, hint, outcome)
        if oa_hit is not None:
            outcome.candidate = oa_hit
            return outcome
        if outcome.sub_reason is not None:
            return outcome

        # DOI fallback: Crossref. Only meaningful when the hint actually has a DOI.
        if hint.doi:
            outcome.sources_tried.append(self._sources.crossref.name)
            cr_hit = await self._call_with_retries(self._sources.crossref, hint, outcome)
            if cr_hit is not None:
                outcome.candidate = cr_hit
                return outcome

        return outcome

    async def _call_with_retries(
        self, client: SourceClient, hint: SourceHint, outcome: _ResolutionOutcome
    ) -> SourceCandidate | None:
        """Run a client with the explicit retry policy from story 38.

        Failures terminate via the outcome's ``sub_reason`` (api_down, paywalled,
        rate_limited-after-N, parse_error-after-1-retry). Returning ``None``
        without a sub_reason means a clean miss.
        """
        rate_attempts = self._retry.rate_limited_attempts
        parse_attempts = self._retry.parse_error_attempts

        while True:
            try:
                return await client.query(hint)
            except RateLimitedError as exc:
                rate_attempts -= 1
                if rate_attempts <= 0:
                    outcome.sub_reason = VerifierErrorSubReason.RATE_LIMITED
                    outcome.evidence = f"{client.name} rate limited after retries: {exc}"
                    return None
                await self._sleep(self._retry.backoff_seconds)
            except ParseClientError as exc:
                parse_attempts -= 1
                if parse_attempts <= 0:
                    outcome.sub_reason = VerifierErrorSubReason.PARSE_ERROR
                    outcome.evidence = f"{client.name} parse error after retry: {exc}"
                    return None
                # one retry — loop continues
            except ApiDownError as exc:
                outcome.sub_reason = VerifierErrorSubReason.API_DOWN
                outcome.evidence = f"{client.name} api down: {exc}"
                return None
            except PaywalledError as exc:
                outcome.sub_reason = VerifierErrorSubReason.PAPER_PAYWALLED_NO_ABSTRACT
                outcome.evidence = f"{client.name} paywalled: {exc}"
                return None
            except SourceClientError as exc:
                # Unknown sub-class: surface as parse_error so it doesn't go silent.
                outcome.sub_reason = VerifierErrorSubReason.PARSE_ERROR
                outcome.evidence = f"{client.name} unknown SourceClientError: {exc}"
                return None

    # -- result construction ----------------------------------------------

    def _error_result(
        self,
        claim: ClaimToVerify,
        *,
        source_canonical_id: SourceId | None,
        sub_reason: VerifierErrorSubReason,
        evidence: str,
    ) -> VerificationResult:
        return self._build_result(
            claim=claim,
            status=VerificationStatus.VERIFIER_ERROR,
            source_canonical_id=source_canonical_id,
            quoted_passage=None,
            evidence=evidence,
            error_sub_reason=sub_reason,
        )

    def _build_result(
        self,
        *,
        claim: ClaimToVerify,
        status: VerificationStatus,
        source_canonical_id: SourceId | None,
        quoted_passage: str | None,
        evidence: str,
        error_sub_reason: VerifierErrorSubReason | None,
    ) -> VerificationResult:
        # Story 37: verifier_error is never silently converted to verified.
        # This is the single chokepoint — assert at the construction boundary.
        assert not (status == VerificationStatus.VERIFIED and error_sub_reason is not None), (
            "verifier_error must never collapse into verified (story 37)"
        )
        return VerificationResult(
            id=new_verification_result_id(self._ids),
            finding_id=claim.finding_id,
            status=status,
            claim_checked=claim.claim_text,
            source_canonical_id=source_canonical_id,
            quoted_passage=quoted_passage,
            evidence=evidence or None,
            error_sub_reason=error_sub_reason,
            verifier_version=self._verifier_version,
        )


# --- helpers ---------------------------------------------------------------


def _entailment_to_status(entailment: EntailmentResult) -> VerificationStatus:
    if entailment.verdict == "supports":
        return VerificationStatus.VERIFIED
    if entailment.verdict == "partially_supports":
        return VerificationStatus.PARTIALLY_SUPPORTED
    return VerificationStatus.CONTRADICTED


def _canonical_id_from_hint(hint: SourceHint) -> SourceId | None:
    if not any((hint.doi, hint.arxiv_id, hint.openalex_id, hint.title)):
        return None
    return canonical_source_id(
        doi=hint.doi,
        arxiv_id=hint.arxiv_id,
        openalex_id=hint.openalex_id,
        title=hint.title,
        first_author=hint.first_author,
        year=hint.year,
    )


def _passage_in_body(passage: str, body: str) -> bool:
    """Story 40 string-match. Whitespace-collapsed, case-insensitive — a quote
    is "verbatim" if it appears in the body modulo whitespace differences."""
    if not passage.strip():
        return False
    norm_passage = " ".join(passage.split()).lower()
    norm_body = " ".join(body.split()).lower()
    return norm_passage in norm_body


__all__ = [
    "UNVERIFIABLE_BY_DESIGN_TYPES",
    "ClaimToVerify",
    "ClaimVerifier",
    "RetryConfig",
]
