"""Three-level cache with ``verifier_version`` invalidation (PRD story 36).

Three independently-keyed caches at three stage boundaries:

  * **Source** — keyed on ``canonical_id``. Cached result is the
    :class:`SourceCandidate` from Stage 1.
  * **Locality** — keyed on ``(source_canonical_id, claim_text_hash)``. Cached
    result is the Stage 2 :class:`LocalityResult`.
  * **Entailment** — keyed on ``(source_canonical_id, claim_text_hash, claim_type)``.
    Cached result is the Stage 3 :class:`EntailmentResult`.

Every key also carries ``verifier_version`` — bumping the version is how a
prompt or model upgrade invalidates everything cleanly without manual eviction.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from ..enums import ClaimType
from ..ids import SourceId
from .judge import EntailmentResult, LocalityResult
from .sources import SourceCandidate


def claim_text_hash(claim_text: str) -> str:
    """Stable hash for a claim's text. Normalized: whitespace-collapsed lower-case.
    Two claims that differ only in spacing/case share a Locality / Entailment key."""
    normalized = " ".join(claim_text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class VerifierCache(Protocol):
    """Three-level cache. ``verifier_version`` is part of every key — a version
    bump partitions old entries off from new ones rather than evicting them."""

    # Source ---------------------------------------------------------------
    def get_source(
        self, *, canonical_id: SourceId, verifier_version: str
    ) -> SourceCandidate | None: ...

    def put_source(
        self, *, canonical_id: SourceId, verifier_version: str, value: SourceCandidate
    ) -> None: ...

    # Locality -------------------------------------------------------------
    def get_locality(
        self, *, canonical_id: SourceId, claim_hash: str, verifier_version: str
    ) -> LocalityResult | None: ...

    def put_locality(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        verifier_version: str,
        value: LocalityResult,
    ) -> None: ...

    # Entailment -----------------------------------------------------------
    def get_entailment(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        claim_type: ClaimType,
        verifier_version: str,
    ) -> EntailmentResult | None: ...

    def put_entailment(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        claim_type: ClaimType,
        verifier_version: str,
        value: EntailmentResult,
    ) -> None: ...


class InMemoryVerifierCache:
    """Plain-dict implementation. Satisfies :class:`VerifierCache`."""

    def __init__(self) -> None:
        self._sources: dict[tuple[SourceId, str], SourceCandidate] = {}
        self._locality: dict[tuple[SourceId, str, str], LocalityResult] = {}
        self._entailment: dict[tuple[SourceId, str, str, str], EntailmentResult] = {}

    # Source ---------------------------------------------------------------
    def get_source(
        self, *, canonical_id: SourceId, verifier_version: str
    ) -> SourceCandidate | None:
        return self._sources.get((canonical_id, verifier_version))

    def put_source(
        self, *, canonical_id: SourceId, verifier_version: str, value: SourceCandidate
    ) -> None:
        self._sources[(canonical_id, verifier_version)] = value

    # Locality -------------------------------------------------------------
    def get_locality(
        self, *, canonical_id: SourceId, claim_hash: str, verifier_version: str
    ) -> LocalityResult | None:
        return self._locality.get((canonical_id, claim_hash, verifier_version))

    def put_locality(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        verifier_version: str,
        value: LocalityResult,
    ) -> None:
        self._locality[(canonical_id, claim_hash, verifier_version)] = value

    # Entailment -----------------------------------------------------------
    def get_entailment(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        claim_type: ClaimType,
        verifier_version: str,
    ) -> EntailmentResult | None:
        return self._entailment.get(
            (canonical_id, claim_hash, claim_type.value, verifier_version)
        )

    def put_entailment(
        self,
        *,
        canonical_id: SourceId,
        claim_hash: str,
        claim_type: ClaimType,
        verifier_version: str,
        value: EntailmentResult,
    ) -> None:
        self._entailment[
            (canonical_id, claim_hash, claim_type.value, verifier_version)
        ] = value
