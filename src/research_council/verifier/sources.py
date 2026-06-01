"""Source-resolution clients and the co-primary tiebreak (PRD stories 31-32).

The verifier queries Semantic Scholar and arXiv as co-primary sources in parallel,
OpenAlex as secondary, and Crossref as a DOI-resolution fallback. Each registered
client implements :class:`SourceClient` and either returns a :class:`SourceCandidate`
(canonical identity + body text we will string-match against) or raises one of the
explicit failure exceptions below.

The tiebreak when both co-primary clients hit:

  * prefer the source with full text;
  * if both have full text, prefer S2 (better-structured metadata);
  * if neither has full text, prefer arXiv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import Source


@dataclass(frozen=True)
class SourceHint:
    """What a lens or the runtime hands the verifier to resolve a source. At
    least one identifier-ish field should be present; otherwise resolution falls
    through to ``source_not_found``."""

    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    first_author: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class SourceCandidate:
    """A resolved source plus the body text the verifier will hand to the judge
    in Stage 2/3 — and string-match the quoted passage against in Stage 3."""

    source: Source
    body: str  # full text if available; otherwise the abstract


# --- Explicit failure exceptions (story 38) ---------------------------------


class SourceClientError(Exception):
    """Base for client-raised failures the verifier translates to sub-reasons."""


class RateLimitedError(SourceClientError):
    """Transient: retry with backoff (3 attempts) before persisting verifier_error."""


class ApiDownError(SourceClientError):
    """Service unreachable: persist verifier_error with ``api_down`` immediately
    (retry-later-bulk is a controller concern, not the verifier's)."""


class PaywalledError(SourceClientError):
    """Paper resolved but no abstract / no usable body. Permanent for this claim
    against this source — persist ``paper_paywalled_no_abstract``."""


class ParseClientError(SourceClientError):
    """Malformed response from the client. Single retry, then persist ``parse_error``."""


# --- Client protocol --------------------------------------------------------


class SourceClient(Protocol):
    """A named source backend (s2 / arxiv / openalex / crossref). The verifier
    dispatches it with a :class:`SourceHint` and expects either a
    :class:`SourceCandidate`, ``None`` (no hit), or one of the four exceptions."""

    @property
    def name(self) -> str: ...

    async def query(self, hint: SourceHint) -> SourceCandidate | None: ...


@dataclass(frozen=True)
class SourceRegistry:
    """The four backends the verifier knows about (story 31)."""

    s2: SourceClient
    arxiv: SourceClient
    openalex: SourceClient
    crossref: SourceClient


# --- Tiebreak (story 32) ----------------------------------------------------


def tiebreak(
    s2_hit: SourceCandidate | None, arxiv_hit: SourceCandidate | None
) -> SourceCandidate | None:
    """Apply the co-primary tiebreak. Returns the winner, or ``None`` if both miss."""
    if s2_hit is None and arxiv_hit is None:
        return None
    if s2_hit is None:
        return arxiv_hit
    if arxiv_hit is None:
        return s2_hit
    s2_full = s2_hit.source.has_full_text
    ax_full = arxiv_hit.source.has_full_text
    if s2_full and not ax_full:
        return s2_hit
    if ax_full and not s2_full:
        return arxiv_hit
    if s2_full and ax_full:
        return s2_hit  # both full text → prefer S2 (better metadata)
    return arxiv_hit  # neither full text → prefer arXiv
