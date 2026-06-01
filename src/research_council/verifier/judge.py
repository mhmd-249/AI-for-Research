"""LLM-backed checks for Stages 2 and 3 plus the tag-appropriateness check
(PRD stories 33-35, 40).

The judge is the seam — production wires a Claude-backed implementation; tests
inject a deterministic fake. Keeping it as a small protocol means the verifier
pipeline can be tested end-to-end without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ..enums import ClaimType
from ..models import Source


@dataclass(frozen=True)
class LocalityResult:
    """Stage 2: do any passages in this source bear on the claim?"""

    has_relevant_content: bool
    note: str = ""


EntailmentVerdict = Literal["supports", "partially_supports", "contradicts"]


@dataclass(frozen=True)
class EntailmentResult:
    """Stage 3: verdict + a verbatim quotation (story 40 — string-matched against
    the source body before persisting)."""

    verdict: EntailmentVerdict
    quoted_passage: str
    reasoning: str = ""


@dataclass(frozen=True)
class TagAppropriatenessResult:
    """Story 35: confirm an ``unverifiable_by_design`` claim is honestly tagged.
    ``appropriate=False`` is the "mistagged empirical as hypothesis" catch."""

    appropriate: bool
    reasoning: str = ""


class Judge(Protocol):
    """The LLM-backed stage checks. Each method corresponds to exactly one stage."""

    async def locality(self, *, source: Source, body: str, claim_text: str) -> LocalityResult: ...

    async def entailment(
        self, *, source: Source, body: str, claim_text: str
    ) -> EntailmentResult: ...

    async def tag_appropriateness(
        self, *, claim_text: str, declared_type: ClaimType
    ) -> TagAppropriatenessResult: ...
