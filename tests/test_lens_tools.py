"""The two tier-1 lens tools — ``source_fetch`` and ``verifier_query`` — that the
prior-art and adversarial lenses depend on (PRD stories 47, 55, 56, 159).

* ``source_fetch`` returns the retrieved body wrapped in the labeled
  untrusted-retrieved-content quarantine block (story 159).
* ``verifier_query`` runs a claim through the 3-stage ClaimVerifier and returns
  a structured verifier verdict, including the verbatim quoted passage when the
  judge supplied one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from research_council.domain.source_ops import canonical_source_id
from research_council.enums import ClaimType, ToolName, VerificationStatus
from research_council.ids import SequentialIdGenerator
from research_council.models import Source
from research_council.runtime.lens_tools import SourceFetchTool, VerifierQueryTool
from research_council.verifier import (
    ClaimVerifier,
    EntailmentResult,
    InMemoryVerifierCache,
    LocalityResult,
    SourceCandidate,
    SourceHint,
    SourceRegistry,
    TagAppropriatenessResult,
)

# --- fake source backends ---------------------------------------------------


@dataclass
class _StubClient:
    name: str
    response: SourceCandidate | None = None

    async def query(self, hint: SourceHint) -> SourceCandidate | None:
        return self.response


def _empty_registry() -> SourceRegistry:
    return SourceRegistry(
        s2=_StubClient("s2"),
        arxiv=_StubClient("arxiv"),
        openalex=_StubClient("openalex"),
        crossref=_StubClient("crossref"),
    )


def _registry_with_hit(candidate: SourceCandidate) -> SourceRegistry:
    return SourceRegistry(
        s2=_StubClient("s2", response=candidate),
        arxiv=_StubClient("arxiv"),
        openalex=_StubClient("openalex"),
        crossref=_StubClient("crossref"),
    )


class _StubJudge:
    def __init__(
        self,
        *,
        locality: LocalityResult,
        entailment: EntailmentResult,
        tag: TagAppropriatenessResult | None = None,
    ) -> None:
        self._locality = locality
        self._entailment = entailment
        self._tag = tag or TagAppropriatenessResult(appropriate=True, reasoning="ok")

    async def locality(
        self, *, source: Source, body: str, claim_text: str
    ) -> LocalityResult:
        return self._locality

    async def entailment(
        self, *, source: Source, body: str, claim_text: str
    ) -> EntailmentResult:
        return self._entailment

    async def tag_appropriateness(
        self, *, claim_text: str, declared_type: ClaimType
    ) -> TagAppropriatenessResult:
        return self._tag


# --- SourceFetchTool --------------------------------------------------------


async def test_source_fetch_wraps_body_in_quarantine_block() -> None:
    fetched_calls: list[str] = []

    async def fetcher(canonical_id: str) -> str | None:
        fetched_calls.append(canonical_id)
        return "RAW PAPER ABSTRACT TEXT"

    tool = SourceFetchTool(fetcher=fetcher)

    assert tool.name is ToolName.SOURCE_FETCH

    result = await tool.handle({"canonical_id": "arxiv:2307.03172"})

    # The whole retrieved body lives inside the labeled quarantine wrapper.
    assert result.startswith("<untrusted_retrieved_content")
    assert result.endswith("</untrusted_retrieved_content>")
    assert "RAW PAPER ABSTRACT TEXT" in result
    # The canonical id labels the block, so the lens knows which source it sees.
    assert "arxiv:2307.03172" in result
    assert fetched_calls == ["arxiv:2307.03172"]


async def test_source_fetch_missing_id_is_handled_gracefully() -> None:
    async def fetcher(canonical_id: str) -> str | None:
        return None

    tool = SourceFetchTool(fetcher=fetcher)
    result = await tool.handle({"canonical_id": "arxiv:9999.99999"})

    # Returns a plain not-found message — NOT a quarantine block (no retrieved
    # text exists to quarantine).
    assert "<untrusted_retrieved_content" not in result
    assert "not found" in result.lower()


async def test_source_fetch_advertised_schema_requires_canonical_id() -> None:
    async def fetcher(_: str) -> str | None:
        return ""

    tool = SourceFetchTool(fetcher=fetcher)
    schema = tool.input_schema
    assert schema["required"] == ["canonical_id"]


# --- VerifierQueryTool ------------------------------------------------------


def _verifier(
    judge: _StubJudge, sources: SourceRegistry, ids: SequentialIdGenerator
) -> ClaimVerifier:
    return ClaimVerifier(
        sources=sources,
        judge=judge,
        cache=InMemoryVerifierCache(),
        verifier_version="test-v1",
        id_generator=ids,
    )


async def test_verifier_query_returns_structured_supports_verdict(
    ids: SequentialIdGenerator,
) -> None:
    body = "Lost-in-the-middle: positional bias persists under SFT."
    source = Source(
        canonical_id=canonical_source_id(arxiv_id="2307.03172"),
        title="Lost in the Middle",
        arxiv_id="2307.03172",
        has_full_text=True,
    )
    candidate = SourceCandidate(source=source, body=body)
    judge = _StubJudge(
        locality=LocalityResult(has_relevant_content=True),
        entailment=EntailmentResult(
            verdict="supports",
            quoted_passage="positional bias persists under SFT",
            reasoning="the paper directly reports the U-shape",
        ),
    )
    verifier = _verifier(judge, _registry_with_hit(candidate), ids)

    tool = VerifierQueryTool(verifier=verifier, id_generator=ids)
    assert tool.name is ToolName.VERIFIER_QUERY

    result = await tool.handle(
        {
            "claim_text": "Models exhibit a U-shape on long-context retrieval.",
            "claim_type": "empirical",
            "arxiv_id": "2307.03172",
        }
    )

    # Structured fields the lens can act on: status + canonical id + the verbatim
    # quoted passage the judge supplied.
    assert f"status: {VerificationStatus.VERIFIED.value}" in result
    assert "positional bias persists under SFT" in result
    # Story 159: the verbatim quoted passage is source-derived text and must
    # enter lens context only inside the labeled quarantine block.
    quote = "positional bias persists under SFT"
    quote_index = result.index(quote)
    open_tag_index = result.rfind("<untrusted_retrieved_content", 0, quote_index)
    close_tag_index = result.find("</untrusted_retrieved_content>", quote_index)
    assert open_tag_index != -1 and close_tag_index != -1, (
        "quoted_passage must be wrapped in the quarantine block"
    )


async def test_verifier_query_surfaces_source_not_found_when_no_hits(
    ids: SequentialIdGenerator,
) -> None:
    judge = _StubJudge(
        locality=LocalityResult(has_relevant_content=False),
        entailment=EntailmentResult(verdict="supports", quoted_passage=""),
    )
    verifier = _verifier(judge, _empty_registry(), ids)

    tool = VerifierQueryTool(verifier=verifier, id_generator=ids)
    result = await tool.handle(
        {
            "claim_text": "A 2024 paper reports X on benchmark Y.",
            "claim_type": "prior_art",
            "title": "Nonexistent paper",
            "first_author": "Nobody",
            "year": 2024,
        }
    )

    assert f"status: {VerificationStatus.SOURCE_NOT_FOUND.value}" in result


async def test_verifier_query_rejects_unknown_claim_type(
    ids: SequentialIdGenerator,
) -> None:
    judge = _StubJudge(
        locality=LocalityResult(has_relevant_content=False),
        entailment=EntailmentResult(verdict="supports", quoted_passage=""),
    )
    verifier = _verifier(judge, _empty_registry(), ids)
    tool = VerifierQueryTool(verifier=verifier, id_generator=ids)

    with pytest.raises(ValueError):
        await tool.handle({"claim_text": "X", "claim_type": "not_a_real_type"})


# --- Composition: source_fetch + a real SourceCandidate body -----------------


async def test_source_fetch_can_adapt_to_source_registry(
    ids: SequentialIdGenerator,
) -> None:
    """Show source_fetch composes with the existing SourceRegistry surface.

    The lens-facing tool takes a callable, so production can adapt a registry
    by mapping canonical id -> body. This test demonstrates that path.
    """
    body = "PAPER BODY"
    source = Source(
        canonical_id=canonical_source_id(arxiv_id="2307.03172"),
        title="Demo",
        arxiv_id="2307.03172",
    )
    cache: dict[str, str] = {str(source.canonical_id): body}

    async def fetcher(canonical_id: str) -> str | None:
        return cache.get(canonical_id)

    tool = SourceFetchTool(fetcher=fetcher)
    result = await tool.handle({"canonical_id": source.canonical_id})
    assert "<untrusted_retrieved_content" in result
    assert body in result
