"""Source canonical identity and dedup (story 13)."""

from __future__ import annotations

from research_council.domain.source_ops import canonical_id_for_source, canonical_source_id
from research_council.models import Source


def test_doi_takes_precedence_over_everything() -> None:
    cid = canonical_source_id(
        doi="10.1234/abc",
        arxiv_id="2307.03172",
        openalex_id="W123",
        title="Some Title",
        first_author="Liu",
        year=2023,
    )
    assert cid == "doi:10.1234/abc"


def test_arxiv_precedence_over_openalex_and_title() -> None:
    cid = canonical_source_id(arxiv_id="arXiv:2307.03172v2", openalex_id="W1", title="T")
    assert cid == "arxiv:2307.03172"  # prefix and version stripped, lowercased


def test_doi_url_forms_normalize_to_same_id() -> None:
    a = canonical_source_id(doi="https://doi.org/10.1/X")
    b = canonical_source_id(doi="10.1/x")
    assert a == b


def test_title_hash_fallback_is_deterministic_and_normalized() -> None:
    a = canonical_source_id(title="Lost in the Middle!", first_author="Liu", year=2023)
    b = canonical_source_id(title="lost in   the middle", first_author="liu", year=2023)
    assert a == b
    assert a.startswith("titlehash:")


def test_distinct_titles_do_not_collide() -> None:
    a = canonical_source_id(title="Attention Is All You Need", first_author="Vaswani", year=2017)
    b = canonical_source_id(title="Lost in the Middle", first_author="Liu", year=2023)
    assert a != b


def test_canonical_id_for_source_uses_first_author() -> None:
    src = Source(
        canonical_id=canonical_source_id(title="T", first_author="Liu", year=2023),
        title="T",
        authors=("Liu", "Lin"),
        year=2023,
    )
    assert canonical_id_for_source(src) == canonical_source_id(
        title="T", first_author="Liu", year=2023
    )
