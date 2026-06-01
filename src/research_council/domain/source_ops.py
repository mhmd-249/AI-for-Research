"""Source canonical identity (story 13).

Identity precedence: DOI > arXiv id > OpenAlex id > hash(normalized
title + first author + year). The first present identifier wins, so the same
paper resolves to one canonical id regardless of which fields a given citation
happened to include. The title-hash fallback is deterministic (not fuzzy): two
citations that normalize to the same title+author+year dedup; near-misses do not.
"""

from __future__ import annotations

import hashlib
import re

from ..ids import SourceId
from ..models import Source

_DOI_PREFIX = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^(https?://)?(arxiv\.org/abs/)?(arxiv:)?", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_OPENALEX_PREFIX = re.compile(r"^(https?://)?openalex\.org/", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def _normalize_doi(doi: str) -> str:
    return _DOI_PREFIX.sub("", doi.strip()).lower()


def _normalize_arxiv(arxiv_id: str) -> str:
    stripped = _ARXIV_PREFIX.sub("", arxiv_id.strip())
    return _ARXIV_VERSION.sub("", stripped).lower()


def _normalize_openalex(openalex_id: str) -> str:
    return _OPENALEX_PREFIX.sub("", openalex_id.strip()).lower()


def _normalize_title(title: str) -> str:
    lowered = _NON_ALNUM.sub(" ", title.strip().lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def canonical_source_id(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    title: str | None = None,
    first_author: str | None = None,
    year: int | None = None,
) -> SourceId:
    """Compute a source's canonical id from its identifying fields."""
    if doi and doi.strip():
        return SourceId(f"doi:{_normalize_doi(doi)}")
    if arxiv_id and arxiv_id.strip():
        return SourceId(f"arxiv:{_normalize_arxiv(arxiv_id)}")
    if openalex_id and openalex_id.strip():
        return SourceId(f"openalex:{_normalize_openalex(openalex_id)}")

    basis = f"{_normalize_title(title or '')}|{(first_author or '').strip().lower()}|{year or ''}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return SourceId(f"titlehash:{digest}")


def canonical_id_for_source(source: Source) -> SourceId:
    """Compute the canonical id for a :class:`Source` from its fields."""
    return canonical_source_id(
        doi=source.doi,
        arxiv_id=source.arxiv_id,
        openalex_id=source.openalex_id,
        title=source.title,
        first_author=source.authors[0] if source.authors else None,
        year=source.year,
    )
