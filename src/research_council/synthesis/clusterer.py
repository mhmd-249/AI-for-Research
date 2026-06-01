"""ClaimClusterer — same-claim detection (PRD stories 133, 134, 157).

This is an isolated deep module because every synthesis count depends on it: if
same-claim detection is wrong, every "N lenses agreed" number downstream is
wrong. The shape is therefore three boxed stages that are individually
fixture-testable:

1. **Pre-clustering sanity filter (story 157).** Schema-valid-but-degenerate
   Findings (empty / whitespace-only / below minimum-length ``claim_text``) are
   routed to the same "excluded with note" path schema-invalid Findings use, so
   the clustering step never has to handle them.

2. **Embedding-similarity candidate grouping (story 133).** Cosine similarity
   over a pluggable :class:`Embedder` proposes candidate pairs above a fixed
   threshold. The embedder alone is not allowed to decide cluster membership;
   it only narrows the search space before the binary adjudication pass.

3. **Boxed binary same-claim adjudication, split-on-uncertainty (stories 133,
   134).** A pluggable :class:`SameClaimAdjudicator` answers exactly
   ``SAME | DIFFERENT | UNCERTAIN`` per candidate pair — never free-form
   clustering, so the one irreducibly-semantic judgment is auditable and
   fixture-testable. ``UNCERTAIN`` defaults to split, so the error made is the
   visible one (two adjacent clusters the researcher can merge by eye) rather
   than the invisible one (a false "7 lenses agree" that silently misleads).

Both seams are protocols, mirroring the runtime's ``LlmClient`` seam: tests
inject deterministic fakes; production wires real embeddings and an LLM
adjudicator behind the same shape. Given mocked embedder + adjudicator inputs,
the output of :func:`cluster_findings` is fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..ids import FindingId
from ..models import Finding

# Minimum non-whitespace claim_text length accepted into clustering. Anything
# shorter is routed to the excluded-with-note path (story 157). 10 chars is the
# smallest number that catches the realistic degenerate cases ("ok", "n/a",
# "TBD", "foo") without filtering legitimate (if terse) claims.
MIN_CLAIM_TEXT_LENGTH = 10

# Cosine-similarity cutoff at which two findings become a candidate pair for
# the binary adjudicator. Tuned conservatively: this is a *recall* dial — pairs
# above the threshold get adjudicated, pairs below are split by default.
DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD = 0.7


# --- exclusion reason codes (used in ExcludedFinding.reason) ----------------

EXCLUSION_EMPTY = "empty_claim_text"
EXCLUSION_WHITESPACE = "whitespace_only_claim_text"
EXCLUSION_TOO_SHORT = "below_minimum_length"


# --- public protocols & data shapes -----------------------------------------


class AdjudicationVerdict(StrEnum):
    """The boxed adjudicator's three legal answers. Anything other than
    :attr:`SAME` is treated as a split (story 134's split-on-uncertainty)."""

    SAME = "same"
    DIFFERENT = "different"
    UNCERTAIN = "uncertain"


class Embedder(Protocol):
    """Turns a claim_text into a dense vector. Production wires a real embedding
    model behind this; tests inject a deterministic mapping."""

    def embed(self, text: str) -> list[float]: ...


class SameClaimAdjudicator(Protocol):
    """The boxed binary judge: given two Findings, answer SAME / DIFFERENT /
    UNCERTAIN. Free-form clustering is deliberately out of scope (story 133)."""

    def adjudicate(
        self, finding_a: Finding, finding_b: Finding
    ) -> AdjudicationVerdict: ...


@dataclass(frozen=True)
class ExcludedFinding:
    """A Finding the pre-clustering filter rejected (story 157)."""

    finding_id: FindingId
    reason: str


@dataclass(frozen=True)
class AdjudicationRecord:
    """One pair's audit record: the inputs the embedder proposed and the
    adjudicator's verdict. Persisted so a researcher can later answer 'why was
    this pair split / merged.'"""

    finding_a: FindingId
    finding_b: FindingId
    similarity: float
    verdict: AdjudicationVerdict


@dataclass(frozen=True)
class Cluster:
    """A same-claim cluster. Singleton clusters (one finding_id) are legal."""

    finding_ids: tuple[FindingId, ...]


@dataclass(frozen=True)
class ClusterResult:
    """Output of :func:`cluster_findings`. ``clusters`` covers every valid input
    finding exactly once; ``excluded`` covers every degenerate one."""

    clusters: tuple[Cluster, ...]
    excluded: tuple[ExcludedFinding, ...]
    adjudications: tuple[AdjudicationRecord, ...]


# --- pre-clustering sanity filter -------------------------------------------


def _exclusion_reason(claim_text: str) -> str | None:
    if claim_text == "":
        return EXCLUSION_EMPTY
    if not claim_text.strip():
        return EXCLUSION_WHITESPACE
    if len(claim_text.strip()) < MIN_CLAIM_TEXT_LENGTH:
        return EXCLUSION_TOO_SHORT
    return None


# --- cosine similarity ------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine over two equal- or unequal-length vectors. Shorter vectors are
    treated as zero-padded; mismatched dimensions never raise (test fakes use
    unique-index one-hots for unrelated texts, deliberately differing in
    length). A zero-norm vector yields 0.0."""
    width = max(len(a), len(b))
    dot = 0.0
    for i in range(width):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        dot += ai * bi
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- union-find -------------------------------------------------------------


class _UnionFind:
    """Path-compressed union-find keyed by FindingId. Plain code rather than a
    dep — the clusterer is otherwise a stdlib-only module."""

    def __init__(self, members: list[FindingId]) -> None:
        self._parent: dict[FindingId, FindingId] = {m: m for m in members}

    def find(self, x: FindingId) -> FindingId:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: FindingId, y: FindingId) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry


# --- main entrypoint --------------------------------------------------------


def cluster_findings(
    findings: list[Finding],
    *,
    embedder: Embedder,
    adjudicator: SameClaimAdjudicator,
    similarity_threshold: float = DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD,
) -> ClusterResult:
    """Cluster ``findings`` by same-claim equivalence.

    Pipeline:
      1. Filter degenerate ``claim_text`` (story 157) → excluded-with-note.
      2. Embed the remaining findings; cosine pairs above ``similarity_threshold``
         become candidates.
      3. Adjudicate each candidate pair as SAME / DIFFERENT / UNCERTAIN.
      4. Union only SAME verdicts (story 134's split-on-uncertainty default).
      5. Group into clusters in input order; every valid finding lands in
         exactly one cluster (singletons are legal).
    """
    valid: list[Finding] = []
    excluded: list[ExcludedFinding] = []
    for finding in findings:
        reason = _exclusion_reason(finding.claim_text)
        if reason is None:
            valid.append(finding)
        else:
            excluded.append(ExcludedFinding(finding_id=finding.id, reason=reason))

    embeddings: dict[FindingId, list[float]] = {
        f.id: embedder.embed(f.claim_text) for f in valid
    }

    adjudications: list[AdjudicationRecord] = []
    uf = _UnionFind([f.id for f in valid])

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            fa, fb = valid[i], valid[j]
            similarity = _cosine_similarity(embeddings[fa.id], embeddings[fb.id])
            if similarity < similarity_threshold:
                continue  # below-threshold pairs are never sent to the adjudicator
            verdict = adjudicator.adjudicate(fa, fb)
            adjudications.append(
                AdjudicationRecord(
                    finding_a=fa.id,
                    finding_b=fb.id,
                    similarity=similarity,
                    verdict=verdict,
                )
            )
            if verdict is AdjudicationVerdict.SAME:
                uf.union(fa.id, fb.id)
            # SPLIT-ON-UNCERTAINTY: DIFFERENT and UNCERTAIN both leave the pair
            # in separate components. Story 134: the visible error beats the
            # invisible one.

    # Group findings by their union-find root, preserving input order both
    # across clusters and within each cluster's member list.
    groups: dict[FindingId, list[FindingId]] = {}
    for finding in valid:
        root = uf.find(finding.id)
        groups.setdefault(root, []).append(finding.id)

    clusters = tuple(Cluster(finding_ids=tuple(members)) for members in groups.values())

    return ClusterResult(
        clusters=clusters,
        excluded=tuple(excluded),
        adjudications=tuple(adjudications),
    )
