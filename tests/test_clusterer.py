"""ClaimClusterer: same-claim detection (stories 133-134, 157).

The clustering step is the load-bearing technical risk of synthesis: if same-claim
detection is wrong, every count is wrong. The module is therefore deep-tested
with fixture Findings, a mocked embedder, and a mocked binary adjudicator —
mirroring the ClaimVerifier fixture pattern (the irreducibly-semantic judgment
is boxed and auditable).
"""

from __future__ import annotations

from research_council.enums import ClaimType, Confidence
from research_council.ids import SequentialIdGenerator, new_finding_id
from research_council.models import Finding
from research_council.synthesis import (
    AdjudicationVerdict,
    ClusterResult,
    Embedder,
    SameClaimAdjudicator,
    cluster_findings,
)

# --- builders / fakes --------------------------------------------------------


def _finding(
    ids: SequentialIdGenerator,
    text: str,
    *,
    claim_type: ClaimType = ClaimType.GAP,
    confidence: Confidence = Confidence.SUPPORTING,
) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=claim_type,
        confidence=confidence,
        failure_modes_if_wrong="...",
        round=1,
    )


class FakeEmbedder:
    """Maps each text to a fixed vector via the provided table. Any unmapped text
    gets a unique one-hot, so unrelated findings stay far apart."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = dict(table)
        self._fallback_index = 1000

    def embed(self, text: str) -> list[float]:
        if text in self._table:
            return self._table[text]
        # Unique sparse vector — orthogonal to every other unmapped text.
        idx = self._fallback_index
        self._fallback_index += 1
        vec = [0.0] * (self._fallback_index + 1)
        vec[idx] = 1.0
        return vec


class FakeAdjudicator:
    """Returns scripted verdicts keyed by a normalized pair of claim_texts."""

    def __init__(
        self,
        verdicts: dict[frozenset[str], AdjudicationVerdict] | None = None,
        *,
        default: AdjudicationVerdict = AdjudicationVerdict.UNCERTAIN,
    ) -> None:
        self._verdicts = verdicts or {}
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def adjudicate(self, finding_a: Finding, finding_b: Finding) -> AdjudicationVerdict:
        self.calls.append((finding_a.claim_text, finding_b.claim_text))
        key = frozenset({finding_a.claim_text, finding_b.claim_text})
        return self._verdicts.get(key, self._default)


# --- pre-clustering sanity filter (story 157) -------------------------------


def test_empty_claim_text_is_excluded_with_note(ids: SequentialIdGenerator) -> None:
    f = _finding(ids, "")
    adjudicator = FakeAdjudicator()
    result = cluster_findings(
        [f],
        embedder=FakeEmbedder({}),
        adjudicator=adjudicator,
    )
    assert result.clusters == ()
    assert len(result.excluded) == 1
    assert result.excluded[0].finding_id == f.id
    assert result.excluded[0].reason == "empty_claim_text"
    assert adjudicator.calls == []  # never reached the adjudicator


def test_whitespace_only_claim_text_is_excluded(ids: SequentialIdGenerator) -> None:
    f = _finding(ids, "   \n\t  ")
    result = cluster_findings(
        [f],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
    )
    assert result.clusters == ()
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "whitespace_only_claim_text"


def test_below_minimum_length_claim_text_is_excluded(ids: SequentialIdGenerator) -> None:
    f = _finding(ids, "tiny")  # 4 chars, below MIN_CLAIM_TEXT_LENGTH
    result = cluster_findings(
        [f],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
    )
    assert result.clusters == ()
    assert result.excluded[0].reason == "below_minimum_length"


def test_degenerate_does_not_choke_clustering(ids: SequentialIdGenerator) -> None:
    """Schema-valid-but-degenerate Findings co-exist with valid ones without
    breaking the clustering step (story 157)."""
    valid_a = _finding(ids, "The filter agent has a circularity problem.")
    degenerate = _finding(ids, "")
    valid_b = _finding(ids, "The proposed filter is a cheap heuristic.")

    result = cluster_findings(
        [valid_a, degenerate, valid_b],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(default=AdjudicationVerdict.DIFFERENT),
    )
    assert len(result.excluded) == 1
    assert result.excluded[0].finding_id == degenerate.id
    cluster_ids = {fid for c in result.clusters for fid in c.finding_ids}
    assert cluster_ids == {valid_a.id, valid_b.id}


# --- embedding candidate grouping → binary adjudication (story 133) ---------


def test_below_threshold_pairs_never_reach_adjudicator(
    ids: SequentialIdGenerator,
) -> None:
    """Embedding similarity is the gate; pairs that are not near-duplicates by
    embedding are NOT sent to the LLM adjudicator. This is what makes the
    irreducibly-semantic judgment boxed rather than free-form."""
    a = _finding(ids, "The filter agent has a circularity problem.")
    b = _finding(ids, "Inference latency doubles under the proposed change.")
    # Orthogonal embeddings -> similarity 0, below threshold.
    embedder = FakeEmbedder({a.claim_text: [1.0, 0.0], b.claim_text: [0.0, 1.0]})
    adjudicator = FakeAdjudicator(default=AdjudicationVerdict.SAME)  # would merge if called

    result = cluster_findings([a, b], embedder=embedder, adjudicator=adjudicator)

    assert adjudicator.calls == []  # never adjudicated
    cluster_ids = {tuple(sorted(c.finding_ids)) for c in result.clusters}
    assert cluster_ids == {(a.id,), (b.id,)}


def test_above_threshold_pair_calls_adjudicator_exactly_once(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(ids, "The filter agent has a circularity problem.")
    b = _finding(ids, "The filter is itself a model and inherits the same failure.")
    # Same embedding -> similarity 1.0, above threshold.
    embedder = FakeEmbedder({a.claim_text: [1.0, 0.0], b.claim_text: [1.0, 0.0]})
    adjudicator = FakeAdjudicator(
        verdicts={frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME},
    )

    result = cluster_findings([a, b], embedder=embedder, adjudicator=adjudicator)

    assert len(adjudicator.calls) == 1
    assert len(result.clusters) == 1
    assert set(result.clusters[0].finding_ids) == {a.id, b.id}
    assert len(result.adjudications) == 1
    assert result.adjudications[0].verdict is AdjudicationVerdict.SAME


# --- split-on-uncertainty default (story 134) --------------------------------


def test_uncertain_pair_is_split(ids: SequentialIdGenerator) -> None:
    """A genuinely-ambiguous adjudicator verdict must default to split (treat
    the two Findings as distinct). The visible error (two adjacent clusters the
    researcher can merge by eye) is preferable to the invisible one (a false
    '7 lenses agree')."""
    a = _finding(ids, "The proposed filter is unlearnable without supervision.")
    b = _finding(ids, "The proposed filter has no learnable training signal.")
    embedder = FakeEmbedder({a.claim_text: [1.0, 0.0], b.claim_text: [1.0, 0.0]})
    adjudicator = FakeAdjudicator(default=AdjudicationVerdict.UNCERTAIN)

    result = cluster_findings([a, b], embedder=embedder, adjudicator=adjudicator)

    # Two singleton clusters, not one merged cluster.
    cluster_sets = [set(c.finding_ids) for c in result.clusters]
    assert {a.id} in cluster_sets
    assert {b.id} in cluster_sets
    # The uncertain adjudication was recorded for audit.
    assert any(
        rec.verdict is AdjudicationVerdict.UNCERTAIN for rec in result.adjudications
    )


def test_different_pair_is_split(ids: SequentialIdGenerator) -> None:
    a = _finding(ids, "The filter agent is a circular dependency.")
    b = _finding(ids, "The filter agent makes inference too slow.")
    embedder = FakeEmbedder({a.claim_text: [1.0, 0.0], b.claim_text: [1.0, 0.0]})
    adjudicator = FakeAdjudicator(default=AdjudicationVerdict.DIFFERENT)

    result = cluster_findings([a, b], embedder=embedder, adjudicator=adjudicator)

    cluster_sets = [set(c.finding_ids) for c in result.clusters]
    assert {a.id} in cluster_sets
    assert {b.id} in cluster_sets


# --- resists over-merging same-sounding-but-distinct findings ---------------


def test_resists_over_merging_two_distinct_same_sounding_gaps(
    ids: SequentialIdGenerator,
) -> None:
    """High embedding similarity is necessary for adjudication but NOT sufficient
    for merging. Two gaps that *sound* the same to an embedder but are
    semantically distinct (e.g., 'no training signal for filter' vs. 'no eval
    signal for filter') must be split when the adjudicator rules DIFFERENT."""
    a = _finding(ids, "There is no training signal for learning the filter agent.")
    b = _finding(ids, "There is no eval signal for measuring the filter agent.")
    # Near-duplicate embeddings (same sentence shape) -> well above threshold.
    embedder = FakeEmbedder({a.claim_text: [1.0, 0.0], b.claim_text: [0.98, 0.2]})
    adjudicator = FakeAdjudicator(
        verdicts={frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.DIFFERENT},
    )

    result = cluster_findings([a, b], embedder=embedder, adjudicator=adjudicator)

    cluster_sets = [set(c.finding_ids) for c in result.clusters]
    assert {a.id} in cluster_sets
    assert {b.id} in cluster_sets
    # The adjudicator WAS consulted — the embedder alone never decides.
    assert len(adjudicator.calls) == 1


# --- transitive merging across multiple SAME verdicts -----------------------


def test_transitive_same_verdicts_form_one_cluster(ids: SequentialIdGenerator) -> None:
    """When (A,B) and (B,C) both adjudicate SAME, the three findings collapse into
    one cluster — agreement counts must reflect the full transitive closure."""
    a = _finding(ids, "The filter agent is a circular dependency.")
    b = _finding(ids, "The filter is itself a model and inherits the same problem.")
    c = _finding(ids, "Filtering with a model that needs filtering is circular.")

    # All three embeddings identical -> every pair becomes a candidate.
    embedder = FakeEmbedder(
        {a.claim_text: [1.0, 0.0], b.claim_text: [1.0, 0.0], c.claim_text: [1.0, 0.0]}
    )
    adjudicator = FakeAdjudicator(
        verdicts={
            frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME,
            frozenset({b.claim_text, c.claim_text}): AdjudicationVerdict.SAME,
            frozenset({a.claim_text, c.claim_text}): AdjudicationVerdict.SAME,
        }
    )

    result = cluster_findings([a, b, c], embedder=embedder, adjudicator=adjudicator)
    assert len(result.clusters) == 1
    assert set(result.clusters[0].finding_ids) == {a.id, b.id, c.id}


# --- determinism (mirrors ClaimVerifier fixture pattern) --------------------


def test_deterministic_given_mocked_inputs(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(ids, "The filter agent has a circularity problem.")
    b = _finding(ids, "The filter is unsuitable as a learning target.")
    c = _finding(ids, "Inference latency doubles under the proposed change.")

    def make_inputs() -> tuple[Embedder, SameClaimAdjudicator]:
        embedder = FakeEmbedder(
            {
                a.claim_text: [1.0, 0.0],
                b.claim_text: [1.0, 0.0],
                c.claim_text: [0.0, 1.0],
            }
        )
        adjudicator = FakeAdjudicator(
            verdicts={frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
        )
        return embedder, adjudicator

    def run() -> object:
        embedder, adjudicator = make_inputs()
        return _normalize(
            cluster_findings([a, b, c], embedder=embedder, adjudicator=adjudicator)
        )

    assert run() == run()


def _normalize(result: ClusterResult) -> object:
    """Cluster ordering is implementation-detail; equality compares by content."""
    cluster_sets = sorted(tuple(sorted(c.finding_ids)) for c in result.clusters)
    excluded = sorted((e.finding_id, e.reason) for e in result.excluded)
    adjudications = sorted(
        (rec.finding_a, rec.finding_b, rec.verdict.value) for rec in result.adjudications
    )
    return (cluster_sets, excluded, adjudications)


# --- empty input edge case --------------------------------------------------


def test_no_findings_yields_empty_result() -> None:
    result = cluster_findings(
        [], embedder=FakeEmbedder({}), adjudicator=FakeAdjudicator()
    )
    assert result.clusters == ()
    assert result.excluded == ()
    assert result.adjudications == ()


def test_single_valid_finding_is_a_singleton_cluster(ids: SequentialIdGenerator) -> None:
    f = _finding(ids, "The filter agent has a circularity problem.")
    result = cluster_findings(
        [f],
        embedder=FakeEmbedder({f.claim_text: [1.0, 0.0]}),
        adjudicator=FakeAdjudicator(),
    )
    assert len(result.clusters) == 1
    assert result.clusters[0].finding_ids == (f.id,)
    assert result.adjudications == ()  # no pairs -> no adjudication calls
