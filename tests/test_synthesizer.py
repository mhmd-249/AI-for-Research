"""Synthesizer — hybrid Round 3 controller (stories 130-144, 156).

Deterministic structure is tested independently of narration: given a fixed set
of typed Findings, the structural layer produces correct counts, contradiction
pairs, and two-tier gap collections; a separate test confirms the narration pass
cannot alter membership/counts (re-validation rejects an LLM output that does).
Mirrors the ClaimClusterer fixture pattern — mocked embedder + adjudicator +
narrator make the whole pipeline deterministic.
"""

from __future__ import annotations

import pytest

from research_council.enums import (
    ChangeReason,
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    TensionSource,
    VerificationStatus,
)
from research_council.ids import (
    FindingId,
    SequentialIdGenerator,
    new_finding_id,
    new_lens_run_id,
    new_session_id,
)
from research_council.models import EmergentGap, Finding, LensRun
from research_council.synthesis import (
    AdjudicationVerdict,
    ComputedStructure,
    NarrationResult,
    NarrationValidationError,
    compute_structure,
    resolve_winning_findings,
    synthesize,
    validate_narration,
)
from research_council.synthesis.synthesizer import _run_narration

# --- builders / fakes (mirror test_clusterer) -------------------------------


def _finding(
    ids: SequentialIdGenerator,
    text: str,
    *,
    claim_type: ClaimType = ClaimType.GAP,
    confidence: Confidence = Confidence.SUPPORTING,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    change_reason: ChangeReason | None = None,
    supersedes: FindingId | None = None,
) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=claim_type,
        confidence=confidence,
        failure_modes_if_wrong="...",
        verification_status=verification_status,
        round=1,
        change_reason=change_reason,
        supersedes=supersedes,
    )


def _run(
    ids: SequentialIdGenerator,
    *,
    lens_id: LensId,
    status: LensRunStatus,
    finding_ids: tuple[FindingId, ...] = (),
) -> LensRun:
    return LensRun(
        id=new_lens_run_id(ids),
        lens_id=lens_id,
        session_id=new_session_id(SequentialIdGenerator()),
        brief_version=1,
        round=1,
        status=status,
        finding_ids=finding_ids,
    )


class FakeEmbedder:
    """Maps text to a vector; unmapped text gets a unique one-hot (orthogonal)."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = dict(table)
        self._fallback_index = 1000

    def embed(self, text: str) -> list[float]:
        if text in self._table:
            return self._table[text]
        idx = self._fallback_index
        self._fallback_index += 1
        vec = [0.0] * (idx + 1)
        vec[idx] = 1.0
        return vec


class FakeAdjudicator:
    def __init__(
        self,
        verdicts: dict[frozenset[str], AdjudicationVerdict] | None = None,
        *,
        default: AdjudicationVerdict = AdjudicationVerdict.DIFFERENT,
    ) -> None:
        self._verdicts = verdicts or {}
        self._default = default

    def adjudicate(self, finding_a: Finding, finding_b: Finding) -> AdjudicationVerdict:
        key = frozenset({finding_a.claim_text, finding_b.claim_text})
        return self._verdicts.get(key, self._default)


class EchoNarrator:
    """An honest narrator: echoes the structure it was handed and adds prose."""

    def __init__(
        self,
        *,
        emergent_gaps: tuple[EmergentGap, ...] = (),
        recommendations: tuple[str, ...] = (),
    ) -> None:
        self._emergent_gaps = emergent_gaps
        self._recommendations = recommendations
        self.calls = 0

    def narrate(self, structure: ComputedStructure) -> NarrationResult:
        self.calls += 1
        return NarrationResult(
            agreement_cluster_refs=tuple(c.finding_refs for c in structure.agreement_clusters),
            declared_gap_refs=tuple(g.finding_refs for g in structure.declared_gaps),
            emergent_gaps=self._emergent_gaps,
            conditional_recommendations=self._recommendations,
        )


class TamperingNarrator:
    """A dishonest narrator: merges all agreement clusters into one (alters
    membership/counts), the exact thing re-validation must reject."""

    def __init__(self) -> None:
        self.calls = 0

    def narrate(self, structure: ComputedStructure) -> NarrationResult:
        self.calls += 1
        merged = tuple(fid for c in structure.agreement_clusters for fid in c.finding_refs)
        return NarrationResult(
            agreement_cluster_refs=(merged,) if merged else (),
            declared_gap_refs=tuple(g.finding_refs for g in structure.declared_gaps),
        )


def _same_embeddings(*texts: str) -> dict[str, list[float]]:
    return {t: [1.0, 0.0] for t in texts}


# --- deterministic counts (story 131/135) -----------------------------------


def test_agreement_cluster_counts_and_confidence_breakdown(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids,
        "Latency doubles under the proposed change.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.LOAD_BEARING,
    )
    b = _finding(
        ids,
        "Latency roughly doubles when the filter runs.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.SUPPORTING,
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
    )

    structure = compute_structure(
        findings=[a, b], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    assert len(structure.agreement_clusters) == 1
    cluster = structure.agreement_clusters[0]
    assert cluster.count == 2
    assert set(cluster.finding_refs) == {a.id, b.id}
    assert cluster.confidence_breakdown.load_bearing == 1
    assert cluster.confidence_breakdown.supporting == 1
    assert cluster.confidence_breakdown.exploratory == 0
    assert cluster.strongest_confidence is Confidence.LOAD_BEARING


def test_split_is_preserved_as_two_clusters_never_averaged(
    ids: SequentialIdGenerator,
) -> None:
    """6-vs-4-style split: two clusters with two counts, never 'mostly agree'."""
    a = _finding(
        ids, "Mechanism A explains the failure.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    b = _finding(
        ids, "Mechanism A is the cause of the failure.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    c = _finding(
        ids, "Mechanism B explains the failure instead.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    # All three look similar to the embedder; adjudicator merges a+b, splits c.
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text, c.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
    )

    structure = compute_structure(
        findings=[a, b, c], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    counts = sorted(c.count for c in structure.agreement_clusters)
    assert counts == [1, 2]


# --- contradiction tension (story 137) --------------------------------------


def test_contradicted_cluster_becomes_tension_not_agreement(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids,
        "The benchmark shows a 2x speedup.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.LOAD_BEARING,
        verification_status=VerificationStatus.CONTRADICTED,
    )
    b = _finding(
        ids,
        "A 2x speedup is observed on the benchmark.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.LOAD_BEARING,
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
    )

    structure = compute_structure(
        findings=[a, b], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    assert structure.agreement_clusters == ()
    assert len(structure.tensions) == 1
    tension = structure.tensions[0]
    assert tension.source is TensionSource.CONTRADICTION
    assert set(tension.finding_refs) == {a.id, b.id}


def test_opposing_mechanism_split_becomes_opposing_pair_tension(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids, "The failure is caused by attention dilution.", claim_type=ClaimType.FAILURE_MODE
    )
    b = _finding(
        ids,
        "The failure is caused by attention saturation instead.",
        claim_type=ClaimType.FAILURE_MODE,
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    # High-similarity pair the adjudicator splits → opposing pair.
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.DIFFERENT}
    )

    structure = compute_structure(
        findings=[a, b], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    opposing = [t for t in structure.tensions if t.source is TensionSource.OPPOSING_PAIR]
    assert len(opposing) == 1
    assert set(opposing[0].finding_refs) == {a.id, b.id}


# --- two-tier gaps (story 136/141) ------------------------------------------


def test_declared_gaps_clustered_from_gap_findings(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids, "No training signal exists for the filter agent.", confidence=Confidence.LOAD_BEARING
    )
    b = _finding(
        ids, "The filter agent lacks any training signal.", confidence=Confidence.SUPPORTING
    )
    other = _finding(
        ids, "Inference latency is acceptable in practice.", claim_type=ClaimType.EMPIRICAL
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
    )

    structure = compute_structure(
        findings=[a, b, other], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    assert len(structure.declared_gaps) == 1
    gap = structure.declared_gaps[0]
    assert set(gap.finding_refs) == {a.id, b.id}
    assert gap.count == 2
    assert gap.has_cross_lens_support
    assert not gap.is_weak
    # the empirical (non-gap) finding never landed in declared_gaps
    assert all(other.id not in g.finding_refs for g in structure.declared_gaps)


def test_weak_exploratory_singleton_gap_is_weak(ids: SequentialIdGenerator) -> None:
    g = _finding(
        ids, "Possibly the cache eviction policy matters here.", confidence=Confidence.EXPLORATORY
    )
    structure = compute_structure(
        findings=[g], runs=[], embedder=FakeEmbedder({}), adjudicator=FakeAdjudicator()
    )
    assert len(structure.declared_gaps) == 1
    assert structure.declared_gaps[0].is_weak


# --- zero-findings anomaly (story 156) --------------------------------------


def test_zero_finding_succeeded_lens_is_soft_anomaly(
    ids: SequentialIdGenerator,
) -> None:
    empty_run = _run(ids, lens_id=LensId.ARCHITECTURE, status=LensRunStatus.SUCCEEDED)
    structure = compute_structure(
        findings=[],
        runs=[empty_run],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
    )
    assert len(structure.anomalies) == 1
    assert structure.anomalies[0].lens_id is LensId.ARCHITECTURE
    assert "no findings" in structure.anomalies[0].message


def test_refused_and_failed_lenses_are_not_anomalies(
    ids: SequentialIdGenerator,
) -> None:
    refused = _run(ids, lens_id=LensId.ADVERSARIAL, status=LensRunStatus.REFUSED)
    invalid = _run(ids, lens_id=LensId.PRIOR_ART, status=LensRunStatus.SCHEMA_INVALID)
    structure = compute_structure(
        findings=[],
        runs=[refused, invalid],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
    )
    assert structure.anomalies == ()


# --- supersession propagation (story 138) -----------------------------------


def test_superseded_finding_is_dropped_synthesis_references_winner(
    ids: SequentialIdGenerator,
) -> None:
    old = _finding(ids, "The filter agent has a circularity problem.")
    new = _finding(
        ids,
        "On reflection, the filter circularity is partly resolvable.",
        supersedes=old.id,
        change_reason=ChangeReason.REVISED_RECONSIDERED,
    )
    winners = resolve_winning_findings([old, new])
    assert winners == [new]
    assert old.id not in {f.id for f in winners}


# --- narration re-validation (story 130/132) --------------------------------


def test_validate_narration_accepts_honest_echo(ids: SequentialIdGenerator) -> None:
    a = _finding(ids, "No training signal exists for the filter agent.")
    structure = compute_structure(
        findings=[a], runs=[], embedder=FakeEmbedder({}), adjudicator=FakeAdjudicator()
    )
    honest = NarrationResult(
        agreement_cluster_refs=(),
        declared_gap_refs=tuple(g.finding_refs for g in structure.declared_gaps),
    )
    validate_narration(structure, honest)  # does not raise


def test_validate_narration_rejects_altered_membership(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids, "Mechanism A explains the failure.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    b = _finding(
        ids, "Mechanism B explains the failure instead.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    # Two distinct agreement clusters (split).
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(default=AdjudicationVerdict.DIFFERENT)
    structure = compute_structure(
        findings=[a, b], runs=[], embedder=embedder, adjudicator=adjudicator
    )
    assert len(structure.agreement_clusters) == 2

    tampered = NarrationResult(
        agreement_cluster_refs=((a.id, b.id),),  # merged the two clusters
        declared_gap_refs=(),
    )
    with pytest.raises(NarrationValidationError):
        validate_narration(structure, tampered)


def test_two_strike_narration_falls_back_to_structure_only(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids, "Mechanism A explains the failure.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    b = _finding(
        ids, "Mechanism B explains the failure instead.", claim_type=ClaimType.MECHANISM_HYPOTHESIS
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(default=AdjudicationVerdict.DIFFERENT)
    structure = compute_structure(
        findings=[a, b], runs=[], embedder=embedder, adjudicator=adjudicator
    )

    narrator = TamperingNarrator()
    narration, fell_back = _run_narration(narrator, structure)

    assert narrator.calls == 2  # two strikes
    assert fell_back is True
    assert narration.emergent_gaps == ()
    assert narration.conditional_recommendations == ()


# --- end-to-end synthesize() ------------------------------------------------


def test_synthesize_end_to_end_orders_gaps_first_agreements_last(
    ids: SequentialIdGenerator,
) -> None:
    session_ids = SequentialIdGenerator()
    session_id = new_session_id(session_ids)

    gap = _finding(
        ids, "No training signal exists for the filter agent.", confidence=Confidence.LOAD_BEARING
    )
    agree_a = _finding(
        ids,
        "Inference latency is acceptable in practice.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.SUPPORTING,
    )
    agree_b = _finding(
        ids,
        "Latency stays acceptable when the filter runs.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.SUPPORTING,
    )
    embedder = FakeEmbedder(_same_embeddings(agree_a.claim_text, agree_b.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({agree_a.claim_text, agree_b.claim_text}): AdjudicationVerdict.SAME}
    )
    narrator = EchoNarrator(
        emergent_gaps=(EmergentGap(description="connect-the-dots gap"),),
        recommendations=("If latency matters, prototype the filter first.",),
    )

    result = synthesize(
        findings=[gap, agree_a, agree_b],
        runs=[],
        embedder=embedder,
        adjudicator=adjudicator,
        narrator=narrator,
        id_generator=ids,
        session_id=session_id,
        brief_version=1,
    )

    # Persisted Synthesis references findings by id.
    assert result.synthesis.session_id == session_id
    assert len(result.synthesis.declared_gaps) == 1
    assert len(result.synthesis.agreement_clusters) == 1
    assert result.synthesis.conditional_recommendations == (
        "If latency matters, prototype the filter first.",
    )
    assert result.synthesis.emergent_gaps[0].description == "connect-the-dots gap"

    # Rendered output: gap leads (primary), agreement trails (agreements).
    assert result.narration_fell_back is False
    assert len(result.rendered.primary) == 1
    assert result.rendered.primary[0].kind == "declared_gap"
    assert len(result.rendered.agreements) == 1
    assert result.rendered.agreements[0].kind == "agreement"
    assert len(result.rendered.emergent_gaps) == 1


def test_synthesize_inline_trust_marks_contradicted_cluster_alarming(
    ids: SequentialIdGenerator,
) -> None:
    a = _finding(
        ids,
        "The benchmark shows a 2x speedup.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.LOAD_BEARING,
        verification_status=VerificationStatus.CONTRADICTED,
    )
    b = _finding(
        ids,
        "A 2x speedup is observed on the benchmark.",
        claim_type=ClaimType.EMPIRICAL,
        confidence=Confidence.LOAD_BEARING,
    )
    embedder = FakeEmbedder(_same_embeddings(a.claim_text, b.claim_text))
    adjudicator = FakeAdjudicator(
        {frozenset({a.claim_text, b.claim_text}): AdjudicationVerdict.SAME}
    )

    result = synthesize(
        findings=[a, b],
        runs=[],
        embedder=embedder,
        adjudicator=adjudicator,
        narrator=EchoNarrator(),
        id_generator=ids,
        session_id=new_session_id(SequentialIdGenerator()),
        brief_version=1,
    )

    # The contradicted, multi-lens claim leads as an alarming tension.
    assert result.rendered.primary[0].kind == "tension"
    trust = result.rendered.primary[0].trust
    assert trust.alarming is True
    assert VerificationStatus.CONTRADICTED in trust.verification_markers
    assert trust.cross_lens_count == 2


def test_synthesize_marks_revised_under_challenge_no_evidence(
    ids: SequentialIdGenerator,
) -> None:
    g = _finding(
        ids,
        "On reflection the circularity concern is weaker than stated.",
        confidence=Confidence.SUPPORTING,
        change_reason=ChangeReason.REVISED_RECONSIDERED,
    )
    result = synthesize(
        findings=[g],
        runs=[],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
        narrator=EchoNarrator(),
        id_generator=ids,
        session_id=new_session_id(SequentialIdGenerator()),
        brief_version=1,
    )
    item = result.rendered.primary[0]
    assert item.trust.revised_under_challenge_no_evidence is True


def test_weak_gap_collapsed_into_minor_speculative_band(
    ids: SequentialIdGenerator,
) -> None:
    weak = _finding(
        ids, "Possibly the cache eviction policy matters here.", confidence=Confidence.EXPLORATORY
    )
    strong = _finding(
        ids, "No training signal exists for the filter agent.", confidence=Confidence.LOAD_BEARING
    )
    result = synthesize(
        findings=[weak, strong],
        runs=[],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
        narrator=EchoNarrator(),
        id_generator=ids,
        session_id=new_session_id(SequentialIdGenerator()),
        brief_version=1,
    )
    primary_refs = {fid for item in result.rendered.primary for fid in item.finding_refs}
    minor_refs = {fid for item in result.rendered.minor_speculative for fid in item.finding_refs}
    assert strong.id in primary_refs
    assert weak.id in minor_refs


def test_synthesize_surfaces_zero_finding_anomaly_in_output(
    ids: SequentialIdGenerator,
) -> None:
    empty_run = _run(ids, lens_id=LensId.ARCHITECTURE, status=LensRunStatus.SUCCEEDED)
    g = _finding(ids, "No training signal exists for the filter agent.")
    result = synthesize(
        findings=[g],
        runs=[empty_run],
        embedder=FakeEmbedder({}),
        adjudicator=FakeAdjudicator(),
        narrator=EchoNarrator(),
        id_generator=ids,
        session_id=new_session_id(SequentialIdGenerator()),
        brief_version=1,
    )
    assert any("no findings" in a for a in result.rendered.anomalies)
