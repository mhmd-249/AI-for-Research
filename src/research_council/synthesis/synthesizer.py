"""Synthesizer — the hybrid Round 3 controller (PRD stories 130-144, 156).

Synthesis is *hybrid*: deterministic code owns the structural layer (clusters,
counts, contradiction pairs, gap collections computed directly from typed
Finding fields), and a constrained LLM narration pass writes prose *over* that
fixed structure with **no authority to alter** what code computed. The
narration's output is re-validated against the computed structure — same spirit
as the two-strike code-enforced schema check in the lens runtime: if narration
moves a Finding between clusters or changes a count, it is rejected and retried,
and on a second failure synthesis falls back to structure-only (story 130/132,
177: never a pure-LLM synthesizer).

The two seams that follow that division:

* :class:`Embedder` / :class:`SameClaimAdjudicator` (re-used from
  :mod:`.clusterer`) — the same-claim clustering the counts depend on.
* :class:`Narrator` — the constrained narration pass. It receives the frozen
  :class:`ComputedStructure` and may only *add* prose: conditional
  recommendations and emergent (synthesizer-inferred) gaps, plus an echo of the
  cluster membership it operated on. The echo is what re-validation checks.

Anti-averaging falls out by construction: a split (6 vs 4) is two clusters with
two counts, never a blended "mostly agree". Agreement is displayed as counts
with a confidence breakdown, never a numerical percentage (story 135).

The deterministic layer (:func:`compute_structure`) and the output structure
(:func:`render_synthesis`) are pure functions over typed Findings + LensRuns —
fixture-testable with no store and no real LLM, mirroring the ClaimClusterer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from ..enums import (
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
    TensionSource,
    VerificationStatus,
)
from ..ids import FindingId, IdGenerator, LensRunId, SessionId, new_synthesis_id
from ..models import (
    AgreementCluster,
    ConfidenceBreakdown,
    DeclaredGap,
    EmergentGap,
    Finding,
    LensRun,
    Synthesis,
    Tension,
)
from .clusterer import Embedder, ExcludedFinding, SameClaimAdjudicator, cluster_findings

# Claim types whose two-cluster splits read as *opposing* positions rather than
# merely distinct topics: a near-duplicate-embedding pair the adjudicator splits
# on these types is a real disagreement to surface, not noise (story 137).
_OPPOSING_CLAIM_TYPES: frozenset[ClaimType] = frozenset(
    {ClaimType.FAILURE_MODE, ClaimType.MECHANISM_HYPOTHESIS}
)

# Ordinal confidence rank (story 27: ordinal, never numeric). Used only to pick
# the *strongest* tier in a cluster and to severity-order gaps — never averaged.
_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.EXPLORATORY: 1,
    Confidence.SUPPORTING: 2,
    Confidence.LOAD_BEARING: 3,
}

# Verification statuses surfaced honestly as inline markers, in alarm order.
# VERIFIED is the clean state and carries no marker.
_MARKER_ORDER: tuple[VerificationStatus, ...] = (
    VerificationStatus.CONTRADICTED,
    VerificationStatus.SOURCE_NOT_FOUND,
    VerificationStatus.VERIFIER_ERROR,
    VerificationStatus.UNVERIFIED,
    VerificationStatus.PARTIALLY_SUPPORTED,
    VerificationStatus.UNVERIFIABLE_BY_DESIGN,
)


# --- deterministic structure (computed by code, authoritative) --------------


@dataclass(frozen=True)
class ComputedAgreementCluster:
    finding_refs: tuple[FindingId, ...]
    count: int
    confidence_breakdown: ConfidenceBreakdown
    strongest_confidence: Confidence


@dataclass(frozen=True)
class ComputedDeclaredGap:
    finding_refs: tuple[FindingId, ...]
    count: int
    strongest_confidence: Confidence

    @property
    def has_cross_lens_support(self) -> bool:
        return self.count >= 2

    @property
    def is_weak(self) -> bool:
        """Exploratory-confidence gap with no cross-lens support → collapsed
        into the 'minor / speculative' subsection (story 141)."""
        return (
            self.strongest_confidence is Confidence.EXPLORATORY
            and not self.has_cross_lens_support
        )


@dataclass(frozen=True)
class ComputedTension:
    finding_refs: tuple[FindingId, ...]
    description: str
    source: TensionSource


@dataclass(frozen=True)
class ZeroFindingAnomaly:
    """A lens that returned valid schema, did not refuse, but emitted zero
    findings — surfaced as a soft anomaly, never a hard error (story 156)."""

    lens_id: LensId
    lens_run_id: LensRunId
    message: str


@dataclass(frozen=True)
class ComputedStructure:
    agreement_clusters: tuple[ComputedAgreementCluster, ...]
    tensions: tuple[ComputedTension, ...]
    declared_gaps: tuple[ComputedDeclaredGap, ...]
    anomalies: tuple[ZeroFindingAnomaly, ...]
    excluded: tuple[ExcludedFinding, ...]


# --- narration seam (constrained, non-authoritative) ------------------------


@dataclass(frozen=True)
class NarrationResult:
    """What the constrained narration pass returns. ``agreement_cluster_refs``
    and ``declared_gap_refs`` are the narrator's *echo* of the structure it
    operated on — re-validation rejects narration whose echo does not match the
    computed structure exactly (membership and counts unchangeable). The
    narrator may only *add* ``emergent_gaps`` and ``conditional_recommendations``
    over that fixed structure."""

    agreement_cluster_refs: tuple[tuple[FindingId, ...], ...]
    declared_gap_refs: tuple[tuple[FindingId, ...], ...]
    emergent_gaps: tuple[EmergentGap, ...] = ()
    conditional_recommendations: tuple[str, ...] = ()


class Narrator(Protocol):
    """Writes prose over the fixed structure. Production wires a constrained LLM;
    tests inject a deterministic fake. The narrator has no authority to alter
    clusters or counts — :func:`synthesize` re-validates its echo and discards a
    narration that tries."""

    def narrate(self, structure: ComputedStructure) -> NarrationResult: ...


class NarrationValidationError(Exception):
    """Raised when a narration result alters the computed cluster membership or
    counts. Caught by :func:`synthesize`'s two-strike loop."""


# --- output structure (the synthesis the researcher reads) ------------------


@dataclass(frozen=True)
class TrustEncoding:
    """The inline trust marker every rendered claim carries (story 142): the
    cross-lens count, the strongest confidence tier, the honest verification
    markers, and the two alarm flags."""

    cross_lens_count: int
    strongest_confidence: Confidence
    verification_markers: tuple[VerificationStatus, ...]
    # Contradicted claim several lenses leaned on → visually alarming (story 142).
    alarming: bool
    # revised_reconsidered Finding (caved, no new source) → "revised under
    # challenge — no new evidence cited" (story 114/144).
    revised_under_challenge_no_evidence: bool


RenderedKind = Literal["tension", "declared_gap", "agreement"]


@dataclass(frozen=True)
class RenderedItem:
    kind: RenderedKind
    finding_refs: tuple[FindingId, ...]
    description: str
    trust: TrustEncoding


@dataclass(frozen=True)
class RenderedEmergentGap:
    """Lower-trust band: the synthesizer connected dots no single lens asserted
    (story 143). Rendered distinctly so the researcher can distrust it."""

    description: str
    implicated_finding_refs: tuple[FindingId, ...]


@dataclass(frozen=True)
class RenderedSynthesis:
    """Synthesis-first output, ordered gaps-and-tensions before agreements
    (stories 139-143). ``primary`` is severity-ordered; weak gaps are collapsed
    into ``minor_speculative``; emergent gaps render in their own lower-trust
    band; agreements trail last."""

    primary: tuple[RenderedItem, ...]
    minor_speculative: tuple[RenderedItem, ...]
    emergent_gaps: tuple[RenderedEmergentGap, ...]
    agreements: tuple[RenderedItem, ...]
    anomalies: tuple[str, ...]
    conditional_recommendations: tuple[str, ...]


@dataclass(frozen=True)
class SynthesisResult:
    """The synthesizer's full output: the persisted :class:`Synthesis` (which
    *references* Findings by id), the readable :class:`RenderedSynthesis`, and
    whether narration fell back to structure-only after two failed strikes."""

    synthesis: Synthesis
    rendered: RenderedSynthesis
    narration_fell_back: bool


# --- winning-finding resolution ---------------------------------------------


def resolve_winning_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Drop superseded Findings (story 111): a Finding is superseded if some
    other Finding points to it via ``supersedes``. The winners are what
    synthesis references, so a challenge-driven supersession propagates without
    a stale copy (story 138)."""
    findings = list(findings)
    superseded: set[FindingId] = {
        f.supersedes for f in findings if f.supersedes is not None
    }
    return [f for f in findings if f.id not in superseded]


# --- small helpers ----------------------------------------------------------


def _strongest(members: list[Finding]) -> Confidence:
    return max(members, key=lambda m: _CONFIDENCE_RANK[m.confidence]).confidence


def _breakdown(members: list[Finding]) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        load_bearing=sum(1 for m in members if m.confidence is Confidence.LOAD_BEARING),
        supporting=sum(1 for m in members if m.confidence is Confidence.SUPPORTING),
        exploratory=sum(1 for m in members if m.confidence is Confidence.EXPLORATORY),
    )


def _markers(members: list[Finding]) -> tuple[VerificationStatus, ...]:
    present = {m.verification_status for m in members}
    return tuple(s for s in _MARKER_ORDER if s in present)


def _is_contradicted(members: list[Finding]) -> bool:
    return any(m.verification_status is VerificationStatus.CONTRADICTED for m in members)


# --- the deterministic layer ------------------------------------------------


def compute_structure(
    *,
    findings: list[Finding],
    runs: Iterable[LensRun],
    embedder: Embedder,
    adjudicator: SameClaimAdjudicator,
    similarity_threshold: float | None = None,
) -> ComputedStructure:
    """Compute the authoritative structural layer from typed Findings + LensRuns.

    ``findings`` must be the *winning* set (resolve supersessions first via
    :func:`resolve_winning_findings`). Gap-type Findings cluster into declared
    gaps; every other type clusters into agreement clusters. A non-gap cluster
    holding a ``contradicted`` Finding is lifted out of agreements into a
    contradiction tension (anti-averaging: it leads, it does not reassure).
    Opposing failure_mode / mechanism_hypothesis splits become opposing-pair
    tensions. A succeeded LensRun with zero findings is a soft anomaly."""
    cluster_kwargs = {}
    if similarity_threshold is not None:
        cluster_kwargs["similarity_threshold"] = similarity_threshold

    by_id: dict[FindingId, Finding] = {f.id: f for f in findings}
    gap_findings = [f for f in findings if f.claim_type is ClaimType.GAP]
    other_findings = [f for f in findings if f.claim_type is not ClaimType.GAP]

    gap_result = cluster_findings(
        gap_findings, embedder=embedder, adjudicator=adjudicator, **cluster_kwargs
    )
    other_result = cluster_findings(
        other_findings, embedder=embedder, adjudicator=adjudicator, **cluster_kwargs
    )

    agreement_clusters: list[ComputedAgreementCluster] = []
    tensions: list[ComputedTension] = []

    for cluster in other_result.clusters:
        members = [by_id[fid] for fid in cluster.finding_ids]
        if _is_contradicted(members):
            tensions.append(
                ComputedTension(
                    finding_refs=cluster.finding_ids,
                    description=(
                        f"{len(members)} lens(es) leaned on this claim, but "
                        "verification contradicted it."
                    ),
                    source=TensionSource.CONTRADICTION,
                )
            )
        else:
            agreement_clusters.append(
                ComputedAgreementCluster(
                    finding_refs=cluster.finding_ids,
                    count=len(members),
                    confidence_breakdown=_breakdown(members),
                    strongest_confidence=_strongest(members),
                )
            )

    # Opposing-pair tensions: a high-similarity pair the adjudicator split, where
    # both sides are the contradiction-prone claim types. The split is preserved
    # as the tension — never averaged into the cluster (story 137).
    for rec in other_result.adjudications:
        if rec.verdict.value == "same":
            continue
        fa, fb = by_id[rec.finding_a], by_id[rec.finding_b]
        if (
            fa.claim_type in _OPPOSING_CLAIM_TYPES
            and fb.claim_type in _OPPOSING_CLAIM_TYPES
        ):
            tensions.append(
                ComputedTension(
                    finding_refs=(rec.finding_a, rec.finding_b),
                    description=(
                        "Two lenses asserted opposing positions on the same "
                        "point; preserved as a split, not averaged."
                    ),
                    source=TensionSource.OPPOSING_PAIR,
                )
            )

    declared_gaps = tuple(
        ComputedDeclaredGap(
            finding_refs=cluster.finding_ids,
            count=len(cluster.finding_ids),
            strongest_confidence=_strongest([by_id[fid] for fid in cluster.finding_ids]),
        )
        for cluster in gap_result.clusters
    )

    anomalies = tuple(
        ZeroFindingAnomaly(
            lens_id=run.lens_id,
            lens_run_id=run.id,
            message=(
                f"Lens {run.lens_id.value} succeeded but produced no findings "
                "— possible silent failure."
            ),
        )
        for run in runs
        if run.status is LensRunStatus.SUCCEEDED and len(run.finding_ids) == 0
    )

    return ComputedStructure(
        agreement_clusters=tuple(agreement_clusters),
        tensions=tuple(tensions),
        declared_gaps=declared_gaps,
        anomalies=anomalies,
        excluded=tuple(gap_result.excluded) + tuple(other_result.excluded),
    )


# --- narration re-validation ------------------------------------------------


def _ref_set(refs: Iterable[Iterable[FindingId]]) -> set[frozenset[FindingId]]:
    return {frozenset(group) for group in refs}


def validate_narration(structure: ComputedStructure, narration: NarrationResult) -> None:
    """Reject any narration whose echoed cluster membership/counts differ from
    the computed structure. This is what keeps the LLM non-authoritative."""
    computed_agreements = _ref_set(c.finding_refs for c in structure.agreement_clusters)
    if _ref_set(narration.agreement_cluster_refs) != computed_agreements:
        raise NarrationValidationError(
            "narration altered agreement cluster membership/counts"
        )
    computed_gaps = _ref_set(g.finding_refs for g in structure.declared_gaps)
    if _ref_set(narration.declared_gap_refs) != computed_gaps:
        raise NarrationValidationError(
            "narration altered declared-gap membership/counts"
        )


def _structure_only_narration() -> NarrationResult:
    """The fallback after two failed narration strikes: keep the deterministic
    structure, drop all LLM-added prose (no emergent gaps, no recommendations)."""
    return NarrationResult(
        agreement_cluster_refs=(),
        declared_gap_refs=(),
        emergent_gaps=(),
        conditional_recommendations=(),
    )


def _run_narration(
    narrator: Narrator, structure: ComputedStructure
) -> tuple[NarrationResult, bool]:
    """Two-strike narration: narrate, validate; on failure retry once; on a
    second failure fall back to structure-only. Returns (result, fell_back)."""
    for _ in range(2):
        candidate = narrator.narrate(structure)
        try:
            validate_narration(structure, candidate)
        except NarrationValidationError:
            continue
        return candidate, False
    return _structure_only_narration(), True


# --- output rendering -------------------------------------------------------


def _trust_for(members: list[Finding]) -> TrustEncoding:
    markers = _markers(members)
    count = len(members)
    return TrustEncoding(
        cross_lens_count=count,
        strongest_confidence=_strongest(members),
        verification_markers=markers,
        alarming=VerificationStatus.CONTRADICTED in markers and count >= 2,
        revised_under_challenge_no_evidence=any(
            m.change_reason is not None and m.change_reason.value == "revised_reconsidered"
            for m in members
        ),
    )


def _severity(item: RenderedItem) -> int:
    """Higher = surfaced earlier. Contradictions lead (alarming), then load-
    bearing gaps, then opposing-pair tensions, then weaker gaps. Cross-lens
    count breaks ties within a band."""
    trust = item.trust
    base = 0
    if item.kind == "tension":
        base = 100 if trust.alarming else 50
    elif item.kind == "declared_gap":
        base = 10 * _CONFIDENCE_RANK[trust.strongest_confidence]
    return base + trust.cross_lens_count


def render_synthesis(
    structure: ComputedStructure,
    narration: NarrationResult,
    by_id: dict[FindingId, Finding],
) -> RenderedSynthesis:
    """Order the computed structure into the synthesis the researcher reads:
    gaps + tensions first (severity-ordered), weak gaps collapsed, emergent gaps
    in their own lower-trust band, agreements last."""

    def members_of(refs: tuple[FindingId, ...]) -> list[Finding]:
        return [by_id[fid] for fid in refs]

    primary: list[RenderedItem] = []
    minor: list[RenderedItem] = []

    for tension in structure.tensions:
        members = members_of(tension.finding_refs)
        primary.append(
            RenderedItem(
                kind="tension",
                finding_refs=tension.finding_refs,
                description=tension.description,
                trust=_trust_for(members),
            )
        )

    for gap in structure.declared_gaps:
        members = members_of(gap.finding_refs)
        item = RenderedItem(
            kind="declared_gap",
            finding_refs=gap.finding_refs,
            description=members[0].claim_text if members else "",
            trust=_trust_for(members),
        )
        (minor if gap.is_weak else primary).append(item)

    primary.sort(key=_severity, reverse=True)

    agreements = tuple(
        RenderedItem(
            kind="agreement",
            finding_refs=cluster.finding_refs,
            description=members_of(cluster.finding_refs)[0].claim_text,
            trust=_trust_for(members_of(cluster.finding_refs)),
        )
        for cluster in structure.agreement_clusters
    )

    emergent = tuple(
        RenderedEmergentGap(
            description=eg.description,
            implicated_finding_refs=eg.implicated_finding_refs,
        )
        for eg in narration.emergent_gaps
    )

    return RenderedSynthesis(
        primary=tuple(primary),
        minor_speculative=tuple(minor),
        emergent_gaps=emergent,
        agreements=agreements,
        anomalies=tuple(a.message for a in structure.anomalies),
        conditional_recommendations=tuple(narration.conditional_recommendations),
    )


# --- the persisted Synthesis object (references, never copies) --------------


def _to_synthesis(
    *,
    synthesis_id_gen: IdGenerator,
    session_id: SessionId,
    brief_version: int,
    structure: ComputedStructure,
    narration: NarrationResult,
) -> Synthesis:
    return Synthesis(
        id=new_synthesis_id(synthesis_id_gen),
        session_id=session_id,
        brief_version=brief_version,
        agreement_clusters=tuple(
            AgreementCluster(
                finding_refs=c.finding_refs,
                count=c.count,
                confidence_breakdown=c.confidence_breakdown,
            )
            for c in structure.agreement_clusters
        ),
        tensions=tuple(
            Tension(
                finding_refs=t.finding_refs,
                description=t.description,
                source=t.source,
            )
            for t in structure.tensions
        ),
        declared_gaps=tuple(
            DeclaredGap(finding_refs=g.finding_refs) for g in structure.declared_gaps
        ),
        emergent_gaps=narration.emergent_gaps,
        conditional_recommendations=narration.conditional_recommendations,
    )


# --- public entrypoint ------------------------------------------------------


def synthesize(
    *,
    findings: Iterable[Finding],
    runs: Iterable[LensRun],
    embedder: Embedder,
    adjudicator: SameClaimAdjudicator,
    narrator: Narrator,
    id_generator: IdGenerator,
    session_id: SessionId,
    brief_version: int,
    similarity_threshold: float | None = None,
) -> SynthesisResult:
    """Run the hybrid Round 3 synthesizer end to end.

    1. Resolve winning Findings (drop superseded — propagation, not copying).
    2. Compute the authoritative structure (clusters, counts, tensions, gaps,
       anomalies) deterministically.
    3. Narrate over the fixed structure with two-strike re-validation; fall back
       to structure-only on a second failure.
    4. Materialize the persisted :class:`Synthesis` (referencing Findings) and
       the readable :class:`RenderedSynthesis`."""
    winners = resolve_winning_findings(findings)
    runs = list(runs)
    by_id = {f.id: f for f in winners}

    structure = compute_structure(
        findings=winners,
        runs=runs,
        embedder=embedder,
        adjudicator=adjudicator,
        similarity_threshold=similarity_threshold,
    )
    narration, fell_back = _run_narration(narrator, structure)

    synthesis = _to_synthesis(
        synthesis_id_gen=id_generator,
        session_id=session_id,
        brief_version=brief_version,
        structure=structure,
        narration=narration,
    )
    rendered = render_synthesis(structure, narration, by_id)

    return SynthesisResult(
        synthesis=synthesis,
        rendered=rendered,
        narration_fell_back=fell_back,
    )
