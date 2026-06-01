"""The typed data-model spine the entire Research Council is built on.

Every domain object is a frozen Pydantic model: immutable by construction, with
``extra="forbid"`` so unknown fields are rejected (the "invalid payloads are
rejected" acceptance criterion). Derived updates are made with ``model_copy``,
never in-place mutation.

Two shapes exist for lens output:
  * ``*Draft`` models are what a lens *emits* (no ids, no round, no verification
    status — the lens does not invent those).
  * The persisted ``Finding`` / ``LensRun`` are materialized by the runtime,
    which assigns ids, stamps the round, and sets ``verification_status`` to
    ``unverified`` pending the verifier.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    ChangeReason,
    ClaimType,
    Confidence,
    Decision,
    LensId,
    LensRunStatus,
    Mode,
    Round,
    SessionStatus,
    TensionSource,
    VerificationStatus,
    VerifierErrorSubReason,
)
from .ids import (
    ChallengeId,
    DispatchEventId,
    FindingId,
    LensRunId,
    SessionId,
    SourceId,
    SynthesisId,
    TraceRecordId,
    VerdictId,
    VerificationResultId,
)

BRIEF_SUMMARY_MAX_WORDS = 150


class FrozenModel(BaseModel):
    """Immutable base: frozen attributes, no unknown fields, enum values validated."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Sources & references ---------------------------------------------------


class Source(FrozenModel):
    """First-class, globally deduplicated source (story 13). ``canonical_id`` is
    computed by ``domain.source_ops`` from DOI / arXiv / OpenAlex / title-hash."""

    canonical_id: SourceId
    title: str
    authors: tuple[str, ...] = Field(default_factory=tuple)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    has_full_text: bool = False


class SourceRef(FrozenModel):
    """A pointer to a Source by canonical id, as carried on a Finding (story 11)."""

    canonical_id: SourceId


# --- Findings ---------------------------------------------------------------


class FindingDraft(FrozenModel):
    """What a lens emits per claim. No id/round/verification status — the runtime
    assigns those when materializing into a :class:`Finding`."""

    claim_text: str
    claim_type: ClaimType
    confidence: Confidence
    sources: list[SourceRef] = Field(default_factory=list)
    failure_modes_if_wrong: str
    responding_to: list[str] = Field(default_factory=list)  # anonymized lens ids, Round 2


class Finding(FrozenModel):
    """A typed claim, the unit of synthesis (story 11). Immutable; a challenge
    revision creates a *new* Finding pointing back via ``supersedes`` (story 110)."""

    id: FindingId
    claim_text: str
    claim_type: ClaimType
    confidence: Confidence
    sources: tuple[SourceRef, ...] = Field(default_factory=tuple)
    failure_modes_if_wrong: str
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    round: Round | None = None
    responding_to: tuple[str, ...] = Field(default_factory=tuple)
    supersedes: FindingId | None = None
    # set on a challenge revision (story 108); None for Round 1/2 findings
    change_reason: ChangeReason | None = None


class VerificationResult(FrozenModel):
    """The verifier's check on a Finding (story 12). Re-verifiable and auditable;
    keyed to a ``verifier_version`` so prompt/model upgrades invalidate cleanly."""

    id: VerificationResultId
    finding_id: FindingId
    status: VerificationStatus
    claim_checked: str
    source_canonical_id: SourceId | None = None
    quoted_passage: str | None = None  # verbatim; string-matched against source (story 40)
    evidence: str | None = None
    error_sub_reason: VerifierErrorSubReason | None = None
    verifier_version: str


# --- Lens output envelopes --------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


class LensRunOutputDraft(FrozenModel):
    """The typed envelope a lens emits in Round 1/2 (story 44). Validated in code:
    a non-refusing lens must surface at least one self-disagreement (story 45)."""

    findings: list[FindingDraft] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    disagreements_with_my_own_framing: list[str] = Field(default_factory=list)
    brief_summary: str = ""
    refused: bool = False
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def _check_envelope(self) -> LensRunOutputDraft:
        if self.refused:
            if not self.refusal_reason:
                raise ValueError("refused output must carry a refusal_reason")
            return self
        if len(self.disagreements_with_my_own_framing) < 1:
            raise ValueError(
                "disagreements_with_my_own_framing must have >= 1 entry "
                "(anti-sycophancy forcing function, story 45)"
            )
        if not self.brief_summary.strip():
            raise ValueError("brief_summary is required for a non-refusing lens")
        if _word_count(self.brief_summary) > BRIEF_SUMMARY_MAX_WORDS:
            raise ValueError(f"brief_summary exceeds {BRIEF_SUMMARY_MAX_WORDS} words")
        return self


class ScopedChallengeOutputDraft(FrozenModel):
    """The stripped envelope a scoped challenge LensRun emits (story 107): one
    revised-or-reaffirmed finding plus a mandatory change_reason. No brief_summary,
    no fresh disagreements list."""

    finding: FindingDraft
    change_reason: ChangeReason
    sibling_impact_flags: list[str] = Field(default_factory=list)


# --- Brief ------------------------------------------------------------------


class Brief(FrozenModel):
    """The structured intake artifact, versioned and immutable post-confirmation
    (stories 9, 69, 76). Edits produce a new version via ``domain.brief_ops``."""

    session_id: SessionId
    version: int
    mode: Mode
    problem_statement: str
    researcher_context: str
    success_criteria_for_deliberation: str  # NEVER drafted by master (story 74)
    scope_and_non_scope: str
    proposed_solution: str | None = None
    background_claims: tuple[Finding, ...] = Field(default_factory=tuple)
    panel_constraints: tuple[LensId, ...] | None = None
    prior_panel_consultations: tuple[LensRunId, ...] = Field(default_factory=tuple)
    confirmed: bool = False


# --- Session, dispatch, runs ------------------------------------------------


class Session(FrozenModel):
    """Top-level container for one research problem (story 8)."""

    id: SessionId
    status: SessionStatus = SessionStatus.INTAKE
    title: str | None = None
    current_brief_version: int | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)  # round-level metadata (story 19)


class DispatchEvent(FrozenModel):
    """A (brief_version, panel, timestamp) tuple distinguishing re-runs of the same
    brief with different panels without bumping brief version (story 17)."""

    id: DispatchEventId
    session_id: SessionId
    brief_version: int
    panel: tuple[LensId, ...]
    timestamp: datetime
    lens_run_ids: tuple[LensRunId, ...] = Field(default_factory=tuple)


class LensRun(FrozenModel):
    """The atomic unit of work (story 10): one lens against one brief version in
    one round. Findings are persisted separately and referenced by id; the envelope
    metadata (open_questions, disagreements, brief_summary) lives here for synthesis."""

    id: LensRunId
    lens_id: LensId
    session_id: SessionId
    brief_version: int
    round: Round | None
    status: LensRunStatus = LensRunStatus.PENDING
    dispatch_event_id: DispatchEventId | None = None
    finding_ids: tuple[FindingId, ...] = Field(default_factory=tuple)
    open_questions: tuple[str, ...] = Field(default_factory=tuple)
    disagreements_with_my_own_framing: tuple[str, ...] = Field(default_factory=tuple)
    brief_summary: str = ""
    refusal_reason: str | None = None
    raw_output: str | None = None  # preserved when status == schema_invalid (story 91)
    reused_from_version: int | None = None  # set when re-pointed during refinement (story 121)
    # Prose sibling-impact flags carried only by a scoped challenge LensRun
    # (story 112): findings this run believes are now undermined but was not asked
    # to revise. Surfaced as open_questions-style synthesis tensions, never an
    # auto-edit and never a structural depends_on (story 113).
    sibling_impact_flags: tuple[str, ...] = Field(default_factory=tuple)


# --- Challenge --------------------------------------------------------------


class Challenge(FrozenModel):
    """User<->lens re-engagement targeting a specific Finding (stories 14, 103).
    A challenge never bumps brief version (story 115)."""

    id: ChallengeId
    session_id: SessionId
    challenged_finding_ref: FindingId
    challenge_text: str
    response_ref: LensRunId | None = None
    bumps_brief_version: bool = False

    @model_validator(mode="after")
    def _never_bumps(self) -> Challenge:
        if self.bumps_brief_version:
            raise ValueError("a challenge never bumps brief version (story 115)")
        return self


# --- Synthesis --------------------------------------------------------------


class ConfidenceBreakdown(FrozenModel):
    load_bearing: int = 0
    supporting: int = 0
    exploratory: int = 0


class AgreementCluster(FrozenModel):
    finding_refs: tuple[FindingId, ...]
    count: int
    confidence_breakdown: ConfidenceBreakdown


class Tension(FrozenModel):
    finding_refs: tuple[FindingId, ...]
    description: str
    source: TensionSource


class DeclaredGap(FrozenModel):
    finding_refs: tuple[FindingId, ...]


class EmergentGap(FrozenModel):
    """Synthesizer-inferred gap from inter-lens tension; lower-trust band (story 136)."""

    description: str
    implicated_finding_refs: tuple[FindingId, ...] = Field(default_factory=tuple)


class Synthesis(FrozenModel):
    """References (never copies) Findings (story 15/138) so supersessions and
    invalidations propagate into the synthesis the researcher reads."""

    id: SynthesisId
    session_id: SessionId
    brief_version: int
    agreement_clusters: tuple[AgreementCluster, ...] = Field(default_factory=tuple)
    tensions: tuple[Tension, ...] = Field(default_factory=tuple)
    declared_gaps: tuple[DeclaredGap, ...] = Field(default_factory=tuple)
    emergent_gaps: tuple[EmergentGap, ...] = Field(default_factory=tuple)
    conditional_recommendations: tuple[str, ...] = Field(default_factory=tuple)


# --- Verdict & self-use log -------------------------------------------------


class VerdictRef(FrozenModel):
    """A loose human reference (lens, excerpt) with lazily-resolved Finding ids
    (stories 126, 127)."""

    lens: LensId
    claim_text_excerpt: str
    resolved_finding_ids: tuple[FindingId, ...] = Field(default_factory=tuple)


class Verdict(FrozenModel):
    """Researcher's later-stated truth about what worked (stories 16, 124-129)."""

    id: VerdictId
    session_ref: SessionId
    decision: Decision
    outcome_summary: str
    predictions_validated: tuple[VerdictRef, ...] = Field(default_factory=tuple)
    predictions_invalidated: tuple[VerdictRef, ...] = Field(default_factory=tuple)


# --- Verdict loader envelopes -----------------------------------------------
# The hand-authored Verdict file (story 125) carries no id and no
# resolved_finding_ids — those are minted / derived. A Draft mirrors the
# author-visible surface so the loader fails fast if either appears.


class VerdictRefDraft(FrozenModel):
    """A loose reference as an engineer writes it in the file: (lens, excerpt)
    only. Resolution to specific Findings happens later in ``verdict_ops``."""

    lens: LensId
    claim_text_excerpt: str


class VerdictDraft(FrozenModel):
    """The hand-authored Verdict envelope (stories 124-128). The id is minted on
    materialize; resolved_finding_ids are populated by the resolver."""

    session_ref: SessionId
    decision: Decision
    outcome_summary: str
    predictions_validated: list[VerdictRefDraft] = Field(default_factory=list)
    predictions_invalidated: list[VerdictRefDraft] = Field(default_factory=list)


class SelfUseLog(FrozenModel):
    """Four hand-entered fields per session close (story 153)."""

    session_ref: SessionId
    time_to_dispatch_minutes: float
    total_user_attention_minutes: float
    acted_on_output: bool
    trusted_output: bool


# --- Trace ------------------------------------------------------------------


class TraceRecord(FrozenModel):
    """Append-only raw (prompt, completion, model, timestamp) per LLM call
    (story 163), kept out of the domain objects."""

    id: TraceRecordId
    caller_ref: str  # lensrun id | verifier stage id (heterogeneous)
    prompt: str
    completion: str
    model: str
    timestamp: datetime
