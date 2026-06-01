"""ChallengeOrchestrator — the scoped single-Finding challenge loop (Layer 4.1,
stories 102-116).

A challenge **edits inputs, never outputs**: the researcher challenges one
specific Finding, the lens re-engages that single claim, and synthesis is
*recomputed* — synthesis constructs (agreement clusters, tensions, gaps) are
never challenged directly (stories 104, 166). The orchestrator owns the
mechanics around that:

* **Scoped input (stories 105, 106).** It builds the narrow contract input
  ``(brief, this lens's prior Findings on this brief, challenge_text,
  challenged_finding)`` and hands it to a :class:`ScopedChallengeRunner`. It does
  NOT pass other lenses' work — a challenge is not cross-pollination.

* **Stripped envelope (story 107).** The runner emits a
  :class:`~research_council.models.ScopedChallengeOutputDraft` — one
  revised-or-reaffirmed Finding plus a mandatory ``change_reason`` and optional
  prose sibling-impact flags. The envelope has no ``brief_summary`` and no fresh
  ``disagreements_with_my_own_framing`` (those fields simply do not exist on the
  model, so ``extra="forbid"`` rejects them).

* **Supersession chain (stories 108, 110, 111).** Every accepted response
  materializes a *new* response Finding carrying the ``change_reason`` and a
  ``supersedes`` pointer to the challenged Finding. The old Finding stays
  immutable and persisted but drops out of the winning set, so the new one wins
  at synthesis render time. A ``withdrawn`` response is a tombstone: it
  supersedes the challenged Finding and is itself excluded from winners (the
  claim is gone, not replaced) — both handled in
  :func:`~research_council.synthesis.synthesizer.resolve_winning_findings`.

* **Sibling-impact flags (stories 112, 113).** Prose flags ride on the scoped
  challenge LensRun's ``sibling_impact_flags`` and surface as
  ``SIBLING_IMPACT_FLAG`` synthesis tensions — never a silent stale Finding,
  never an auto-edit, never a structural ``depends_on``.

* **Anti-sycophancy is detect-and-label, not block (stories 109, 114, 168).**
  ``reaffirmed_against_challenge`` is a first-class, encouraged outcome (the lens
  may hold the Finding and tell the researcher they are wrong);
  ``revised_reconsidered`` is accepted and visibly marked downstream, never
  gated.

* **No brief-version bump; re-synthesis fires on accept (stories 115, 116).** A
  challenge is dispatch-time activity on the existing brief (the ``Challenge``
  model enforces ``bumps_brief_version=False``). On any accepted response the
  orchestrator fires the injected :data:`Resynthesizer` so the synthesis the
  researcher reads always reflects the current winning set.

Like the round controllers, the actual lens LLM loop is a seam
(:class:`ScopedChallengeRunner`) and re-synthesis is a seam
(:data:`Resynthesizer`), so the orchestrator is fixture-testable with no real
LLM and no real embedder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..enums import ChangeReason, LensId, LensRunStatus, VerificationStatus
from ..ids import FindingId, IdGenerator, SessionId, new_finding_id, new_lens_run_id
from ..lenses.roster import LensConfig, get_lens_config
from ..models import (
    Brief,
    Challenge,
    Finding,
    FindingDraft,
    LensRun,
    ScopedChallengeOutputDraft,
)
from ..store.interface import SessionStore
from ..synthesis import (
    DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD,
    Embedder,
    Narrator,
    SameClaimAdjudicator,
    SynthesisResult,
    resolve_winning_findings,
    synthesize,
)
from .inputs import build_prior_self

# --- the scoped challenge input (the four contract inputs, story 106) --------


@dataclass(frozen=True)
class ScopedChallengeInput:
    """Exactly the inputs a scoped challenge LensRun receives (story 106):
    ``(brief, this lens's prior Findings on this brief, challenge_text,
    challenged_finding)``. There is deliberately **no** cross-pollination field —
    a challenge does not see other lenses' work (Round 2 owns that)."""

    brief: Brief
    lens_config: LensConfig
    challenged_finding: Finding
    prior_self: list[Finding]
    challenge_text: str


def build_scoped_challenge_prompt(inputs: ScopedChallengeInput) -> str:
    """Assemble the scoped challenge prompt from the four contract inputs only.

    The challenged Finding and the researcher's argument lead; the lens's own
    prior Findings provide the context it is allowed to reconsider. No peer
    outputs appear (story 106)."""
    challenged = inputs.challenged_finding
    lines = [
        f"## Brief\nproblem_statement: {inputs.brief.problem_statement}",
        f"proposed_solution: {inputs.brief.proposed_solution or '(none)'}",
        "",
        "## The Finding you are being challenged on",
        f"  ({challenged.claim_type.value}, {challenged.confidence.value}) "
        f"{challenged.claim_text}",
        f"  failure_modes_if_wrong: {challenged.failure_modes_if_wrong}",
        "",
        "## The researcher's challenge",
        inputs.challenge_text,
    ]
    if inputs.prior_self:
        lines.append("")
        lines.append("## Your other prior findings on this brief (context only)")
        for f in inputs.prior_self:
            if f.id == challenged.id:
                continue
            lines.append(f"  ({f.claim_type.value}, {f.confidence.value}) {f.claim_text}")
    lines.append("")
    lines.append(
        "Re-engage ONLY this Finding. Emit a stripped envelope: one "
        "revised-or-reaffirmed finding, a mandatory change_reason, and optional "
        "sibling_impact_flags. You may reaffirm and tell the researcher they are "
        "wrong — caving is not required."
    )
    return "\n".join(lines)


# --- runner seam (the scoped lens loop; fake in tests) ----------------------


@dataclass(frozen=True)
class ScopedChallengeAccepted:
    """The lens engaged and produced a stripped envelope."""

    output: ScopedChallengeOutputDraft
    status: Literal["accepted"] = "accepted"


@dataclass(frozen=True)
class ScopedChallengeRefused:
    """The lens declined to re-engage. No Finding changes; no re-synthesis fires
    (story 116: re-synthesis triggers only on an *accepted* response)."""

    reason: str
    status: Literal["refused"] = "refused"


ScopedChallengeOutcome = ScopedChallengeAccepted | ScopedChallengeRefused


class ScopedChallengeRunner(Protocol):
    """Runs the scoped single-Finding lens loop. Production wires the LLM (the
    stripped-envelope analogue of ``run_lens``); tests inject a fake."""

    async def run(self, inputs: ScopedChallengeInput) -> ScopedChallengeOutcome: ...


# --- re-synthesis seam (story 116) ------------------------------------------

# Recomputes synthesis over the current winning set and returns the result. The
# orchestrator fires it on any accepted response and persists what it returns.
Resynthesizer = Callable[[], SynthesisResult]


# --- result -----------------------------------------------------------------


@dataclass(frozen=True)
class ChallengeResult:
    """The outcome of one scoped challenge.

    On an accepted response: ``response_finding`` is the new (superseding)
    Finding, ``change_reason`` records what kind of response it was, and
    ``resynthesis`` carries the recomputed synthesis. On a refusal: ``accepted``
    is False, ``response_finding`` / ``resynthesis`` are None, and
    ``refusal_reason`` is set. ``challenge`` is the persisted Challenge with its
    ``response_ref`` wired to the scoped run."""

    challenge: Challenge
    scoped_run: LensRun
    accepted: bool
    response_finding: Finding | None = None
    change_reason: ChangeReason | None = None
    sibling_impact_flags: tuple[str, ...] = field(default_factory=tuple)
    refusal_reason: str | None = None
    resynthesis: SynthesisResult | None = None


# --- internals --------------------------------------------------------------


def _resolve_emitting_run(
    store: SessionStore, session_id: SessionId, finding_id: FindingId
) -> LensRun:
    """Find the LensRun that emitted the challenged Finding — the transitive
    identification of story 103. A challenge never bumps brief version (story
    115), so the Finding lives on a run against the session's current brief."""
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"session {session_id} not found")
    version = session.current_brief_version
    if version is None:
        raise ValueError(f"session {session_id} has no current brief version")
    for run in store.list_lens_runs_for_brief(session_id, version):
        if finding_id in run.finding_ids:
            return run
    raise ValueError(
        f"no lens run on brief v{version} emitted finding {finding_id}"
    )


def _build_prior_self(
    store: SessionStore, session_id: SessionId, lens_id: LensId, version: int
) -> list[Finding]:
    """This lens's current (winning) Findings on this brief, with verifier
    results stripped — the same projection Round 2 prior_self uses (story 49)."""
    runs = store.list_lens_runs_for_lens(session_id, lens_id, version)
    findings: list[Finding] = []
    for run in runs:
        for fid in run.finding_ids:
            found = store.get_finding(fid)
            if found is not None:
                findings.append(found)
    return build_prior_self(resolve_winning_findings(findings))


def _materialize_response(
    draft: FindingDraft,
    *,
    id_generator: IdGenerator,
    change_reason: ChangeReason,
    supersedes: FindingId,
) -> Finding:
    """Materialize the response FindingDraft into a new persisted Finding that
    supersedes the challenged one and carries the ``change_reason`` (story 110).
    ``round`` is None — a challenge revision is outside the round model."""
    return Finding(
        id=new_finding_id(id_generator),
        claim_text=draft.claim_text,
        claim_type=draft.claim_type,
        confidence=draft.confidence,
        sources=tuple(draft.sources),
        failure_modes_if_wrong=draft.failure_modes_if_wrong,
        verification_status=VerificationStatus.UNVERIFIED,
        round=None,
        responding_to=tuple(draft.responding_to),
        supersedes=supersedes,
        change_reason=change_reason,
    )


# --- public entrypoint ------------------------------------------------------


async def run_challenge(
    *,
    challenge: Challenge,
    store: SessionStore,
    runner: ScopedChallengeRunner,
    resynthesize: Resynthesizer,
    id_generator: IdGenerator,
    get_config: Callable[[LensId], LensConfig] = get_lens_config,
) -> ChallengeResult:
    """Run one scoped single-Finding challenge end to end.

    1. Resolve the challenged Finding and its emitting LensRun (→ lens, brief
       version) — the lens is not re-run wholesale (story 105).
    2. Build the scoped input ``(brief, prior self, challenge_text, finding)``
       and dispatch to the runner (story 106).
    3. On refusal: persist a refused scoped run, wire the Challenge, fire NO
       re-synthesis. On accept: materialize the superseding response Finding,
       persist the scoped run (carrying any sibling-impact flags), wire the
       Challenge, and fire re-synthesis (story 116).
    """
    challenged = store.get_finding(challenge.challenged_finding_ref)
    if challenged is None:
        raise ValueError(
            f"challenged finding {challenge.challenged_finding_ref} not found"
        )

    emitting_run = _resolve_emitting_run(store, challenge.session_id, challenged.id)
    lens_id = emitting_run.lens_id
    brief_version = emitting_run.brief_version

    brief = store.get_brief(challenge.session_id, brief_version)
    if brief is None:
        raise ValueError(
            f"brief v{brief_version} for session {challenge.session_id} not found"
        )

    scoped_input = ScopedChallengeInput(
        brief=brief,
        lens_config=get_config(lens_id),
        challenged_finding=challenged,
        prior_self=_build_prior_self(
            store, challenge.session_id, lens_id, brief_version
        ),
        challenge_text=challenge.challenge_text,
    )

    outcome = await runner.run(scoped_input)
    run_id = new_lens_run_id(id_generator)

    if isinstance(outcome, ScopedChallengeRefused):
        scoped_run = LensRun(
            id=run_id,
            lens_id=lens_id,
            session_id=challenge.session_id,
            brief_version=brief_version,
            round=None,
            status=LensRunStatus.REFUSED,
            refusal_reason=outcome.reason,
        )
        store.save_lens_run(scoped_run)
        updated = challenge.model_copy(update={"response_ref": run_id})
        store.save_challenge(updated)
        return ChallengeResult(
            challenge=updated,
            scoped_run=scoped_run,
            accepted=False,
            refusal_reason=outcome.reason,
        )

    draft = outcome.output
    response_finding = _materialize_response(
        draft.finding,
        id_generator=id_generator,
        change_reason=draft.change_reason,
        supersedes=challenged.id,
    )
    store.save_finding(response_finding)

    scoped_run = LensRun(
        id=run_id,
        lens_id=lens_id,
        session_id=challenge.session_id,
        brief_version=brief_version,
        round=None,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(response_finding.id,),
        sibling_impact_flags=tuple(draft.sibling_impact_flags),
    )
    store.save_lens_run(scoped_run)

    updated = challenge.model_copy(update={"response_ref": run_id})
    store.save_challenge(updated)

    # Re-synthesis fires on the accepted response (story 116); the orchestrator
    # owns persistence of what the seam computes.
    resynthesis = resynthesize()
    store.save_synthesis(resynthesis.synthesis)

    return ChallengeResult(
        challenge=updated,
        scoped_run=scoped_run,
        accepted=True,
        response_finding=response_finding,
        change_reason=draft.change_reason,
        sibling_impact_flags=tuple(draft.sibling_impact_flags),
        resynthesis=resynthesis,
    )


# --- convenience wiring -----------------------------------------------------


def make_resynthesizer(
    *,
    store: SessionStore,
    session_id: SessionId,
    brief_version: int,
    embedder: Embedder,
    adjudicator: SameClaimAdjudicator,
    narrator: Narrator,
    id_generator: IdGenerator,
    similarity_threshold: float = DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD,
) -> Resynthesizer:
    """Wire a :data:`Resynthesizer` that recomputes synthesis over the session's
    current findings and runs (including the just-persisted scoped challenge
    run). Pure compute — the orchestrator persists the result."""

    def _resynthesize() -> SynthesisResult:
        runs = store.list_lens_runs_for_brief(session_id, brief_version)
        findings = store.list_findings_for_session(session_id)
        return synthesize(
            findings=findings,
            runs=runs,
            embedder=embedder,
            adjudicator=adjudicator,
            narrator=narrator,
            id_generator=id_generator,
            session_id=session_id,
            brief_version=brief_version,
            similarity_threshold=similarity_threshold,
        )

    return _resynthesize
