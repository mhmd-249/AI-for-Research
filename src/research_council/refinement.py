"""RefinementOrchestrator — the brief-edit refinement loop (Layer 4.2, stories 117-123).

User-declares-scope / system-proposes — the same pattern PanelRouter Screen 2 uses
for panels, applied here to brief edits. A brief edit produces a new brief version
(immutability holds, locked story 9/76 via :func:`domain.brief_ops.revise_brief`);
the master diffs old vs. new and *proposes* an invalidation set with one-line
reasoning per affected lens covering BOTH re-run and reuse decisions; the user
confirms or overrides; reused LensRuns are re-pointed to the new version with a
``reused_from_version`` marker, and invalidated LensRuns re-dispatch as fresh runs.

The material-edit rule (story 120) gives "materially different" a concrete,
mechanically-proposable definition:

    an edit is material to a lens iff that lens had a ``load_bearing`` or
    ``supporting`` Finding whose ``claim_text`` referenced the edited content.

``exploratory`` findings and untouched fields default to reuse. "Referenced" is a
keyword-overlap heuristic (:class:`KeywordOverlapDetector`) — deliberately
imperfect: the known limitation (story 123) is that implicit-assumption staleness
the heuristic cannot see slips through, with the user-confirm step as the
mitigation and the 5% manual audit as the backstop.

Everything here is pure functions over typed Briefs + LensRuns + Findings —
fixture-testable with no store and no real LLM, mirroring the Synthesizer and
ClaimClusterer. :func:`propose_invalidation` is deterministic and authoritative
about the *proposal*; the user owns the final set, applied by
:func:`apply_invalidation`.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .domain.brief_ops import revise_brief
from .enums import Confidence, LensId
from .ids import FindingId, IdGenerator, LensRunId, new_lens_run_id
from .models import Brief, Finding, LensRun

# Brief *content* fields a Finding's claim_text could plausibly reference. Identity
# / lineage / dispatch-config fields (session_id, version, panel_constraints,
# prior_panel_consultations, confirmed) are not content and never make an edit
# material — a panel_constraints edit, for instance, is a dispatch change.
CONTENT_FIELDS: tuple[str, ...] = (
    "problem_statement",
    "researcher_context",
    "success_criteria_for_deliberation",
    "scope_and_non_scope",
    "proposed_solution",
)

# Only load-bearing / supporting findings can make an edit material (story 120);
# exploratory findings default to reuse.
_MATERIAL_CONFIDENCES: frozenset[Confidence] = frozenset(
    {Confidence.LOAD_BEARING, Confidence.SUPPORTING}
)

Disposition = Literal["rerun", "reuse"]


# --- brief diff -------------------------------------------------------------


@dataclass(frozen=True)
class FieldChange:
    """One changed brief content field, carrying both sides so the material-edit
    rule can test the OLD value (what a Finding could have referenced)."""

    field: str
    old_value: str
    new_value: str


def diff_briefs(old: Brief, new: Brief) -> tuple[FieldChange, ...]:
    """Return the content fields that differ between two brief versions, in
    :data:`CONTENT_FIELDS` order. ``None`` (e.g. an absent proposed_solution) is
    normalized to the empty string so adding/removing a field is a real change."""
    changes: list[FieldChange] = []
    for name in CONTENT_FIELDS:
        old_val = getattr(old, name) or ""
        new_val = getattr(new, name) or ""
        if old_val != new_val:
            changes.append(FieldChange(field=name, old_value=old_val, new_value=new_val))
    return tuple(changes)


# --- "referenced the edited content" heuristic ------------------------------


class ReferenceDetector(Protocol):
    """Decides whether a Finding's ``claim_text`` referenced some edited brief
    content. Production may wire something richer; the default is keyword overlap.
    The heuristic is the proposal, never the final word — the user confirms."""

    def references(self, claim_text: str, edited_content: str) -> bool: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words long enough to clear the length filter but too generic to count as
# a real topical reference — excluded so "the model is a thing" doesn't "reference"
# every brief.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "this", "that", "these", "those", "with", "from", "have", "will", "would",
        "could", "should", "their", "there", "which", "when", "what", "into",
        "than", "then", "they", "them", "your", "about", "over", "under", "been",
        "being", "does", "also", "such", "some", "more", "most", "other", "much",
        "many", "very", "only", "must", "shall", "here", "where", "while", "each",
        "both", "same", "thing", "things", "make", "made", "using", "used",
    }
)


@dataclass(frozen=True)
class KeywordOverlapDetector:
    """Default heuristic: a claim references edited content iff they share a
    distinctive term (length >= ``min_term_length``, not a stopword). Deliberately
    coarse — it errs toward proposing re-run when in doubt, with the user as the
    backstop (story 123)."""

    min_term_length: int = 4

    def _terms(self, text: str) -> set[str]:
        return {
            tok
            for tok in _TOKEN_RE.findall(text.lower())
            if len(tok) >= self.min_term_length and tok not in _STOPWORDS
        }

    def references(self, claim_text: str, edited_content: str) -> bool:
        return bool(self._terms(claim_text) & self._terms(edited_content))


# --- the proposal -----------------------------------------------------------


@dataclass(frozen=True)
class LensReuseProposal:
    """The system's proposed disposition for one LensRun, with one-line reasoning
    (story 119) covering whichever decision it landed on."""

    lens_run_id: LensRunId
    lens_id: LensId
    disposition: Disposition
    reasoning: str
    triggering_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvalidationProposal:
    """The full system proposal for a brief edit: the new (unconfirmed) brief
    version, the field diff, and a per-lens disposition. The user confirms it
    as-is or hands :func:`apply_invalidation` an override set."""

    old_brief_version: int
    new_brief: Brief
    diff: tuple[FieldChange, ...]
    lens_proposals: tuple[LensReuseProposal, ...]

    @property
    def proposed_rerun_ids(self) -> frozenset[LensRunId]:
        return frozenset(
            p.lens_run_id for p in self.lens_proposals if p.disposition == "rerun"
        )


@dataclass(frozen=True)
class RefinementResult:
    """The outcome of applying a (confirmed or overridden) invalidation set:
    reused runs re-pointed to the new version, invalidated runs re-dispatched as
    fresh PENDING runs."""

    new_brief: Brief
    reused_runs: tuple[LensRun, ...] = field(default_factory=tuple)
    redispatched_runs: tuple[LensRun, ...] = field(default_factory=tuple)


# --- classification ---------------------------------------------------------


def _classify_run(
    run: LensRun,
    run_findings: list[Finding],
    diff: tuple[FieldChange, ...],
    detector: ReferenceDetector,
) -> LensReuseProposal:
    """Apply the material-edit rule to one run and produce its reasoning."""
    triggering_fields: list[str] = []
    triggering_confidences: set[Confidence] = set()
    material_count = 0

    for change in diff:
        matched = [
            f
            for f in run_findings
            if f.confidence in _MATERIAL_CONFIDENCES
            and detector.references(f.claim_text, change.old_value)
        ]
        if matched:
            triggering_fields.append(change.field)
            triggering_confidences.update(f.confidence for f in matched)
            material_count += len(matched)

    lens = run.lens_id.value
    if triggering_fields:
        confs = ", ".join(sorted(c.value for c in triggering_confidences))
        fields = ", ".join(triggering_fields)
        reasoning = (
            f"{lens}: {material_count} {confs} finding(s) referenced edited "
            f"{fields} → re-run."
        )
        return LensReuseProposal(
            lens_run_id=run.id,
            lens_id=run.lens_id,
            disposition="rerun",
            reasoning=reasoning,
            triggering_fields=tuple(triggering_fields),
        )

    if not diff:
        reasoning = f"{lens}: no brief content changed → reuse."
    else:
        changed = ", ".join(c.field for c in diff)
        reasoning = (
            f"{lens}: no load_bearing/supporting finding referenced edited "
            f"{changed} (exploratory/untouched default to reuse) → reuse."
        )
    return LensReuseProposal(
        lens_run_id=run.id,
        lens_id=run.lens_id,
        disposition="reuse",
        reasoning=reasoning,
    )


def propose_invalidation(
    *,
    old_brief: Brief,
    new_brief: Brief,
    runs: Iterable[LensRun],
    findings: Iterable[Finding],
    detector: ReferenceDetector | None = None,
) -> InvalidationProposal:
    """Diff ``old_brief`` vs ``new_brief`` and propose, per LensRun, whether to
    re-run or reuse under the material-edit rule (story 120), with one-line
    reasoning for each (story 119). Deterministic and store-free."""
    detector = detector or KeywordOverlapDetector()
    by_id: dict[FindingId, Finding] = {f.id: f for f in findings}
    diff = diff_briefs(old_brief, new_brief)

    proposals = tuple(
        _classify_run(
            run,
            [by_id[fid] for fid in run.finding_ids if fid in by_id],
            diff,
            detector,
        )
        for run in runs
    )
    return InvalidationProposal(
        old_brief_version=old_brief.version,
        new_brief=new_brief,
        diff=diff,
        lens_proposals=proposals,
    )


def refine_brief(
    prior: Brief,
    *,
    edits: Mapping[str, Any],
    runs: Iterable[LensRun],
    findings: Iterable[Finding],
    detector: ReferenceDetector | None = None,
) -> InvalidationProposal:
    """Orchestrator entrypoint: a brief edit produces a new version (story 118)
    and the system's proposed invalidation set in one step.

    The new version is minted by :func:`domain.brief_ops.revise_brief`, so the
    prior (confirmed) brief is never mutated; the new version starts unconfirmed.
    """
    new_brief = revise_brief(prior, **edits)
    return propose_invalidation(
        old_brief=prior,
        new_brief=new_brief,
        runs=runs,
        findings=findings,
        detector=detector,
    )


# --- applying the (confirmed / overridden) set ------------------------------


def _redispatch(run: LensRun, new_version: int, id_generator: IdGenerator) -> LensRun:
    """A fresh PENDING LensRun for ``run``'s lens against the new brief version —
    invalidated runs re-dispatch, they are not edited in place (story 122). The
    dispatch path (PanelRouter.make_dispatch_event) assigns ``dispatch_event_id``."""
    return LensRun(
        id=new_lens_run_id(id_generator),
        lens_id=run.lens_id,
        session_id=run.session_id,
        brief_version=new_version,
        round=run.round,
    )


def apply_invalidation(
    *,
    proposal: InvalidationProposal,
    runs: Iterable[LensRun],
    id_generator: IdGenerator,
    rerun_ids: Collection[LensRunId] | None = None,
) -> RefinementResult:
    """Carry out the refinement against the new brief version.

    ``rerun_ids`` is the user's final set (the override, story 117). When ``None``
    the system's proposed set (:attr:`InvalidationProposal.proposed_rerun_ids`) is
    used as-is. Runs in the final set re-dispatch as fresh PENDING runs; every
    other run is re-pointed to the new version with ``reused_from_version`` set to
    the old version (story 121) — preserving its identity and findings rather than
    spending budget re-running it.
    """
    target = frozenset(rerun_ids) if rerun_ids is not None else proposal.proposed_rerun_ids
    new_version = proposal.new_brief.version

    reused: list[LensRun] = []
    redispatched: list[LensRun] = []
    for run in runs:
        if run.id in target:
            redispatched.append(_redispatch(run, new_version, id_generator))
        else:
            reused.append(
                run.model_copy(
                    update={
                        "brief_version": new_version,
                        "reused_from_version": proposal.old_brief_version,
                    }
                )
            )

    return RefinementResult(
        new_brief=proposal.new_brief,
        reused_runs=tuple(reused),
        redispatched_runs=tuple(redispatched),
    )
