"""EvalReader (module #14, PRD Layer 6.1, stories 149-155).

There is **no separate eval subsystem**: eval is the Verdict loop plus a read
over already-persisted data. This module is that read. It computes the two v0
metrics and supports the one manual comparison the PRD keeps:

* **verification-fail-rate** — the *guardrail* metric and the locked abandon
  trigger (story 152; abandon trigger (a), >10% verification fail → kill). A
  pure read over Findings' ``verification_status``.
* **gap-recall on Verdict-labeled sessions** — the *value* metric: of the gaps
  later confirmed real via a Verdict, what fraction did the council surface
  before commitment (story 152). Computed from the resolver's cached
  ``resolved_finding_ids`` (a label that resolved to a Finding *is* a gap the
  council surfaced) so it is "fed for free by the Verdict loop"
  (``VerdictResolver -> EvalReader``).
* **golden set** — begins with the ECF proposal as entry #1 and its 5 known
  gaps as the labeled answer key (story 149), and grows by one labeled case per
  Verdict-producing dogfooding session (story 150) — the golden set and the
  Verdict loop are the same thing viewed twice.
* **lens-ablation comparison** — manual and rare (story 151): re-run a brief
  with lens X removed and see whether the gaps lens X was responsible for
  disappear. Supported by the DispatchEvent model (story 17) that distinguishes
  panels on the same brief; *automating* ablation is out of scope (story 170).

Pure functions over typed models / the store, matching :mod:`.iteration_cap`
and :mod:`.verdict_ops`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..enums import ClaimType, LensId, VerificationStatus
from ..ids import SessionId
from ..models import DispatchEvent, Finding, Verdict, VerdictRef
from ..store import SessionStore
from ..synthesis.synthesizer import resolve_winning_findings

# --- verification-fail-rate (guardrail metric, stories 3, 152) --------------

# The abandon trigger is *strictly* greater than 10% (story 3 / trigger (a)):
# exactly 10% does not kill.
ABANDON_FAIL_RATE_THRESHOLD = 0.10

# A "fail" is a conclusive check that went against the citation: the source
# contradicts the claim ("don't say what was claimed") or the citation does not
# resolve ("don't exist / can't be located") — story 3's literal three modes.
# ``partially_supported`` is deliberately excluded: story 30 keeps it separate
# from ``contradicted`` as "sloppy, not wrong", so the kill-switch is not tripped
# by overstated-but-relevant citations. Exposed as a constant so the boundary is
# auditable and tunable on dogfooding data.
FAIL_STATUSES: frozenset[VerificationStatus] = frozenset(
    {VerificationStatus.CONTRADICTED, VerificationStatus.SOURCE_NOT_FOUND}
)

# The denominator: only conclusive checks count. ``unverified`` (not run yet),
# ``unverifiable_by_design`` (not a failure state, story 29), and
# ``verifier_error`` (the verifier couldn't run — never silently "verified",
# story 37) are excluded entirely; counting them would let infra noise move the
# guardrail.
EVALUATED_STATUSES: frozenset[VerificationStatus] = frozenset(
    {
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_SUPPORTED,
        VerificationStatus.CONTRADICTED,
        VerificationStatus.SOURCE_NOT_FOUND,
    }
)


@dataclass(frozen=True)
class VerificationFailRate:
    """The guardrail metric over a set of Findings.

    ``rate`` is ``fail_count / evaluated_count``, or ``0.0`` when nothing was
    conclusively checked (an empty read is not a failure)."""

    fail_count: int
    evaluated_count: int

    @property
    def rate(self) -> float:
        if self.evaluated_count == 0:
            return 0.0
        return self.fail_count / self.evaluated_count

    @property
    def exceeds_abandon_threshold(self) -> bool:
        return self.rate > ABANDON_FAIL_RATE_THRESHOLD


def verification_fail_rate(
    findings: Iterable[Finding],
    *,
    fail_statuses: frozenset[VerificationStatus] = FAIL_STATUSES,
    evaluated_statuses: frozenset[VerificationStatus] = EVALUATED_STATUSES,
) -> VerificationFailRate:
    """Compute the verification-fail-rate over ``findings`` (story 152)."""
    evaluated = [f for f in findings if f.verification_status in evaluated_statuses]
    fails = sum(1 for f in evaluated if f.verification_status in fail_statuses)
    return VerificationFailRate(fail_count=fails, evaluated_count=len(evaluated))


def verification_fail_rate_for_session(
    store: SessionStore, session_id: SessionId
) -> VerificationFailRate:
    """The guardrail metric over a session's *winning* Findings (story 111):
    superseded / withdrawn claims are not what the researcher reads, so they do
    not move the credibility guardrail."""
    winners = resolve_winning_findings(store.list_findings_for_session(session_id))
    return verification_fail_rate(winners)


# --- gap-recall (value metric, story 152) -----------------------------------


@dataclass(frozen=True)
class GapRecall:
    """The value metric: of the confirmed-real gap *labels*, what fraction the
    council surfaced (resolved to a Finding) before commitment."""

    surfaced_count: int
    label_count: int
    missed_excerpts: tuple[str, ...]

    @property
    def recall(self) -> float:
        if self.label_count == 0:
            return 0.0
        return self.surfaced_count / self.label_count


def gap_recall(labels: Iterable[VerdictRef]) -> GapRecall:
    """Compute gap-recall from resolved Verdict references (story 152).

    A label that resolved to at least one Finding (non-empty
    ``resolved_finding_ids``) is a gap the council surfaced; a label that never
    resolved is a confirmed-real gap the council missed — it counts against
    recall and is reported in ``missed_excerpts`` so the miss is visible, never
    silently dropped."""
    labels = tuple(labels)
    missed = tuple(ref.claim_text_excerpt for ref in labels if not ref.resolved_finding_ids)
    return GapRecall(
        surfaced_count=len(labels) - len(missed),
        label_count=len(labels),
        missed_excerpts=missed,
    )


def gap_recall_for_verdict(verdict: Verdict) -> GapRecall:
    """Gap-recall over a Verdict's ``predictions_validated`` — the gaps later
    confirmed real (story 152). ``predictions_invalidated`` are claims that
    turned out wrong, not confirmed-real gaps, so they are not recall labels."""
    return gap_recall(verdict.predictions_validated)


# --- golden set (stories 149, 150) ------------------------------------------

# The ECF proposal's 5 known gaps (Layers 1-3 abandon-trigger (d)), each paired
# with the lens most likely to emit it. The pairing is the seed answer key; it is
# refined against the first real ECF run (the lens that actually surfaces each
# gap may differ — gap #1's circularity, notably, lives in the tension *between*
# lenses). Excerpts are short so the resolver's fuzzy match binds them to
# whatever phrasing the council uses.
ECF_GOLDEN_CASE_NAME = "ECF — Active Context Inhibition"

ECF_GOLDEN_GAPS: tuple[tuple[LensId, str], ...] = (
    (LensId.FIRST_PRINCIPLES, "filter agent circularity problem"),
    (LensId.PRIOR_ART, "baselines exclude RAG and context-compression methods"),
    (LensId.MECHANISTIC_INTERPRETABILITY, "PFC analogy is rhetorical not mechanistic"),
    (LensId.ADVERSARIAL, "citation auditing risk: claims may not be quotable"),
    (LensId.EMPIRICAL_BENCHMARKING, "agent vs static long-context conflation"),
)


@dataclass(frozen=True)
class GoldenCase:
    """One labeled case in the golden set: a name, the session it was run in (or
    ``None`` for the still-unrun ECF seed), and the gap labels (the answer key).

    ``gap_labels`` are :class:`VerdictRef`s so a case grown from a Verdict reuses
    the resolver's cached ``resolved_finding_ids`` directly — the golden set and
    the Verdict loop are the same thing viewed twice (story 150)."""

    name: str
    session_ref: SessionId | None
    gap_labels: tuple[VerdictRef, ...]


def seed_golden_set() -> tuple[GoldenCase, ...]:
    """The golden set's first entry: the ECF case with its 5 known gaps as the
    labeled answer key (story 149). The seed is unresolved (no run bound to it
    yet), so :func:`gap_recall_for_case` reports recall ``0.0`` until the real
    ECF run resolves the labels."""
    ecf = GoldenCase(
        name=ECF_GOLDEN_CASE_NAME,
        session_ref=None,
        gap_labels=tuple(
            VerdictRef(lens=lens, claim_text_excerpt=excerpt)
            for lens, excerpt in ECF_GOLDEN_GAPS
        ),
    )
    return (ecf,)


def golden_case_from_verdict(verdict: Verdict, *, name: str | None = None) -> GoldenCase:
    """Convert a Verdict-producing session into a golden case (story 150): the
    validated predictions *are* the labels."""
    return GoldenCase(
        name=name or f"session {verdict.session_ref}",
        session_ref=verdict.session_ref,
        gap_labels=verdict.predictions_validated,
    )


def append_golden_case(
    golden_set: tuple[GoldenCase, ...],
    verdict: Verdict,
    *,
    name: str | None = None,
) -> tuple[GoldenCase, ...]:
    """Grow the golden set by one labeled case per Verdict (story 150). Pure —
    returns a new tuple; the input is untouched."""
    return (*golden_set, golden_case_from_verdict(verdict, name=name))


def gap_recall_for_case(case: GoldenCase) -> GapRecall:
    """Gap-recall for one golden case, over its gap labels."""
    return gap_recall(case.gap_labels)


# --- lens-ablation comparison (story 151) -----------------------------------

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower().strip())


@dataclass(frozen=True)
class AblationComparison:
    """The result of comparing the gaps two panels surfaced on the same brief.

    Gaps are compared by normalized claim text (each DispatchEvent mints its own
    Finding ids, so id-equality cannot detect "the same gap"). The gaps that
    appear only in the full panel are the ones attributable to the removed lens —
    they disappeared when it was cut (the cognitive-science-lens-cut logic
    generalized, story 151)."""

    removed_lens: LensId
    gaps_only_in_full: tuple[str, ...]
    gaps_in_both: tuple[str, ...]
    gaps_only_in_ablated: tuple[str, ...]

    @property
    def attributable_to_removed_lens(self) -> tuple[str, ...]:
        return self.gaps_only_in_full


def _gap_claim_texts(store: SessionStore, event: DispatchEvent) -> dict[str, str]:
    """Normalized -> original claim_text for the GAP-type winning Findings the
    event's runs surfaced. Winners only (story 111): a withdrawn/superseded gap
    is not something the panel still stands behind."""
    findings: list[Finding] = []
    for run_id in event.lens_run_ids:
        run = store.get_lens_run(run_id)
        if run is None:
            continue
        for fid in run.finding_ids:
            f = store.get_finding(fid)
            if f is not None and f.claim_type is ClaimType.GAP:
                findings.append(f)
    return {
        _normalize(f.claim_text): f.claim_text
        for f in resolve_winning_findings(findings)
    }


def compare_lens_ablation(
    store: SessionStore,
    *,
    full_event: DispatchEvent,
    ablated_event: DispatchEvent,
    removed_lens: LensId,
) -> AblationComparison:
    """Compare the gaps surfaced by two DispatchEvents on the same brief, one
    with ``removed_lens`` cut from the panel (story 151).

    Both events must target the same (session, brief version) — comparing gaps
    across different briefs is meaningless. Raises ``ValueError`` otherwise."""
    if (full_event.session_id, full_event.brief_version) != (
        ablated_event.session_id,
        ablated_event.brief_version,
    ):
        raise ValueError(
            "lens-ablation comparison requires both DispatchEvents on the same brief "
            f"(got {full_event.session_id} v{full_event.brief_version} vs "
            f"{ablated_event.session_id} v{ablated_event.brief_version})"
        )

    full = _gap_claim_texts(store, full_event)
    ablated = _gap_claim_texts(store, ablated_event)
    only_full = tuple(v for k, v in full.items() if k not in ablated)
    in_both = tuple(v for k, v in full.items() if k in ablated)
    only_ablated = tuple(v for k, v in ablated.items() if k not in full)
    return AblationComparison(
        removed_lens=removed_lens,
        gaps_only_in_full=only_full,
        gaps_in_both=in_both,
        gaps_only_in_ablated=only_ablated,
    )
