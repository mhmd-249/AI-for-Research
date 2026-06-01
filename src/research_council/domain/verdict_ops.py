"""Verdict load + resolve (stories 124-129).

The Verdict loop in v0 is data-model-only: an engineer writes a JSON/YAML file
matching the schema (story 125) using loose ``(lens, claim_text excerpt)``
references (story 126); a resolver lazily binds each excerpt to a specific
``Finding`` within the session via fuzzy match (stories 127, 129), populating a
derived, cached ``resolved_finding_ids`` field on the reference.

The cache is the field itself: if a ref already carries resolved_finding_ids,
the resolver leaves it alone and does not re-run the match. That keeps the
resolver idempotent and lets a downstream eval pipeline trust the cache after
a single resolve pass.

Ambiguous (>1 match) and no-match excerpts are surfaced via diagnostics rather
than silently dropped — the engineer is expected to refine the excerpt until
each ref resolves unambiguously.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml

from ..enums import LensId
from ..ids import FindingId, IdGenerator, SessionId, new_verdict_id
from ..models import Finding, Verdict, VerdictDraft, VerdictRef
from ..store import SessionStore

# Threshold for the summed-matching-blocks / len(excerpt) fuzzy ratio. 0.8
# tolerates light typos and minor paraphrase while rejecting unrelated claims;
# the engineer can always refine the excerpt to disambiguate.
FUZZY_MATCH_THRESHOLD = 0.8

_WHITESPACE = re.compile(r"\s+")


class ResolutionStatus(StrEnum):
    """Outcome of resolving one VerdictRef's excerpt against the session."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class ResolutionDiagnostic:
    """Surfaces the outcome of a ref the resolver actively matched this pass.

    Emitted for every ref the resolver matched this pass — RESOLVED (the cache
    was just populated), AMBIGUOUS, and NO_MATCH alike — so the caller can audit
    each decision and refine ambiguous/no-match excerpts. Refs that arrived
    already carrying ``resolved_finding_ids`` are treated as cached, skipped
    without re-matching, and emit no diagnostic.
    """

    ref_kind: Literal["validated", "invalidated"]
    ref_index: int
    lens: LensId
    excerpt: str
    status: ResolutionStatus
    candidate_finding_ids: tuple[FindingId, ...]


@dataclass(frozen=True)
class ResolutionReport:
    verdict: Verdict
    diagnostics: tuple[ResolutionDiagnostic, ...]


# --- Loader -----------------------------------------------------------------


def load_verdict_draft(path: Path | str) -> VerdictDraft:
    """Load and schema-validate a hand-authored Verdict file (JSON or YAML).

    Invalid payloads raise ``pydantic.ValidationError``; unsupported file
    extensions and non-object roots raise ``ValueError``. The author writes
    excerpts only; ``id`` and ``resolved_finding_ids`` are rejected here so the
    schema cannot drift from the persisted-vs-authored split.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raise ValueError(
            f"unsupported verdict file extension {suffix!r}; expected .json, .yaml, or .yml"
        )

    if not isinstance(raw, dict):
        raise ValueError(
            f"verdict file must contain a JSON/YAML object at the root, got {type(raw).__name__}"
        )

    return VerdictDraft.model_validate(raw)


# --- Materialize ------------------------------------------------------------


def materialize_verdict(draft: VerdictDraft, gen: IdGenerator) -> Verdict:
    """Mint a Verdict id and lift the draft refs into persisted VerdictRefs.

    The resolver runs separately; a freshly-materialized Verdict has empty
    ``resolved_finding_ids`` on every ref.
    """
    validated = tuple(
        VerdictRef(lens=r.lens, claim_text_excerpt=r.claim_text_excerpt)
        for r in draft.predictions_validated
    )
    invalidated = tuple(
        VerdictRef(lens=r.lens, claim_text_excerpt=r.claim_text_excerpt)
        for r in draft.predictions_invalidated
    )
    return Verdict(
        id=new_verdict_id(gen),
        session_ref=draft.session_ref,
        decision=draft.decision,
        outcome_summary=draft.outcome_summary,
        predictions_validated=validated,
        predictions_invalidated=invalidated,
    )


# --- Resolver ---------------------------------------------------------------


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower().strip())


def _fuzzy_matches(excerpt: str, claim_text: str, threshold: float) -> bool:
    """True iff ``excerpt`` plausibly references ``claim_text``.

    Substring (after lowercase + whitespace collapse) is the cheap path. The
    fallback sums every matching block between excerpt and claim and divides
    by ``len(excerpt)`` — using the excerpt length (not the longer string)
    keeps short hand-authored excerpts from scoring near-zero against a longer
    claim, while summing blocks tolerates typos that would shrink any single
    longest run."""
    e = _normalize(excerpt)
    c = _normalize(claim_text)
    if not e:
        return False
    if e in c:
        return True
    matcher = difflib.SequenceMatcher(a=e, b=c, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (matched / len(e)) >= threshold


def _collect_findings_by_lens(
    store: SessionStore, session_id: SessionId
) -> dict[LensId, list[Finding]]:
    """Group every session finding by the lens that emitted it.

    Findings carry no lens_id; the only path from Finding to lens is the
    LensRun that references its id. We walk every brief version's runs once,
    deduping by FindingId since the same finding can appear on multiple runs
    (e.g., re-pointed via ``reused_from_version``)."""
    by_lens: dict[LensId, list[Finding]] = {}
    seen: set[FindingId] = set()
    for brief in store.list_brief_versions(session_id):
        for run in store.list_lens_runs_for_brief(brief.session_id, brief.version):
            for fid in run.finding_ids:
                if fid in seen:
                    continue
                seen.add(fid)
                finding = store.get_finding(fid)
                if finding is None:
                    continue
                by_lens.setdefault(run.lens_id, []).append(finding)
    return by_lens


def _resolve_refs(
    refs: tuple[VerdictRef, ...],
    kind: Literal["validated", "invalidated"],
    findings_by_lens: dict[LensId, list[Finding]],
    threshold: float,
) -> tuple[tuple[VerdictRef, ...], tuple[ResolutionDiagnostic, ...]]:
    resolved: list[VerdictRef] = []
    diagnostics: list[ResolutionDiagnostic] = []
    for i, ref in enumerate(refs):
        if ref.resolved_finding_ids:
            # Cache hit: leave the ref alone and emit no diagnostic.
            resolved.append(ref)
            continue
        candidates = findings_by_lens.get(ref.lens, [])
        matches = tuple(
            f.id
            for f in candidates
            if _fuzzy_matches(ref.claim_text_excerpt, f.claim_text, threshold)
        )
        if len(matches) == 1:
            status = ResolutionStatus.RESOLVED
            resolved.append(ref.model_copy(update={"resolved_finding_ids": matches}))
        elif len(matches) > 1:
            # Ambiguous and no-match refs keep an empty cache so the engineer
            # is forced to refine the excerpt; candidates are surfaced via the
            # diagnostic rather than silently dropped.
            status = ResolutionStatus.AMBIGUOUS
            resolved.append(ref)
        else:
            status = ResolutionStatus.NO_MATCH
            resolved.append(ref)
        diagnostics.append(
            ResolutionDiagnostic(
                ref_kind=kind,
                ref_index=i,
                lens=ref.lens,
                excerpt=ref.claim_text_excerpt,
                status=status,
                candidate_finding_ids=matches,
            )
        )
    return tuple(resolved), tuple(diagnostics)


def resolve_verdict(
    verdict: Verdict,
    store: SessionStore,
    *,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> ResolutionReport:
    """Populate ``resolved_finding_ids`` on each unambiguous VerdictRef.

    Refs already carrying resolved_finding_ids are treated as cached and not
    re-matched. Ambiguous and no-match refs are surfaced via diagnostics so
    the engineer can refine the excerpt; their cache stays empty.
    """
    findings_by_lens = _collect_findings_by_lens(store, verdict.session_ref)
    validated, val_diags = _resolve_refs(
        verdict.predictions_validated, "validated", findings_by_lens, threshold
    )
    invalidated, inv_diags = _resolve_refs(
        verdict.predictions_invalidated, "invalidated", findings_by_lens, threshold
    )
    resolved_verdict = verdict.model_copy(
        update={
            "predictions_validated": validated,
            "predictions_invalidated": invalidated,
        }
    )
    return ResolutionReport(
        verdict=resolved_verdict, diagnostics=val_diags + inv_diags
    )
