"""Backward pointer-graph walk for synthesis claims (PRD story 162).

Reconstructs the full chain that produced a Finding: the LensRun that emitted
it, the brief version that LensRun ran against, the verifier's checks on the
claim, the sources those checks resolved, and the raw LLM I/O captured at
dispatch time. Everything is read off persisted pointers — no separate
logging.

A "synthesis claim" is the user-facing handle for a cluster / tension / gap in
a :class:`~research_council.models.Synthesis`; each carries ``finding_refs``
pointing at the constituent Findings. The walk is therefore parameterized by
:class:`FindingId` (single) or a list of them (cluster).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..ids import FindingId
from ..models import (
    Brief,
    Finding,
    LensRun,
    Source,
    TraceRecord,
    VerificationResult,
)
from ..store.interface import SessionStore, TraceSink


@dataclass(frozen=True)
class FindingTrace:
    """The full backward chain for one Finding."""

    finding: Finding
    emitting_run: LensRun | None  # None for orphan Findings (e.g. brief background claims)
    brief: Brief | None  # the brief_version the emitting run ran against
    verifications: list[VerificationResult]
    sources: list[Source]  # sources referenced by the Finding and its verifications
    traces: list[TraceRecord]  # raw I/O records keyed to the emitting LensRun


def _gather_sources(
    store: SessionStore,
    finding: Finding,
    verifications: list[VerificationResult],
) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []
    canonical_ids = [ref.canonical_id for ref in finding.sources]
    for verification in verifications:
        if verification.source_canonical_id is not None:
            canonical_ids.append(verification.source_canonical_id)
    for canonical in canonical_ids:
        if canonical in seen:
            continue
        seen.add(canonical)
        source = store.get_source(canonical)
        if source is not None:
            sources.append(source)
    return sources


def walk_finding(
    store: SessionStore,
    traces: TraceSink,
    finding_id: FindingId,
) -> FindingTrace | None:
    """Reconstruct the backward chain for a single Finding.

    Returns ``None`` if the Finding does not exist. A Finding that exists but
    has no emitting LensRun (e.g. a background claim attached to a Brief) is
    returned with ``emitting_run=None`` and empty trace records — the chain is
    truncated, not absent."""
    finding = store.get_finding(finding_id)
    if finding is None:
        return None

    emitting_run = store.find_lens_run_by_finding(finding_id)
    brief: Brief | None = None
    trace_records: list[TraceRecord] = []
    if emitting_run is not None:
        brief = store.get_brief(emitting_run.session_id, emitting_run.brief_version)
        trace_records = traces.list_traces(emitting_run.id)

    verifications = store.list_verification_results_for_finding(finding_id)
    sources = _gather_sources(store, finding, verifications)

    return FindingTrace(
        finding=finding,
        emitting_run=emitting_run,
        brief=brief,
        verifications=verifications,
        sources=sources,
        traces=trace_records,
    )


def walk_findings(
    store: SessionStore,
    traces: TraceSink,
    finding_ids: Iterable[FindingId],
) -> list[FindingTrace]:
    """Walk many Findings. Missing ids are dropped (not represented as ``None``)
    so the caller can iterate the result directly without filtering."""
    out: list[FindingTrace] = []
    for fid in finding_ids:
        chain = walk_finding(store, traces, fid)
        if chain is not None:
            out.append(chain)
    return out


def walk_synthesis_claim(
    store: SessionStore,
    traces: TraceSink,
    finding_refs: Iterable[FindingId],
) -> list[FindingTrace]:
    """The PRD's "synthesis claim → constituent Findings → ..." walk. A
    synthesis claim's handle is its ``finding_refs`` tuple; this just walks
    each constituent Finding."""
    return walk_findings(store, traces, finding_refs)
