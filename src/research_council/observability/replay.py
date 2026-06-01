"""Partial replay (PRD story 165).

Manually re-dispatch a single LensRun or a single verifier check against a
*stored* brief version. The same stateless reconstruction the system does
normally, triggered manually for debugging. Full-session deterministic replay
is not achievable (LLM nondeterminism) and is deliberately NOT faked (PRD
out-of-scope item 174) — replay always re-invokes the model, the returned
outcome may differ from the original.

Reconstruction reads only from persisted pointers (brief by version, prior
runs by session/lens/version, peer Round 1 runs for cross-pollination) — no
state is held outside the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ids import FindingId, IdGenerator, LensRunId
from ..lenses import get_lens_config
from ..models import Finding, LensRun, Source, VerificationResult
from ..runtime import (
    DEFAULT_MODEL,
    AnonymizedPeer,
    LensRunInput,
    LensRunOutcome,
    LlmClient,
    PeerOutput,
    ToolRegistry,
    anonymize_peers,
    build_prior_self,
    run_lens,
)
from ..store.interface import SessionStore


@dataclass(frozen=True)
class ReplayLensRunInputs:
    """The reconstructed (LensRun, LensRunInput) pair ready for re-dispatch."""

    run: LensRun
    inputs: LensRunInput


def reconstruct_lens_run_inputs(
    store: SessionStore,
    run_id: LensRunId,
) -> ReplayLensRunInputs | None:
    """Rebuild the four contract inputs for ``run_id`` from persisted state.

    Returns ``None`` if the LensRun or its brief version is not in the store.
    Round 1 reconstructs with empty ``prior_self`` and empty
    ``cross_pollination`` (the locked Round 1 contract). Round 2 populates
    ``prior_self`` from this lens's Round 1 findings and ``cross_pollination``
    from peers' Round 1 outputs, anonymized in completion order — the same
    construction the dispatcher does normally."""
    run = store.get_lens_run(run_id)
    if run is None:
        return None
    brief = store.get_brief(run.session_id, run.brief_version)
    if brief is None:
        return None

    lens_config = get_lens_config(run.lens_id)
    effective_round = run.round if run.round is not None else 1

    prior_self_findings: list[Finding] = []
    cross_pollination: list[AnonymizedPeer] = []

    if effective_round == 2:
        prior_runs = [
            r
            for r in store.list_lens_runs_for_lens(run.session_id, run.lens_id, run.brief_version)
            if r.id != run.id and r.round == 1
        ]
        for prior in prior_runs:
            prior_self_findings.extend(store.list_findings_for_lens_run(prior.id))

        peer_round1_runs = [
            r
            for r in store.list_lens_runs_for_brief(run.session_id, run.brief_version, round=1)
            if r.lens_id != run.lens_id
        ]
        peers = [
            PeerOutput(
                lens_id=peer.lens_id,
                findings=store.list_findings_for_lens_run(peer.id),
                brief_summary=peer.brief_summary,
                open_questions=list(peer.open_questions),
            )
            for peer in peer_round1_runs
        ]
        cross_pollination = anonymize_peers(peers)

    inputs = LensRunInput(
        brief=brief,
        lens_config=lens_config,
        round=effective_round,
        prior_self=build_prior_self(prior_self_findings),
        cross_pollination=cross_pollination,
    )
    return ReplayLensRunInputs(run=run, inputs=inputs)


async def replay_lens_run(
    store: SessionStore,
    run_id: LensRunId,
    *,
    client: LlmClient,
    id_generator: IdGenerator,
    tools: ToolRegistry | None = None,
    model: str = DEFAULT_MODEL,
    caller_ref: str | None = None,
) -> LensRunOutcome | None:
    """Re-dispatch ``run_id`` against its stored brief version.

    Returns ``None`` if the run or its brief cannot be reconstructed; otherwise
    returns the (possibly different) outcome of the new call. The new outcome
    is NOT persisted — replay is a diagnostic, not a normal write path.

    ``caller_ref`` defaults to ``run_id`` so any TraceRecords emitted by a
    decorating ``TracingLlmClient`` are keyed to the same LensRun the replay
    targets."""
    reconstructed = reconstruct_lens_run_inputs(store, run_id)
    if reconstructed is None:
        return None
    return await run_lens(
        inputs=reconstructed.inputs,
        client=client,
        id_generator=id_generator,
        tools=tools,
        model=model,
        caller_ref=caller_ref or run_id,
    )


class Verifier(Protocol):
    """The ClaimVerifier seam (PRD module 3). The verifier owns the verdict
    (story 160) — partial replay just re-invokes it against the stored Finding
    and (if present) its primary source."""

    async def verify(
        self, finding: Finding, source: Source | None
    ) -> VerificationResult: ...


async def replay_verifier_check(
    store: SessionStore,
    finding_id: FindingId,
    *,
    verifier: Verifier,
) -> VerificationResult | None:
    """Re-invoke the verifier on a single Finding, using its primary stored
    source if any. Returns ``None`` if the Finding does not exist.

    The result is NOT persisted — replay is a diagnostic. A caller that wants
    to upgrade the historical verdict should save it through the normal
    SessionStore write path."""
    finding = store.get_finding(finding_id)
    if finding is None:
        return None
    source: Source | None = None
    if finding.sources:
        source = store.get_source(finding.sources[0].canonical_id)
    return await verifier.verify(finding, source)
