"""The ``SessionStore`` and ``TraceSink`` protocols.

Explicit, typed, per-entity methods (rather than a stringly-typed generic store)
so the access patterns the later modules need are discoverable and the SQLite
implementation has a concrete contract to satisfy. Query methods are shaped by
the PRD's access patterns: prior_self (runs for one lens on one brief), synthesis
(findings for a brief's runs), the global Source cache, etc.
"""

from __future__ import annotations

from typing import Protocol

from ..enums import LensId, Round
from ..ids import (
    ChallengeId,
    DispatchEventId,
    FindingId,
    LensRunId,
    SessionId,
    SourceId,
)
from ..models import (
    Brief,
    Challenge,
    DispatchEvent,
    Finding,
    LensRun,
    SelfUseLog,
    Session,
    Source,
    Synthesis,
    TraceRecord,
    Verdict,
    VerificationResult,
)


class SessionStore(Protocol):
    """Persistence for every domain object. Implementations must round-trip values
    (a returned object equals what was saved) without aliasing stored state."""

    # Session ---------------------------------------------------------------
    def save_session(self, session: Session) -> None: ...
    def get_session(self, session_id: SessionId) -> Session | None: ...

    # Brief (versioned) -----------------------------------------------------
    def save_brief(self, brief: Brief) -> None: ...
    def get_brief(self, session_id: SessionId, version: int) -> Brief | None: ...
    def get_latest_brief(self, session_id: SessionId) -> Brief | None: ...
    def list_brief_versions(self, session_id: SessionId) -> list[Brief]: ...

    # DispatchEvent ---------------------------------------------------------
    def save_dispatch_event(self, event: DispatchEvent) -> None: ...
    def get_dispatch_event(self, event_id: DispatchEventId) -> DispatchEvent | None: ...
    def list_dispatch_events(self, session_id: SessionId) -> list[DispatchEvent]: ...

    # LensRun ---------------------------------------------------------------
    def save_lens_run(self, run: LensRun) -> None: ...
    def get_lens_run(self, run_id: LensRunId) -> LensRun | None: ...
    def list_lens_runs_for_brief(
        self, session_id: SessionId, version: int, round: Round | None = None
    ) -> list[LensRun]: ...
    def list_lens_runs_for_lens(
        self, session_id: SessionId, lens_id: LensId, version: int
    ) -> list[LensRun]: ...

    # Finding ---------------------------------------------------------------
    def save_finding(self, finding: Finding) -> None: ...
    def get_finding(self, finding_id: FindingId) -> Finding | None:
        """Returns findings as stored, including superseded ones; the store does
        not resolve supersedes chains."""
        ...

    def list_findings_for_lens_run(self, run_id: LensRunId) -> list[Finding]:
        """Returns findings as stored, including superseded ones; the store does
        not resolve supersedes chains."""
        ...

    def list_findings_for_session(self, session_id: SessionId) -> list[Finding]:
        """All findings in the session, INCLUDING superseded ones (literal storage).

        Audit / Verdict mining / trace-walk read this. Winners-resolution
        (excluding superseded per story 111) is a domain-layer concern, not the
        store's."""
        ...

    # Source (global, deduplicated cache) -----------------------------------
    def upsert_source(self, source: Source) -> Source: ...
    def get_source(self, canonical_id: SourceId) -> Source | None: ...

    # VerificationResult ----------------------------------------------------
    def save_verification_result(self, result: VerificationResult) -> None: ...
    def list_verification_results_for_finding(
        self, finding_id: FindingId
    ) -> list[VerificationResult]: ...

    # Challenge -------------------------------------------------------------
    def save_challenge(self, challenge: Challenge) -> None: ...
    def get_challenge(self, challenge_id: ChallengeId) -> Challenge | None: ...
    def list_challenges(self, session_id: SessionId) -> list[Challenge]: ...

    # Synthesis -------------------------------------------------------------
    def save_synthesis(self, synthesis: Synthesis) -> None: ...
    def get_latest_synthesis(self, session_id: SessionId) -> Synthesis | None: ...

    # Verdict ---------------------------------------------------------------
    def save_verdict(self, verdict: Verdict) -> None: ...
    def get_verdict(self, session_id: SessionId) -> Verdict | None: ...

    # SelfUseLog ------------------------------------------------------------
    def save_self_use_log(self, log: SelfUseLog) -> None: ...
    def get_self_use_log(self, session_id: SessionId) -> SelfUseLog | None: ...


class TraceSink(Protocol):
    """Append-only raw-I/O trace capture (story 163). A later slice adds the
    backward-walk trace view and partial replay on top of this."""

    def append_trace(self, record: TraceRecord) -> None: ...
    def list_traces(self, caller_ref: str) -> list[TraceRecord]: ...


class SessionStoreAndTrace(SessionStore, TraceSink, Protocol):
    """Convenience protocol for stores that satisfy both ``SessionStore`` and
    ``TraceSink`` (both v0 implementations do; tests use this where the parameter
    needs to accept either store implementation interchangeably)."""
