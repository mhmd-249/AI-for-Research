"""In-memory ``SessionStore`` + ``TraceSink``.

Deep-copies every value on the way in and on the way out (``model_copy(deep=True)``)
so callers never hold a live reference into stored state — the same isolation a
real database gives. This is what lets tests written against this store stay valid
when the SQLite implementation lands.
"""

from __future__ import annotations

from ..domain.source_ops import canonical_id_for_source
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
    FrozenModel,
    LensRun,
    SelfUseLog,
    Session,
    Source,
    Synthesis,
    TraceRecord,
    Verdict,
    VerificationResult,
)


def _clone[M: FrozenModel](model: M) -> M:
    return model.model_copy(deep=True)


class InMemorySessionStore:
    """Dict-backed store. Satisfies ``SessionStore`` and ``TraceSink``."""

    def __init__(self) -> None:
        self._sessions: dict[SessionId, Session] = {}
        self._briefs: dict[tuple[SessionId, int], Brief] = {}
        self._dispatch_events: dict[DispatchEventId, DispatchEvent] = {}
        self._lens_runs: dict[LensRunId, LensRun] = {}
        self._findings: dict[FindingId, Finding] = {}
        self._sources: dict[SourceId, Source] = {}
        self._verifications: dict[str, VerificationResult] = {}
        self._challenges: dict[ChallengeId, Challenge] = {}
        self._syntheses: dict[str, Synthesis] = {}
        self._verdicts: dict[SessionId, Verdict] = {}
        self._self_use_logs: dict[SessionId, SelfUseLog] = {}
        self._traces: list[TraceRecord] = []

    # Session ---------------------------------------------------------------
    def save_session(self, session: Session) -> None:
        self._sessions[session.id] = _clone(session)

    def get_session(self, session_id: SessionId) -> Session | None:
        found = self._sessions.get(session_id)
        return _clone(found) if found else None

    # Brief -----------------------------------------------------------------
    def save_brief(self, brief: Brief) -> None:
        self._briefs[(brief.session_id, brief.version)] = _clone(brief)

    def get_brief(self, session_id: SessionId, version: int) -> Brief | None:
        found = self._briefs.get((session_id, version))
        return _clone(found) if found else None

    def get_latest_brief(self, session_id: SessionId) -> Brief | None:
        versions = [b for (sid, _), b in self._briefs.items() if sid == session_id]
        if not versions:
            return None
        return _clone(max(versions, key=lambda b: b.version))

    def list_brief_versions(self, session_id: SessionId) -> list[Brief]:
        versions = [b for (sid, _), b in self._briefs.items() if sid == session_id]
        return [_clone(b) for b in sorted(versions, key=lambda b: b.version)]

    # DispatchEvent ---------------------------------------------------------
    def save_dispatch_event(self, event: DispatchEvent) -> None:
        self._dispatch_events[event.id] = _clone(event)

    def get_dispatch_event(self, event_id: DispatchEventId) -> DispatchEvent | None:
        found = self._dispatch_events.get(event_id)
        return _clone(found) if found else None

    def list_dispatch_events(self, session_id: SessionId) -> list[DispatchEvent]:
        return [_clone(e) for e in self._dispatch_events.values() if e.session_id == session_id]

    # LensRun ---------------------------------------------------------------
    def save_lens_run(self, run: LensRun) -> None:
        self._lens_runs[run.id] = _clone(run)

    def get_lens_run(self, run_id: LensRunId) -> LensRun | None:
        found = self._lens_runs.get(run_id)
        return _clone(found) if found else None

    def list_lens_runs_for_brief(
        self, session_id: SessionId, version: int, round: Round | None = None
    ) -> list[LensRun]:
        return [
            _clone(r)
            for r in self._lens_runs.values()
            if r.session_id == session_id
            and r.brief_version == version
            and (round is None or r.round == round)
        ]

    def list_lens_runs_for_lens(
        self, session_id: SessionId, lens_id: LensId, version: int
    ) -> list[LensRun]:
        return [
            _clone(r)
            for r in self._lens_runs.values()
            if r.session_id == session_id and r.lens_id == lens_id and r.brief_version == version
        ]

    # Finding ---------------------------------------------------------------
    def save_finding(self, finding: Finding) -> None:
        self._findings[finding.id] = _clone(finding)

    def get_finding(self, finding_id: FindingId) -> Finding | None:
        found = self._findings.get(finding_id)
        return _clone(found) if found else None

    def list_findings_for_lens_run(self, run_id: LensRunId) -> list[Finding]:
        # Literal storage: returns findings as stored, including superseded ones.
        run = self._lens_runs.get(run_id)
        if run is None:
            return []
        return [_clone(self._findings[fid]) for fid in run.finding_ids if fid in self._findings]

    def list_findings_for_session(self, session_id: SessionId) -> list[Finding]:
        # Every finding in the session, INCLUDING superseded ones (literal storage).
        # Findings carry no session_id; they reach a session via a LensRun's
        # finding_ids, so we union the findings referenced by the session's runs.
        seen: set[FindingId] = set()
        result: list[Finding] = []
        for run in self._lens_runs.values():
            if run.session_id != session_id:
                continue
            for fid in run.finding_ids:
                if fid in seen or fid not in self._findings:
                    continue
                seen.add(fid)
                result.append(_clone(self._findings[fid]))
        return result

    # Source ----------------------------------------------------------------
    def upsert_source(self, source: Source) -> Source:
        canonical = canonical_id_for_source(source)
        existing = self._sources.get(canonical)
        if existing is not None:
            return _clone(existing)
        self._sources[canonical] = _clone(source)
        return _clone(source)

    def get_source(self, canonical_id: SourceId) -> Source | None:
        found = self._sources.get(canonical_id)
        return _clone(found) if found else None

    # VerificationResult ----------------------------------------------------
    def save_verification_result(self, result: VerificationResult) -> None:
        self._verifications[result.id] = _clone(result)

    def list_verification_results_for_finding(
        self, finding_id: FindingId
    ) -> list[VerificationResult]:
        return [_clone(v) for v in self._verifications.values() if v.finding_id == finding_id]

    # Challenge -------------------------------------------------------------
    def save_challenge(self, challenge: Challenge) -> None:
        self._challenges[challenge.id] = _clone(challenge)

    def get_challenge(self, challenge_id: ChallengeId) -> Challenge | None:
        found = self._challenges.get(challenge_id)
        return _clone(found) if found else None

    def list_challenges(self, session_id: SessionId) -> list[Challenge]:
        return [_clone(c) for c in self._challenges.values() if c.session_id == session_id]

    # Synthesis -------------------------------------------------------------
    def save_synthesis(self, synthesis: Synthesis) -> None:
        self._syntheses[synthesis.id] = _clone(synthesis)

    def get_latest_synthesis(self, session_id: SessionId) -> Synthesis | None:
        matching = [s for s in self._syntheses.values() if s.session_id == session_id]
        if not matching:
            return None
        return _clone(max(matching, key=lambda s: s.brief_version))

    # Verdict ---------------------------------------------------------------
    def save_verdict(self, verdict: Verdict) -> None:
        self._verdicts[verdict.session_ref] = _clone(verdict)

    def get_verdict(self, session_id: SessionId) -> Verdict | None:
        found = self._verdicts.get(session_id)
        return _clone(found) if found else None

    # SelfUseLog ------------------------------------------------------------
    def save_self_use_log(self, log: SelfUseLog) -> None:
        self._self_use_logs[log.session_ref] = _clone(log)

    def get_self_use_log(self, session_id: SessionId) -> SelfUseLog | None:
        found = self._self_use_logs.get(session_id)
        return _clone(found) if found else None

    # TraceSink -------------------------------------------------------------
    def append_trace(self, record: TraceRecord) -> None:
        self._traces.append(_clone(record))

    def list_traces(self, caller_ref: str) -> list[TraceRecord]:
        return [_clone(t) for t in self._traces if t.caller_ref == caller_ref]
