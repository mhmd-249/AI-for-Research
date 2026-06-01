"""SQLite-backed ``SessionStore`` + ``TraceSink``.

Persists every domain object as a JSON blob keyed by its id (with secondary
columns for the few query patterns the protocol exposes). JSON round-trip via
Pydantic preserves the typed shape — frozen models, tuple-typed collections,
enums, ``datetime`` — so a value read back is ``==`` to what was saved, the
isolation the in-memory store gets from ``model_copy(deep=True)``.

The Sources table is shared across the entire database, deduplicated by
``canonical_id`` (PRD story 13 / 145). Between-rounds resume (PRD story 146):
reopening the store at the same path reloads the session at a round boundary —
the persistence layer stores no mid-round state because the controllers never
write any.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS briefs (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, version)
);

CREATE TABLE IF NOT EXISTS dispatch_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dispatch_session ON dispatch_events(session_id);

CREATE TABLE IF NOT EXISTS lens_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    lens_id TEXT NOT NULL,
    brief_version INTEGER NOT NULL,
    round INTEGER,
    reused_from_version INTEGER,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_brief ON lens_runs(session_id, brief_version, round);
CREATE INDEX IF NOT EXISTS idx_runs_lens ON lens_runs(session_id, lens_id, brief_version);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    supersedes TEXT,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    canonical_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_results (
    id TEXT PRIMARY KEY,
    finding_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vr_finding ON verification_results(finding_id);

CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_challenges_session ON challenges(session_id);

CREATE TABLE IF NOT EXISTS syntheses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    brief_version INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_synth_session ON syntheses(session_id);

CREATE TABLE IF NOT EXISTS verdicts (
    session_ref TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS self_use_logs (
    session_ref TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    caller_ref TEXT NOT NULL,
    rowid_order INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_caller ON traces(caller_ref, rowid_order);
"""

def _dump(model: FrozenModel) -> str:
    return model.model_dump_json()


def _load[M: FrozenModel](cls: type[M], payload: str) -> M:
    return cls.model_validate_json(payload)


class SqliteSessionStore:
    """File-backed store. Satisfies ``SessionStore`` and ``TraceSink``.

    A single embedded SQLite database (PRD story 145). Reopening the store at the
    same path reloads everything previously committed — the persistence half of
    between-rounds resume (PRD story 146)."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            # WAL gives concurrent readers + one writer without locking the
            # whole DB; harmless for the single-writer v0 controller too.
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._trace_counter = self._read_trace_high_water()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteSessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_trace_high_water(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(rowid_order), 0) FROM traces").fetchone()
        return int(row[0]) if row else 0

    # Session ---------------------------------------------------------------
    def save_session(self, session: Session) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (id, status, payload) VALUES (?, ?, ?)",
            (session.id, session.status.value, _dump(session)),
        )

    def get_session(self, session_id: SessionId) -> Session | None:
        row = self._conn.execute(
            "SELECT payload FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _load(Session, row[0]) if row else None

    # Brief -----------------------------------------------------------------
    def save_brief(self, brief: Brief) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO briefs (session_id, version, payload) VALUES (?, ?, ?)",
            (brief.session_id, brief.version, _dump(brief)),
        )

    def get_brief(self, session_id: SessionId, version: int) -> Brief | None:
        row = self._conn.execute(
            "SELECT payload FROM briefs WHERE session_id = ? AND version = ?",
            (session_id, version),
        ).fetchone()
        return _load(Brief, row[0]) if row else None

    def get_latest_brief(self, session_id: SessionId) -> Brief | None:
        row = self._conn.execute(
            "SELECT payload FROM briefs WHERE session_id = ? ORDER BY version DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return _load(Brief, row[0]) if row else None

    def list_brief_versions(self, session_id: SessionId) -> list[Brief]:
        rows = self._conn.execute(
            "SELECT payload FROM briefs WHERE session_id = ? ORDER BY version ASC",
            (session_id,),
        ).fetchall()
        return [_load(Brief, r[0]) for r in rows]

    # DispatchEvent ---------------------------------------------------------
    def save_dispatch_event(self, event: DispatchEvent) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO dispatch_events (id, session_id, payload) VALUES (?, ?, ?)",
            (event.id, event.session_id, _dump(event)),
        )

    def get_dispatch_event(self, event_id: DispatchEventId) -> DispatchEvent | None:
        row = self._conn.execute(
            "SELECT payload FROM dispatch_events WHERE id = ?", (event_id,)
        ).fetchone()
        return _load(DispatchEvent, row[0]) if row else None

    def list_dispatch_events(self, session_id: SessionId) -> list[DispatchEvent]:
        rows = self._conn.execute(
            "SELECT payload FROM dispatch_events WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [_load(DispatchEvent, r[0]) for r in rows]

    # LensRun ---------------------------------------------------------------
    def save_lens_run(self, run: LensRun) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO lens_runs "
            "(id, session_id, lens_id, brief_version, round, reused_from_version, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.session_id,
                run.lens_id.value,
                run.brief_version,
                run.round,
                run.reused_from_version,
                _dump(run),
            ),
        )

    def get_lens_run(self, run_id: LensRunId) -> LensRun | None:
        row = self._conn.execute(
            "SELECT payload FROM lens_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _load(LensRun, row[0]) if row else None

    def list_lens_runs_for_brief(
        self, session_id: SessionId, version: int, round: Round | None = None
    ) -> list[LensRun]:
        if round is None:
            rows = self._conn.execute(
                "SELECT payload FROM lens_runs WHERE session_id = ? AND brief_version = ?",
                (session_id, version),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM lens_runs "
                "WHERE session_id = ? AND brief_version = ? AND round = ?",
                (session_id, version, round),
            ).fetchall()
        return [_load(LensRun, r[0]) for r in rows]

    def list_lens_runs_for_lens(
        self, session_id: SessionId, lens_id: LensId, version: int
    ) -> list[LensRun]:
        rows = self._conn.execute(
            "SELECT payload FROM lens_runs "
            "WHERE session_id = ? AND lens_id = ? AND brief_version = ?",
            (session_id, lens_id.value, version),
        ).fetchall()
        return [_load(LensRun, r[0]) for r in rows]

    # Finding ---------------------------------------------------------------
    def save_finding(self, finding: Finding) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO findings (id, supersedes, payload) VALUES (?, ?, ?)",
            (finding.id, finding.supersedes, _dump(finding)),
        )

    def get_finding(self, finding_id: FindingId) -> Finding | None:
        row = self._conn.execute(
            "SELECT payload FROM findings WHERE id = ?", (finding_id,)
        ).fetchone()
        return _load(Finding, row[0]) if row else None

    def list_findings_for_lens_run(self, run_id: LensRunId) -> list[Finding]:
        run = self.get_lens_run(run_id)
        if run is None:
            return []
        return [f for fid in run.finding_ids if (f := self.get_finding(fid)) is not None]

    def list_findings_for_session(self, session_id: SessionId) -> list[Finding]:
        # Findings carry no session_id; they reach a session through a LensRun's
        # finding_ids, so we union the findings referenced by the session's runs.
        rows = self._conn.execute(
            "SELECT payload FROM lens_runs WHERE session_id = ?", (session_id,)
        ).fetchall()
        seen: set[FindingId] = set()
        result: list[Finding] = []
        for (payload,) in rows:
            run = _load(LensRun, payload)
            for fid in run.finding_ids:
                if fid in seen:
                    continue
                seen.add(fid)
                finding = self.get_finding(fid)
                if finding is not None:
                    result.append(finding)
        return result

    # Source (global, deduplicated) -----------------------------------------
    def upsert_source(self, source: Source) -> Source:
        canonical = canonical_id_for_source(source)
        existing = self.get_source(canonical)
        if existing is not None:
            return existing  # first-writer wins; dup deduped to the existing source
        self._conn.execute(
            "INSERT INTO sources (canonical_id, payload) VALUES (?, ?)",
            (canonical, _dump(source)),
        )
        return source

    def get_source(self, canonical_id: SourceId) -> Source | None:
        row = self._conn.execute(
            "SELECT payload FROM sources WHERE canonical_id = ?", (canonical_id,)
        ).fetchone()
        return _load(Source, row[0]) if row else None

    # VerificationResult ----------------------------------------------------
    def save_verification_result(self, result: VerificationResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO verification_results (id, finding_id, payload) "
            "VALUES (?, ?, ?)",
            (result.id, result.finding_id, _dump(result)),
        )

    def list_verification_results_for_finding(
        self, finding_id: FindingId
    ) -> list[VerificationResult]:
        rows = self._conn.execute(
            "SELECT payload FROM verification_results WHERE finding_id = ?",
            (finding_id,),
        ).fetchall()
        return [_load(VerificationResult, r[0]) for r in rows]

    # Challenge -------------------------------------------------------------
    def save_challenge(self, challenge: Challenge) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO challenges (id, session_id, payload) VALUES (?, ?, ?)",
            (challenge.id, challenge.session_id, _dump(challenge)),
        )

    def get_challenge(self, challenge_id: ChallengeId) -> Challenge | None:
        row = self._conn.execute(
            "SELECT payload FROM challenges WHERE id = ?", (challenge_id,)
        ).fetchone()
        return _load(Challenge, row[0]) if row else None

    def list_challenges(self, session_id: SessionId) -> list[Challenge]:
        rows = self._conn.execute(
            "SELECT payload FROM challenges WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [_load(Challenge, r[0]) for r in rows]

    # Synthesis -------------------------------------------------------------
    def save_synthesis(self, synthesis: Synthesis) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO syntheses (id, session_id, brief_version, payload) "
            "VALUES (?, ?, ?, ?)",
            (synthesis.id, synthesis.session_id, synthesis.brief_version, _dump(synthesis)),
        )

    def get_latest_synthesis(self, session_id: SessionId) -> Synthesis | None:
        row = self._conn.execute(
            "SELECT payload FROM syntheses WHERE session_id = ? "
            "ORDER BY brief_version DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return _load(Synthesis, row[0]) if row else None

    # Verdict ---------------------------------------------------------------
    def save_verdict(self, verdict: Verdict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO verdicts (session_ref, payload) VALUES (?, ?)",
            (verdict.session_ref, _dump(verdict)),
        )

    def get_verdict(self, session_id: SessionId) -> Verdict | None:
        row = self._conn.execute(
            "SELECT payload FROM verdicts WHERE session_ref = ?", (session_id,)
        ).fetchone()
        return _load(Verdict, row[0]) if row else None

    # SelfUseLog ------------------------------------------------------------
    def save_self_use_log(self, log: SelfUseLog) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO self_use_logs (session_ref, payload) VALUES (?, ?)",
            (log.session_ref, _dump(log)),
        )

    def get_self_use_log(self, session_id: SessionId) -> SelfUseLog | None:
        row = self._conn.execute(
            "SELECT payload FROM self_use_logs WHERE session_ref = ?", (session_id,)
        ).fetchone()
        return _load(SelfUseLog, row[0]) if row else None

    # TraceSink -------------------------------------------------------------
    def append_trace(self, record: TraceRecord) -> None:
        self._trace_counter += 1
        self._conn.execute(
            "INSERT OR REPLACE INTO traces (id, caller_ref, rowid_order, payload) "
            "VALUES (?, ?, ?, ?)",
            (record.id, record.caller_ref, self._trace_counter, _dump(record)),
        )

    def list_traces(self, caller_ref: str) -> list[TraceRecord]:
        rows = self._conn.execute(
            "SELECT payload FROM traces WHERE caller_ref = ? ORDER BY rowid_order ASC",
            (caller_ref,),
        ).fetchall()
        return [_load(TraceRecord, r[0]) for r in rows]
