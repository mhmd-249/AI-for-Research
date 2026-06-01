"""Shared fixtures and tiny builders for valid domain objects."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_council.domain.source_ops import canonical_source_id
from research_council.enums import (
    ClaimType,
    Confidence,
    Mode,
    SessionStatus,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_finding_id,
    new_session_id,
)
from research_council.models import (
    Brief,
    Finding,
    Session,
    Source,
)
from research_council.store import (
    InMemorySessionStore,
    SessionStoreAndTrace,
    SqliteSessionStore,
)


@pytest.fixture
def ids() -> SequentialIdGenerator:
    return SequentialIdGenerator()


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SessionStoreAndTrace]:
    # Parametrizing the store fixture is how the in-memory test suite passes
    # unchanged against SQLite (PRD story 145, issue #13 acceptance).
    if request.param == "memory":
        yield InMemorySessionStore()
        return
    sqlite_store = SqliteSessionStore(tmp_path / "store.db")
    try:
        yield sqlite_store
    finally:
        sqlite_store.close()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def session(ids: SequentialIdGenerator) -> Session:
    return Session(id=new_session_id(ids), status=SessionStatus.INTAKE, title="ECF")


@pytest.fixture
def finding(ids: SequentialIdGenerator) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text="The filter agent has a circularity problem.",
        claim_type=ClaimType.GAP,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="If the filter is cheap, the circularity argument weakens.",
        verification_status=VerificationStatus.UNVERIFIED,
        round=1,
    )


@pytest.fixture
def brief(session: Session) -> Brief:
    return Brief(
        session_id=session.id,
        version=1,
        mode=Mode.DEEP_DIVE,
        problem_statement="Long-context models under-attend to mid-context information.",
        researcher_context="AI engineer researching open ML problems.",
        success_criteria_for_deliberation="Surface gaps I haven't already found myself.",
        scope_and_non_scope="In scope: architecture. Out of scope: training from scratch.",
        proposed_solution="Active Context Inhibition via a filter agent.",
    )


@pytest.fixture
def source() -> Source:
    return Source(
        canonical_id=canonical_source_id(arxiv_id="2307.03172"),
        title="Lost in the Middle: How Language Models Use Long Contexts",
        authors=("Liu", "Lin"),
        year=2023,
        arxiv_id="2307.03172",
    )
