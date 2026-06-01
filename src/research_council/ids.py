"""Branded identifier types and the injectable id generator.

Every persisted object carries a type-prefixed string id (e.g. ``finding_a1b2``).
The prefix makes the pointer graph readable when walking it backward for
observability (PRD story 162); ``NewType`` makes mypy treat the ids nominally,
so a ``SessionId`` cannot be passed where a ``FindingId`` is expected.

Ids are minted through an injectable :class:`IdGenerator` so tests can swap in a
deterministic counter instead of random tokens.
"""

from __future__ import annotations

from typing import NewType, Protocol
from uuid import uuid4

# --- Branded id types -------------------------------------------------------
# Generated ids (minted by an IdGenerator):
SessionId = NewType("SessionId", str)
LensRunId = NewType("LensRunId", str)
FindingId = NewType("FindingId", str)
DispatchEventId = NewType("DispatchEventId", str)
ChallengeId = NewType("ChallengeId", str)
SynthesisId = NewType("SynthesisId", str)
VerdictId = NewType("VerdictId", str)
VerificationResultId = NewType("VerificationResultId", str)
TraceRecordId = NewType("TraceRecordId", str)

# Computed id (derived from a source's identifying fields, never minted):
SourceId = NewType("SourceId", str)


class IdGenerator(Protocol):
    """Mints a fresh id with the given type prefix, e.g. ``("finding") -> 'finding_x7k2'``."""

    def __call__(self, prefix: str) -> str: ...


class UuidGenerator:
    """Production generator: ``{prefix}_{12 hex chars}``."""

    def __call__(self, prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"


class SequentialIdGenerator:
    """Deterministic generator for tests: ``{prefix}_0001``, ``{prefix}_0002``, ..."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        nxt = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = nxt
        return f"{prefix}_{nxt:04d}"


# --- Typed constructors -----------------------------------------------------
# Thin wrappers that fix the prefix per entity so call sites stay consistent.


def new_session_id(gen: IdGenerator) -> SessionId:
    return SessionId(gen("session"))


def new_lens_run_id(gen: IdGenerator) -> LensRunId:
    return LensRunId(gen("lensrun"))


def new_finding_id(gen: IdGenerator) -> FindingId:
    return FindingId(gen("finding"))


def new_dispatch_event_id(gen: IdGenerator) -> DispatchEventId:
    return DispatchEventId(gen("dispatch"))


def new_challenge_id(gen: IdGenerator) -> ChallengeId:
    return ChallengeId(gen("challenge"))


def new_synthesis_id(gen: IdGenerator) -> SynthesisId:
    return SynthesisId(gen("synthesis"))


def new_verdict_id(gen: IdGenerator) -> VerdictId:
    return VerdictId(gen("verdict"))


def new_verification_result_id(gen: IdGenerator) -> VerificationResultId:
    return VerificationResultId(gen("verification"))


def new_trace_record_id(gen: IdGenerator) -> TraceRecordId:
    return TraceRecordId(gen("trace"))
