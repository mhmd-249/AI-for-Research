"""Persistence boundary. ``SessionStore`` is the contract every module codes
against; ``InMemorySessionStore`` is the v0 reference implementation and
``SqliteSessionStore`` is the persistent one. Both satisfy the same interface,
so the in-memory test suite passes unchanged against either."""

from .interface import SessionStore, SessionStoreAndTrace, TraceSink
from .memory import InMemorySessionStore
from .sqlite import SqliteSessionStore

__all__ = [
    "InMemorySessionStore",
    "SessionStore",
    "SessionStoreAndTrace",
    "SqliteSessionStore",
    "TraceSink",
]
