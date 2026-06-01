"""Persistence boundary. ``SessionStore`` is the contract every module codes
against; ``InMemorySessionStore`` is the v0 implementation. The SQLite
implementation (a later slice) must satisfy the same interface.
"""

from .interface import SessionStore, TraceSink
from .memory import InMemorySessionStore

__all__ = ["SessionStore", "TraceSink", "InMemorySessionStore"]
