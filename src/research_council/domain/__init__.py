"""Pure domain operations that enforce the core invariants: source dedup
identity, brief versioning/immutability, and the session status machine. These
live outside the store so the SQLite implementation (a later slice) stays a pure
persistence layer.
"""
