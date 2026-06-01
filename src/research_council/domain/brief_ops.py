"""Brief versioning and immutability (stories 9, 76).

A Brief is an immutable versioned snapshot. ``revise_brief`` produces the next
version and never touches the prior one; combined with the store's deep-copy on
read/write, a confirmed brief can never be mutated in place — an edit always
yields a new version.
"""

from __future__ import annotations

from typing import Any

from ..models import Brief

# Fields a revision may change. Identity/lineage fields are off-limits.
_IMMUTABLE_FIELDS = frozenset({"session_id", "version"})


def revise_brief(prior: Brief, **edits: Any) -> Brief:
    """Return the next version of ``prior`` with ``edits`` applied.

    The new version is ``prior.version + 1`` and starts unconfirmed (a revision is
    a fresh draft until re-confirmed). ``prior`` is returned unchanged.
    """
    illegal = _IMMUTABLE_FIELDS.intersection(edits)
    if illegal:
        raise ValueError(f"cannot edit immutable brief fields: {sorted(illegal)}")

    return prior.model_copy(update={**edits, "version": prior.version + 1, "confirmed": False})


def confirm_brief(brief: Brief) -> Brief:
    """Mark a draft brief confirmed without bumping its version.

    Post-confirmation the snapshot is treated as immutable; further edits go
    through :func:`revise_brief`, which produces a new version.
    """
    if brief.confirmed:
        return brief
    return brief.model_copy(update={"confirmed": True})
