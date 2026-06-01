"""Brief immutability and versioning (stories 9, 76)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_council.domain.brief_ops import confirm_brief, revise_brief
from research_council.models import Brief


def test_confirm_sets_flag_without_bumping_version(brief: Brief) -> None:
    confirmed = confirm_brief(brief)
    assert confirmed.confirmed is True
    assert confirmed.version == brief.version


def test_revise_bumps_version_and_resets_confirmed(brief: Brief) -> None:
    confirmed = confirm_brief(brief)
    revised = revise_brief(confirmed, problem_statement="A sharper problem statement.")

    assert revised.version == confirmed.version + 1
    assert revised.confirmed is False
    assert revised.problem_statement == "A sharper problem statement."


def test_revise_never_mutates_the_prior_version(brief: Brief) -> None:
    confirmed = confirm_brief(brief)
    original_statement = confirmed.problem_statement

    revise_brief(confirmed, problem_statement="Edited.")

    # The confirmed brief is untouched: still confirmed, same version, same content.
    assert confirmed.confirmed is True
    assert confirmed.version == 1
    assert confirmed.problem_statement == original_statement


def test_frozen_brief_rejects_in_place_mutation(brief: Brief) -> None:
    with pytest.raises(ValidationError):
        brief.problem_statement = "mutated"  # type: ignore[misc]


def test_revise_rejects_editing_immutable_fields(brief: Brief) -> None:
    with pytest.raises(ValueError, match="immutable"):
        revise_brief(brief, version=99)
