"""The static nine-lens roster (stories 54-63)."""

from __future__ import annotations

from research_council.enums import ClaimType, LensId, ToolName
from research_council.lenses import ROSTER, get_lens_config


def test_roster_has_all_nine_lenses() -> None:
    assert set(ROSTER) == set(LensId)
    assert len(ROSTER) == 9


def test_each_config_id_matches_its_key() -> None:
    for lens_id, config in ROSTER.items():
        assert config.id is lens_id


def test_prior_art_has_both_tools() -> None:
    config = get_lens_config(LensId.PRIOR_ART)
    assert set(config.tool_access) == {ToolName.VERIFIER_QUERY, ToolName.SOURCE_FETCH}


def test_adversarial_has_only_source_fetch() -> None:
    assert get_lens_config(LensId.ADVERSARIAL).tool_access == [ToolName.SOURCE_FETCH]


def test_no_tool_lenses_have_empty_access() -> None:
    for lens_id in (
        LensId.MECHANISTIC_INTERPRETABILITY,
        LensId.INFORMATION_THEORETIC,
        LensId.FIRST_PRINCIPLES,
    ):
        assert get_lens_config(lens_id).tool_access == []


def test_first_principles_is_restricted_to_mechanism_and_gap() -> None:
    config = get_lens_config(LensId.FIRST_PRINCIPLES)
    assert config.allowed_claim_types == [ClaimType.MECHANISM_HYPOTHESIS, ClaimType.GAP]


def test_other_lenses_have_no_hard_claim_type_restriction() -> None:
    assert get_lens_config(LensId.PRIOR_ART).allowed_claim_types is None
