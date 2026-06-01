"""Schema validation: valid payloads construct, invalid payloads are rejected,
and the lens-output envelope's code-enforced invariants hold (stories 44-45)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_council.enums import ChangeReason, ClaimType, Confidence
from research_council.ids import FindingId, SourceId
from research_council.models import (
    Finding,
    FindingDraft,
    LensRunOutputDraft,
    ScopedChallengeOutputDraft,
    SourceRef,
)


def _finding() -> Finding:
    return Finding(
        id=FindingId("finding_1"),
        claim_text="RAG baselines are excluded from the evaluation.",
        claim_type=ClaimType.GAP,
        confidence=Confidence.SUPPORTING,
        sources=(SourceRef(canonical_id=SourceId("source_1")),),
        failure_modes_if_wrong="If RAG is irrelevant here, the gap is moot.",
    )


def _draft(**overrides: object) -> FindingDraft:
    base: dict[str, object] = {
        "claim_text": "RAG baselines are excluded from the evaluation.",
        "claim_type": ClaimType.GAP,
        "confidence": Confidence.SUPPORTING,
        "failure_modes_if_wrong": "If RAG is irrelevant here, the gap is moot.",
    }
    base.update(overrides)
    return FindingDraft(**base)  # type: ignore[arg-type]


def test_valid_finding_draft_constructs() -> None:
    draft = _draft()
    assert draft.claim_type is ClaimType.GAP


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _draft(severity="high")


def test_bad_enum_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _draft(claim_type="speculative")


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FindingDraft(  # type: ignore[call-arg]
            claim_text="x",
            claim_type=ClaimType.GAP,
            confidence=Confidence.SUPPORTING,
        )


def test_lens_output_requires_a_self_disagreement() -> None:
    # story 45: zero entries in disagreements_with_my_own_framing is a violation.
    with pytest.raises(ValidationError):
        LensRunOutputDraft(
            findings=[_draft()],
            disagreements_with_my_own_framing=[],
            brief_summary="A brief summary.",
        )


def test_lens_output_valid_with_disagreement() -> None:
    out = LensRunOutputDraft(
        findings=[_draft()],
        disagreements_with_my_own_framing=["I might be over-weighting RAG."],
        brief_summary="A brief summary.",
    )
    assert out.refused is False


def test_refused_output_requires_reason() -> None:
    with pytest.raises(ValidationError):
        LensRunOutputDraft(refused=True)


def test_refused_output_skips_disagreement_requirement() -> None:
    out = LensRunOutputDraft(refused=True, refusal_reason="Out of my frame.")
    assert out.refused is True
    assert out.disagreements_with_my_own_framing == []


def test_brief_summary_word_limit_enforced() -> None:
    with pytest.raises(ValidationError):
        LensRunOutputDraft(
            findings=[_draft()],
            disagreements_with_my_own_framing=["one"],
            brief_summary=" ".join(["word"] * 151),
        )


def test_frozen_collection_cannot_be_mutated() -> None:
    # frozen=True blocks attribute reassignment; tuple fields block content
    # mutation too, so a persisted Finding is immutable all the way down.
    f = _finding()
    with pytest.raises(AttributeError):  # tuple has no .append
        f.sources.append(SourceRef(canonical_id=SourceId("source_2")))  # type: ignore[attr-defined]


def test_persisted_finding_is_hashable() -> None:
    # A model with a list field is unhashable; tuple fields make Finding usable
    # in a set / as a dict key (the clusterer hashes Findings).
    f = _finding()
    assert hash(f) == hash(f)
    assert f in {f}


def test_scoped_challenge_output_constructs() -> None:
    out = ScopedChallengeOutputDraft(
        finding=_draft(),
        change_reason=ChangeReason.REAFFIRMED_AGAINST_CHALLENGE,
        sibling_impact_flags=["This likely undermines my earlier point about G."],
    )
    assert out.change_reason is ChangeReason.REAFFIRMED_AGAINST_CHALLENGE
