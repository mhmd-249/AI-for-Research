"""Input construction enforces the sub-agent contract in code (stories 42-49, 97-98)."""

from __future__ import annotations

from research_council.enums import (
    ClaimType,
    Confidence,
    LensId,
    VerificationStatus,
)
from research_council.ids import SequentialIdGenerator, new_finding_id
from research_council.lenses import get_lens_config
from research_council.models import Brief, Finding
from research_council.runtime import (
    LensRunInput,
    PeerOutput,
    anonymize_peers,
    build_lens_prompt,
    build_prior_self,
    build_system_prompt,
)


def _finding(ids: SequentialIdGenerator, text: str, status: VerificationStatus) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=ClaimType.MECHANISM_HYPOTHESIS,
        confidence=Confidence.SUPPORTING,
        failure_modes_if_wrong="...",
        verification_status=status,
        round=1,
    )


def test_build_prior_self_strips_verifier_results(ids: SequentialIdGenerator) -> None:
    verified = _finding(ids, "selective gating suffices", VerificationStatus.VERIFIED)
    projected = build_prior_self([verified])
    assert projected[0].verification_status is VerificationStatus.UNVERIFIED
    assert projected[0].claim_text == verified.claim_text


def test_anonymize_peers_removes_identity_and_labels_in_order(
    ids: SequentialIdGenerator,
) -> None:
    peers = [
        PeerOutput(
            lens_id=LensId.PRIOR_ART,
            findings=[_finding(ids, "close prior work exists", VerificationStatus.VERIFIED)],
            brief_summary="grounding summary",
            open_questions=["which baseline?"],
        ),
        PeerOutput(
            lens_id=LensId.ADVERSARIAL,
            findings=[_finding(ids, "the core assumption breaks", VerificationStatus.UNVERIFIED)],
            brief_summary="attack summary",
            open_questions=[],
        ),
    ]
    anon = anonymize_peers(peers)

    assert [p.label for p in anon] == ["Lens A", "Lens B"]
    # The real lens identities are literally absent from the anonymized structure.
    assert "prior_art" not in repr(anon)
    assert "adversarial" not in repr(anon)


def test_build_system_prompt_states_first_principles_restriction() -> None:
    config = get_lens_config(LensId.FIRST_PRINCIPLES)
    prompt = build_system_prompt(config, "deep_dive")
    assert "mechanism_hypothesis" in prompt
    assert "gap" in prompt
    assert "at least one entry in disagreements" in prompt


def test_build_lens_prompt_contains_brief_and_not_master_reasoning(brief: Brief) -> None:
    inputs = LensRunInput(brief=brief, lens_config=get_lens_config(LensId.FIRST_PRINCIPLES))
    prompt = build_lens_prompt(inputs)
    assert brief.problem_statement in prompt
    # LensRunInput has no field through which master selection reasoning could enter.
    assert not hasattr(inputs, "master_reasoning")
