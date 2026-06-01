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


def test_build_system_prompt_states_mech_interp_restriction() -> None:
    # Story 58: the same code-enforced claim-type rule applies to mech-interp.
    config = get_lens_config(LensId.MECHANISTIC_INTERPRETABILITY)
    prompt = build_system_prompt(config, "deep_dive")
    assert "mechanism_hypothesis" in prompt
    assert "failure_mode" in prompt


def test_build_system_prompt_states_info_theoretic_prediction_rule() -> None:
    # Story 59: anti-decoration rule is surfaced in the prompt and enforced in code.
    config = get_lens_config(LensId.INFORMATION_THEORETIC)
    prompt = build_system_prompt(config, "deep_dive")
    assert "quantitative or conditional prediction" in prompt


def test_build_lens_prompt_contains_brief_and_not_master_reasoning(brief: Brief) -> None:
    inputs = LensRunInput(brief=brief, lens_config=get_lens_config(LensId.FIRST_PRINCIPLES))
    prompt = build_lens_prompt(inputs)
    assert brief.problem_statement in prompt
    # LensRunInput has no field through which master selection reasoning could enter.
    assert not hasattr(inputs, "master_reasoning")


# --- Round-aware adversarial system prompt (story 56) ----------------------


def test_adversarial_round_1_targets_proposed_solution() -> None:
    config = get_lens_config(LensId.ADVERSARIAL)
    prompt = build_system_prompt(config, "deep_dive", round=1)
    # Round 1: target the researcher's proposed solution.
    assert "proposed solution" in prompt.lower()
    assert "Round 1" in prompt
    # The Round 2 framing (emerging consensus) is NOT surfaced in Round 1.
    assert "emerging" not in prompt.lower()


def test_adversarial_round_2_targets_emerging_consensus() -> None:
    config = get_lens_config(LensId.ADVERSARIAL)
    prompt = build_system_prompt(config, "deep_dive", round=2)
    # Round 2: target the emerging Round 1 consensus.
    assert "emerging" in prompt.lower()
    assert "Round 2" in prompt
    # Round 1's "proposed solution" target should not be the operative instruction
    # at this round — it's been replaced by the consensus target.
    assert "proposed solution" not in prompt.lower()


def test_prior_art_round_2_references_emerging_analysis() -> None:
    # Story 95: Round 2 prior-art participates against the council's emerging analysis.
    config = get_lens_config(LensId.PRIOR_ART)
    prompt = build_system_prompt(config, "deep_dive", round=2)
    assert "emerging" in prompt.lower() or "council" in prompt.lower()
    assert "Round 2" in prompt
