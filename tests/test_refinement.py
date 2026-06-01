"""RefinementOrchestrator — brief-edit refinement loop (Layer 4.2, stories 117-123).

User-declares-scope / system-proposes — the PanelRouter Screen 2 pattern applied
to brief edits. Tests cover the five acceptance criteria:

- A brief edit produces a new version; the confirmed brief is never mutated (118).
- The material-edit rule classifies lenses correctly: a touched load_bearing /
  supporting finding's referenced content → re-run; exploratory / untouched →
  reuse (120).
- Per-lens reasoning is produced for both re-run and reuse decisions (119).
- The user can override the proposed invalidation set (117).
- Reused runs are re-pointed with ``reused_from_version``; invalidated runs
  re-dispatch as fresh runs against the new version (121, 122).

Fixture-testable with no store and no real LLM — pure functions over typed
Briefs + LensRuns + Findings, mirroring the Synthesizer / ClaimClusterer style.
"""

from __future__ import annotations

from research_council.domain.brief_ops import confirm_brief
from research_council.enums import (
    ClaimType,
    Confidence,
    LensId,
    LensRunStatus,
)
from research_council.ids import (
    FindingId,
    SequentialIdGenerator,
    new_finding_id,
    new_lens_run_id,
)
from research_council.models import Brief, Finding, LensRun
from research_council.refinement import (
    KeywordOverlapDetector,
    apply_invalidation,
    diff_briefs,
    propose_invalidation,
    refine_brief,
)

# --- builders ---------------------------------------------------------------


def _finding(
    ids: SequentialIdGenerator,
    text: str,
    *,
    confidence: Confidence = Confidence.LOAD_BEARING,
    claim_type: ClaimType = ClaimType.MECHANISM_HYPOTHESIS,
) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=text,
        claim_type=claim_type,
        confidence=confidence,
        failure_modes_if_wrong="...",
        round=1,
    )


def _run(
    brief: Brief,
    *,
    ids: SequentialIdGenerator,
    lens_id: LensId,
    finding_ids: tuple[FindingId, ...] = (),
    status: LensRunStatus = LensRunStatus.SUCCEEDED,
) -> LensRun:
    return LensRun(
        id=new_lens_run_id(ids),
        lens_id=lens_id,
        session_id=brief.session_id,
        brief_version=brief.version,
        round=1,
        status=status,
        finding_ids=finding_ids,
    )


# --- AC1: edit produces a new version, never mutates the confirmed brief -----


def test_brief_edit_produces_new_version_without_mutating_confirmed(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    confirmed = confirm_brief(brief)
    assert confirmed.confirmed is True

    proposal = refine_brief(
        confirmed,
        edits={"proposed_solution": "A learned router that gates context windows."},
        runs=[],
        findings=[],
    )

    # The prior confirmed brief is untouched.
    assert confirmed.version == 1
    assert confirmed.confirmed is True
    assert confirmed.proposed_solution == "Active Context Inhibition via a filter agent."

    # The edit yields a fresh, unconfirmed next version.
    assert proposal.new_brief.version == 2
    assert proposal.new_brief.confirmed is False
    assert proposal.new_brief.proposed_solution == "A learned router that gates context windows."
    assert proposal.old_brief_version == 1


# --- AC: brief diff ---------------------------------------------------------


def test_diff_briefs_detects_only_changed_content_fields(brief: Brief) -> None:
    edited = brief.model_copy(update={"version": 2, "proposed_solution": "A learned router."})
    diff = diff_briefs(brief, edited)

    assert len(diff) == 1
    (change,) = diff
    assert change.field == "proposed_solution"
    assert change.old_value == "Active Context Inhibition via a filter agent."
    assert change.new_value == "A learned router."


def test_propose_invalidation_is_pure_over_old_and_new_briefs(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # The store-free core: hand it two brief versions directly (no revise_brief).
    new = brief.model_copy(
        update={"version": 2, "proposed_solution": "A learned router that gates windows."}
    )
    f = _finding(ids, "The filter agent introduces a circularity.")
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = propose_invalidation(
        old_brief=brief, new_brief=new, runs=[run], findings=[f]
    )
    assert proposal.old_brief_version == 1
    assert proposal.new_brief.version == 2
    assert proposal.proposed_rerun_ids == frozenset({run.id})


def test_diff_briefs_empty_when_only_noncontent_fields_change(brief: Brief) -> None:
    # A panel_constraints edit is not brief *content* a finding can reference.
    edited = brief.model_copy(
        update={"version": 2, "panel_constraints": (LensId.PRIOR_ART,)}
    )
    assert diff_briefs(brief, edited) == ()


# --- AC2/AC3: material-edit rule + per-lens reasoning ------------------------


def test_load_bearing_reference_triggers_rerun_with_reasoning(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # prior_art leaned load-bearing on the OLD filter-agent solution.
    f = _finding(ids, "The filter agent introduces a circularity in attention.")
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )

    (lp,) = proposal.lens_proposals
    assert lp.disposition == "rerun"
    assert lp.lens_run_id == run.id
    assert "proposed_solution" in lp.triggering_fields
    assert lp.reasoning.strip()
    assert "re-run" in lp.reasoning
    assert run.id in proposal.proposed_rerun_ids


def test_supporting_reference_also_triggers_rerun(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    f = _finding(
        ids,
        "The filter agent's selective attention has prior art.",
        confidence=Confidence.SUPPORTING,
    )
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )
    (lp,) = proposal.lens_proposals
    assert lp.disposition == "rerun"


def test_exploratory_reference_defaults_to_reuse(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # Same referencing text, but only exploratory confidence → reuse (story 120).
    f = _finding(
        ids,
        "The filter agent might interact oddly with caching.",
        confidence=Confidence.EXPLORATORY,
    )
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )
    (lp,) = proposal.lens_proposals
    assert lp.disposition == "reuse"
    assert lp.triggering_fields == ()
    assert lp.reasoning.strip()
    assert "reuse" in lp.reasoning


def test_untouched_lens_reuses(brief: Brief, ids: SequentialIdGenerator) -> None:
    # information-theoretic's finding does not reference the solution at all.
    f = _finding(ids, "Compression bounds limit recoverable mid-sequence detail.")
    run = _run(
        brief, ids=ids, lens_id=LensId.INFORMATION_THEORETIC, finding_ids=(f.id,)
    )

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )
    (lp,) = proposal.lens_proposals
    assert lp.disposition == "reuse"


def test_no_content_change_reuses_all(brief: Brief, ids: SequentialIdGenerator) -> None:
    f = _finding(ids, "The filter agent introduces a circularity.")
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = refine_brief(
        brief,
        edits={"panel_constraints": (LensId.PRIOR_ART,)},
        runs=[run],
        findings=[f],
    )
    (lp,) = proposal.lens_proposals
    assert lp.disposition == "reuse"
    assert proposal.proposed_rerun_ids == frozenset()


def test_reasoning_produced_for_both_rerun_and_reuse(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # Story 119: one-line reasoning per affected lens, both re-run AND reuse.
    rerun_f = _finding(ids, "The filter agent has a circularity problem.")
    reuse_f = _finding(ids, "Benchmarks measure recall, not precision.")
    rerun_run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(rerun_f.id,))
    reuse_run = _run(
        brief, ids=ids, lens_id=LensId.EMPIRICAL_BENCHMARKING, finding_ids=(reuse_f.id,)
    )

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[rerun_run, reuse_run],
        findings=[rerun_f, reuse_f],
    )

    by_run = {lp.lens_run_id: lp for lp in proposal.lens_proposals}
    assert by_run[rerun_run.id].disposition == "rerun"
    assert by_run[reuse_run.id].disposition == "reuse"
    # Both carry non-empty, decision-specific reasoning.
    assert all(lp.reasoning.strip() for lp in proposal.lens_proposals)


# --- AC5: re-point reused runs / re-dispatch invalidated runs ----------------


def test_reused_runs_repointed_with_reused_from_version(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    f = _finding(ids, "Benchmarks measure recall, not precision.")
    run = _run(
        brief, ids=ids, lens_id=LensId.EMPIRICAL_BENCHMARKING, finding_ids=(f.id,)
    )

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )
    result = apply_invalidation(proposal=proposal, runs=[run], id_generator=ids)

    assert len(result.reused_runs) == 1
    assert result.redispatched_runs == ()
    reused = result.reused_runs[0]
    # Same identity and findings — re-pointed, NOT re-executed.
    assert reused.id == run.id
    assert reused.finding_ids == run.finding_ids
    assert reused.brief_version == proposal.new_brief.version
    assert reused.reused_from_version == 1


def test_invalidated_runs_redispatch_as_fresh_pending_runs(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    f = _finding(ids, "The filter agent introduces a circularity.")
    run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(f.id,))

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[run],
        findings=[f],
    )
    result = apply_invalidation(proposal=proposal, runs=[run], id_generator=ids)

    assert result.reused_runs == ()
    assert len(result.redispatched_runs) == 1
    fresh = result.redispatched_runs[0]
    assert fresh.id != run.id  # a NEW run
    assert fresh.lens_id == run.lens_id
    assert fresh.round == run.round
    assert fresh.brief_version == proposal.new_brief.version
    assert fresh.status is LensRunStatus.PENDING
    assert fresh.finding_ids == ()
    assert fresh.reused_from_version is None


# --- AC4: user override of the proposed set ---------------------------------


def test_user_can_override_proposed_set(
    brief: Brief, ids: SequentialIdGenerator
) -> None:
    # System proposes: prior_art → re-run, benchmarking → reuse.
    rerun_f = _finding(ids, "The filter agent has a circularity problem.")
    reuse_f = _finding(ids, "Benchmarks measure recall, not precision.")
    rerun_run = _run(brief, ids=ids, lens_id=LensId.PRIOR_ART, finding_ids=(rerun_f.id,))
    reuse_run = _run(
        brief, ids=ids, lens_id=LensId.EMPIRICAL_BENCHMARKING, finding_ids=(reuse_f.id,)
    )

    proposal = refine_brief(
        brief,
        edits={"proposed_solution": "A learned router that gates windows."},
        runs=[rerun_run, reuse_run],
        findings=[rerun_f, reuse_f],
    )
    assert proposal.proposed_rerun_ids == frozenset({rerun_run.id})

    # User overrides: keep prior_art (reuse it) and instead re-run benchmarking.
    result = apply_invalidation(
        proposal=proposal,
        runs=[rerun_run, reuse_run],
        id_generator=ids,
        rerun_ids={reuse_run.id},
    )

    reused_ids = {r.id for r in result.reused_runs}
    redispatched_lenses = {r.lens_id for r in result.redispatched_runs}
    assert rerun_run.id in reused_ids  # overridden back to reuse
    assert LensId.EMPIRICAL_BENCHMARKING in redispatched_lenses  # overridden to re-run


# --- detector unit ----------------------------------------------------------


def test_keyword_overlap_detector_ignores_stopwords_and_short_tokens() -> None:
    detector = KeywordOverlapDetector()
    # Only shared tokens are stopwords / short — no real reference.
    assert not detector.references("The model is a thing.", "This is the other one.")
    # A shared distinctive term is a reference.
    assert detector.references("The filter agent loops.", "A filter over context.")
