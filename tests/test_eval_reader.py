"""EvalReader: the two v0 metrics (verification-fail-rate guardrail and
gap-recall value) plus the golden set and manual lens-ablation comparison
(stories 149-155; PRD Layer 6.1, module #14)."""

from __future__ import annotations

from datetime import datetime

import pytest

from research_council.domain.eval_reader import (
    ABANDON_FAIL_RATE_THRESHOLD,
    ECF_GOLDEN_GAPS,
    GoldenCase,
    append_golden_case,
    compare_lens_ablation,
    gap_recall,
    gap_recall_for_case,
    gap_recall_for_verdict,
    golden_case_from_verdict,
    seed_golden_set,
    verification_fail_rate,
)
from research_council.enums import (
    ClaimType,
    Confidence,
    Decision,
    LensId,
    LensRunStatus,
    VerificationStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_dispatch_event_id,
    new_finding_id,
    new_lens_run_id,
    new_verdict_id,
)
from research_council.models import (
    Brief,
    DispatchEvent,
    Finding,
    LensRun,
    Session,
    Verdict,
    VerdictRef,
)
from research_council.store import InMemorySessionStore

# --- helpers ----------------------------------------------------------------


def _finding(
    ids: SequentialIdGenerator,
    *,
    claim_text: str,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    claim_type: ClaimType = ClaimType.EMPIRICAL,
) -> Finding:
    return Finding(
        id=new_finding_id(ids),
        claim_text=claim_text,
        claim_type=claim_type,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="x",
        verification_status=status,
        round=1,
    )


# --- verification-fail-rate (guardrail metric, story 152) -------------------


def test_fail_rate_zero_when_all_verified(ids: SequentialIdGenerator) -> None:
    findings = [
        _finding(ids, claim_text="a", status=VerificationStatus.VERIFIED),
        _finding(ids, claim_text="b", status=VerificationStatus.VERIFIED),
    ]
    report = verification_fail_rate(findings)
    assert report.fail_count == 0
    assert report.evaluated_count == 2
    assert report.rate == 0.0
    assert report.exceeds_abandon_threshold is False


def test_fail_rate_counts_contradicted_and_source_not_found(
    ids: SequentialIdGenerator,
) -> None:
    findings = [
        _finding(ids, claim_text="a", status=VerificationStatus.VERIFIED),
        _finding(ids, claim_text="b", status=VerificationStatus.CONTRADICTED),
        _finding(ids, claim_text="c", status=VerificationStatus.SOURCE_NOT_FOUND),
        _finding(ids, claim_text="d", status=VerificationStatus.VERIFIED),
    ]
    report = verification_fail_rate(findings)
    assert report.fail_count == 2
    assert report.evaluated_count == 4
    assert report.rate == 0.5


def test_fail_rate_excludes_non_evaluated_statuses_from_denominator(
    ids: SequentialIdGenerator,
) -> None:
    # unverified (not run yet), unverifiable_by_design (not a failure state), and
    # verifier_error (verifier couldn't run, never silently 'verified') are not
    # conclusive checks and are excluded from the rate entirely (stories 29, 37).
    findings = [
        _finding(ids, claim_text="a", status=VerificationStatus.VERIFIED),
        _finding(ids, claim_text="b", status=VerificationStatus.CONTRADICTED),
        _finding(ids, claim_text="c", status=VerificationStatus.UNVERIFIED),
        _finding(ids, claim_text="d", status=VerificationStatus.UNVERIFIABLE_BY_DESIGN),
        _finding(ids, claim_text="e", status=VerificationStatus.VERIFIER_ERROR),
    ]
    report = verification_fail_rate(findings)
    # only a (verified) + b (contradicted) are conclusive checks.
    assert report.evaluated_count == 2
    assert report.fail_count == 1
    assert report.rate == 0.5


def test_fail_rate_empty_is_zero_not_division_error() -> None:
    report = verification_fail_rate([])
    assert report.evaluated_count == 0
    assert report.rate == 0.0
    assert report.exceeds_abandon_threshold is False


def test_fail_rate_threshold_is_strict_over_ten_percent(
    ids: SequentialIdGenerator,
) -> None:
    # The abandon trigger is >10% (story 3): exactly 10% does NOT trip it.
    ten_pct = [
        _finding(ids, claim_text=f"v{i}", status=VerificationStatus.VERIFIED)
        for i in range(9)
    ] + [_finding(ids, claim_text="bad", status=VerificationStatus.CONTRADICTED)]
    report = verification_fail_rate(ten_pct)
    assert report.rate == pytest.approx(0.1)
    assert report.exceeds_abandon_threshold is False

    over = ten_pct + [_finding(ids, claim_text="bad2", status=VerificationStatus.CONTRADICTED)]
    report2 = verification_fail_rate(over)
    assert report2.rate > ABANDON_FAIL_RATE_THRESHOLD
    assert report2.exceeds_abandon_threshold is True


# --- gap-recall (value metric, story 152) -----------------------------------


def test_gap_recall_counts_resolved_refs_as_surfaced(
    ids: SequentialIdGenerator,
) -> None:
    surfaced = VerdictRef(
        lens=LensId.FIRST_PRINCIPLES,
        claim_text_excerpt="circularity",
        resolved_finding_ids=(new_finding_id(ids),),
    )
    missed = VerdictRef(
        lens=LensId.PRIOR_ART,
        claim_text_excerpt="baselines exclude RAG",
    )
    report = gap_recall([surfaced, missed])
    assert report.surfaced_count == 1
    assert report.label_count == 2
    assert report.recall == 0.5
    assert report.missed_excerpts == ("baselines exclude RAG",)


def test_gap_recall_empty_labels_is_zero() -> None:
    report = gap_recall([])
    assert report.label_count == 0
    assert report.recall == 0.0
    assert report.missed_excerpts == ()


def test_gap_recall_for_verdict_uses_validated_predictions(
    ids: SequentialIdGenerator, session: Session
) -> None:
    verdict = Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.PURSUED_MODIFIED,
        outcome_summary="x",
        predictions_validated=(
            VerdictRef(
                lens=LensId.FIRST_PRINCIPLES,
                claim_text_excerpt="circularity",
                resolved_finding_ids=(new_finding_id(ids),),
            ),
            VerdictRef(lens=LensId.PRIOR_ART, claim_text_excerpt="missed gap"),
        ),
        # invalidated predictions are NOT part of gap-recall: those are claims the
        # council made that turned out wrong, not gaps later confirmed real.
        predictions_invalidated=(
            VerdictRef(
                lens=LensId.ADVERSARIAL,
                claim_text_excerpt="brittle",
                resolved_finding_ids=(new_finding_id(ids),),
            ),
        ),
    )
    report = gap_recall_for_verdict(verdict)
    assert report.label_count == 2
    assert report.surfaced_count == 1
    assert report.recall == 0.5


# --- golden set (stories 149, 150) ------------------------------------------


def test_seed_golden_set_has_ecf_case_with_five_gaps() -> None:
    golden = seed_golden_set()
    assert len(golden) == 1
    ecf = golden[0]
    assert isinstance(ecf, GoldenCase)
    assert len(ecf.gap_labels) == 5
    assert len(ECF_GOLDEN_GAPS) == 5
    # The seed is an answer key only: no run resolved it yet, so recall is 0.
    assert gap_recall_for_case(ecf).recall == 0.0


def test_golden_set_grows_one_case_per_verdict(
    ids: SequentialIdGenerator, session: Session
) -> None:
    golden = seed_golden_set()
    verdict = Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.PURSUED_MODIFIED,
        outcome_summary="x",
        predictions_validated=(
            VerdictRef(
                lens=LensId.FIRST_PRINCIPLES,
                claim_text_excerpt="circularity",
                resolved_finding_ids=(new_finding_id(ids),),
            ),
        ),
    )
    grown = append_golden_case(golden, verdict)
    assert len(grown) == len(golden) + 1
    # original tuple is untouched (pure append).
    assert len(golden) == 1
    new_case = grown[-1]
    assert new_case.session_ref == session.id
    assert gap_recall_for_case(new_case).recall == 1.0


def test_golden_case_from_verdict_carries_validated_labels(
    ids: SequentialIdGenerator, session: Session
) -> None:
    verdict = Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.ABANDONED,
        outcome_summary="x",
        predictions_validated=(
            VerdictRef(lens=LensId.PRIOR_ART, claim_text_excerpt="g1"),
            VerdictRef(lens=LensId.ADVERSARIAL, claim_text_excerpt="g2"),
        ),
    )
    case = golden_case_from_verdict(verdict)
    assert case.session_ref == session.id
    assert tuple(r.claim_text_excerpt for r in case.gap_labels) == ("g1", "g2")


# --- lens-ablation comparison (story 151) -----------------------------------


def _dispatch_with_gaps(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    brief: Brief,
    *,
    panel: tuple[LensId, ...],
    gaps_by_lens: dict[LensId, list[str]],
    now: datetime,
) -> DispatchEvent:
    run_ids = []
    for lens_id in panel:
        finding_ids = []
        for text in gaps_by_lens.get(lens_id, []):
            f = _finding(ids, claim_text=text, claim_type=ClaimType.GAP)
            store.save_finding(f)
            finding_ids.append(f.id)
        run = LensRun(
            id=new_lens_run_id(ids),
            lens_id=lens_id,
            session_id=session.id,
            brief_version=brief.version,
            round=1,
            status=LensRunStatus.SUCCEEDED,
            finding_ids=tuple(finding_ids),
        )
        store.save_lens_run(run)
        run_ids.append(run.id)
    event = DispatchEvent(
        id=new_dispatch_event_id(ids),
        session_id=session.id,
        brief_version=brief.version,
        panel=panel,
        timestamp=now,
        lens_run_ids=tuple(run_ids),
    )
    store.save_dispatch_event(event)
    return event


def test_ablation_surfaces_gaps_that_disappear_when_lens_removed(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    brief: Brief,
    now: datetime,
) -> None:
    store.save_brief(brief)
    full = _dispatch_with_gaps(
        store,
        ids,
        session,
        brief,
        panel=(LensId.PRIOR_ART, LensId.ADVERSARIAL),
        gaps_by_lens={
            LensId.PRIOR_ART: ["baselines exclude RAG and compression methods"],
            LensId.ADVERSARIAL: ["citation auditing risk"],
        },
        now=now,
    )
    ablated = _dispatch_with_gaps(
        store,
        ids,
        session,
        brief,
        panel=(LensId.ADVERSARIAL,),
        gaps_by_lens={LensId.ADVERSARIAL: ["citation auditing risk"]},
        now=now,
    )

    comparison = compare_lens_ablation(
        store, full_event=full, ablated_event=ablated, removed_lens=LensId.PRIOR_ART
    )
    assert comparison.removed_lens is LensId.PRIOR_ART
    assert "baselines exclude rag and compression methods" in [
        g.lower() for g in comparison.gaps_only_in_full
    ]
    assert any("citation auditing" in g.lower() for g in comparison.gaps_in_both)
    assert comparison.attributable_to_removed_lens == comparison.gaps_only_in_full


def test_ablation_requires_same_brief(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    brief: Brief,
    now: datetime,
) -> None:
    store.save_brief(brief)
    full = _dispatch_with_gaps(
        store, ids, session, brief, panel=(LensId.PRIOR_ART,), gaps_by_lens={}, now=now
    )
    other_brief = brief.model_copy(update={"version": 2})
    store.save_brief(other_brief)
    ablated = _dispatch_with_gaps(
        store,
        ids,
        session,
        other_brief,
        panel=(LensId.PRIOR_ART,),
        gaps_by_lens={},
        now=now,
    )
    with pytest.raises(ValueError, match="same brief"):
        compare_lens_ablation(
            store, full_event=full, ablated_event=ablated, removed_lens=LensId.PRIOR_ART
        )
