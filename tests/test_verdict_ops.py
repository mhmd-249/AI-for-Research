"""VerdictResolver: load + validate hand-authored Verdict files and lazily
resolve loose ``(lens, excerpt)`` refs to ``FindingId``s within the session
(stories 124-129)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from research_council.domain.verdict_ops import (
    ResolutionStatus,
    load_verdict_draft,
    materialize_verdict,
    resolve_verdict,
)
from research_council.enums import (
    ClaimType,
    Confidence,
    Decision,
    LensId,
    LensRunStatus,
)
from research_council.ids import (
    SequentialIdGenerator,
    new_finding_id,
    new_lens_run_id,
    new_verdict_id,
)
from research_council.models import (
    Brief,
    Finding,
    LensRun,
    Session,
    Verdict,
    VerdictDraft,
    VerdictRef,
    VerdictRefDraft,
)
from research_council.store import InMemorySessionStore

# --- draft validation -------------------------------------------------------


def _draft_payload(session: Session) -> dict[str, object]:
    return {
        "session_ref": session.id,
        "decision": Decision.ABANDONED.value,
        "outcome_summary": "Killed by the circularity gap.",
        "predictions_validated": [
            {"lens": LensId.FIRST_PRINCIPLES.value, "claim_text_excerpt": "circularity"},
        ],
        "predictions_invalidated": [],
    }


def test_verdict_draft_constructs_from_valid_payload(session: Session) -> None:
    draft = VerdictDraft.model_validate(_draft_payload(session))
    assert draft.decision is Decision.ABANDONED
    assert draft.predictions_validated[0].claim_text_excerpt == "circularity"


def test_verdict_draft_rejects_unknown_field(session: Session) -> None:
    payload = _draft_payload(session)
    payload["mood"] = "grim"
    with pytest.raises(ValidationError):
        VerdictDraft.model_validate(payload)


def test_verdict_draft_rejects_bad_decision(session: Session) -> None:
    payload = _draft_payload(session)
    payload["decision"] = "still_thinking"
    with pytest.raises(ValidationError):
        VerdictDraft.model_validate(payload)


def test_verdict_draft_rejects_id_field(session: Session) -> None:
    # The draft is the *hand-authored* envelope; ids are minted by materialize,
    # not provided by the author.
    payload = _draft_payload(session)
    payload["id"] = "verdict_handwritten"
    with pytest.raises(ValidationError):
        VerdictDraft.model_validate(payload)


def test_verdict_ref_draft_rejects_resolved_ids() -> None:
    # The resolved ids are derived; an author writing them in the file is a bug
    # to surface, not a payload to honor.
    with pytest.raises(ValidationError):
        VerdictRefDraft.model_validate(
            {
                "lens": LensId.FIRST_PRINCIPLES.value,
                "claim_text_excerpt": "x",
                "resolved_finding_ids": ["finding_1"],
            }
        )


# --- loader (JSON/YAML) -----------------------------------------------------


def test_load_verdict_from_json(tmp_path: Path, session: Session) -> None:
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(_draft_payload(session)))

    draft = load_verdict_draft(path)
    assert isinstance(draft, VerdictDraft)
    assert draft.decision is Decision.ABANDONED
    assert draft.session_ref == session.id


def test_load_verdict_from_yaml(tmp_path: Path, session: Session) -> None:
    path = tmp_path / "verdict.yaml"
    path.write_text(yaml.safe_dump(_draft_payload(session)))

    draft = load_verdict_draft(path)
    assert isinstance(draft, VerdictDraft)
    assert draft.decision is Decision.ABANDONED


def test_load_verdict_from_yml_extension(tmp_path: Path, session: Session) -> None:
    path = tmp_path / "verdict.yml"
    path.write_text(yaml.safe_dump(_draft_payload(session)))

    draft = load_verdict_draft(path)
    assert draft.session_ref == session.id


def test_load_verdict_rejects_invalid_payload(tmp_path: Path, session: Session) -> None:
    payload = _draft_payload(session)
    del payload["decision"]
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValidationError):
        load_verdict_draft(path)


def test_load_verdict_rejects_unknown_extension(tmp_path: Path, session: Session) -> None:
    path = tmp_path / "verdict.txt"
    path.write_text(json.dumps(_draft_payload(session)))

    with pytest.raises(ValueError, match="unsupported"):
        load_verdict_draft(path)


def test_load_verdict_rejects_non_object_root(tmp_path: Path) -> None:
    # A JSON array at the root is not a Verdict payload; pydantic will reject it,
    # but the loader should surface this rather than crash on a list-vs-dict bug.
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(["not", "a", "verdict"]))

    with pytest.raises(ValueError, match="object"):
        load_verdict_draft(path)


# --- materialize ------------------------------------------------------------


def test_materialize_verdict_mints_id_and_empty_resolved_ids(
    ids: SequentialIdGenerator, session: Session
) -> None:
    draft = VerdictDraft.model_validate(_draft_payload(session))
    verdict = materialize_verdict(draft, ids)

    assert isinstance(verdict, Verdict)
    assert verdict.id.startswith("verdict_")
    assert verdict.session_ref == session.id
    assert verdict.predictions_validated[0].resolved_finding_ids == ()


# --- resolver ---------------------------------------------------------------


def _save_finding_under_lens(
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    *,
    lens_id: LensId,
    brief_version: int,
    claim_text: str,
) -> Finding:
    finding = Finding(
        id=new_finding_id(ids),
        claim_text=claim_text,
        claim_type=ClaimType.GAP,
        confidence=Confidence.LOAD_BEARING,
        failure_modes_if_wrong="x",
        round=1,
    )
    store.save_finding(finding)
    run = LensRun(
        id=new_lens_run_id(ids),
        lens_id=lens_id,
        session_id=session.id,
        brief_version=brief_version,
        round=1,
        status=LensRunStatus.SUCCEEDED,
        finding_ids=(finding.id,),
    )
    store.save_lens_run(run)
    return finding


def _build_verdict(
    ids: SequentialIdGenerator,
    session: Session,
    refs: list[VerdictRef],
) -> Verdict:
    return Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.ABANDONED,
        outcome_summary="x",
        predictions_validated=tuple(refs),
    )


def test_resolver_populates_resolved_ids_for_unambiguous_match(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    store.save_brief(brief)
    finding = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    verdict = _build_verdict(
        ids,
        session,
        [VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity")],
    )

    report = resolve_verdict(verdict, store)
    resolved_ref = report.verdict.predictions_validated[0]
    assert resolved_ref.resolved_finding_ids == (finding.id,)
    assert report.diagnostics[0].status is ResolutionStatus.RESOLVED


def test_resolver_filters_by_lens(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    store.save_brief(brief)
    # A different lens also emitted a claim with the same excerpt — the resolver
    # must NOT consider it (eval asks "when lens X emitted Y, was it right?").
    _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.PRIOR_ART,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    target = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    verdict = _build_verdict(
        ids,
        session,
        [VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity")],
    )

    report = resolve_verdict(verdict, store)
    assert report.verdict.predictions_validated[0].resolved_finding_ids == (target.id,)


def test_resolver_caches_already_resolved_refs(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    # If a ref already carries resolved_finding_ids, the resolver leaves it as
    # the cached truth — even if the excerpt no longer matches anything in the
    # store (the cache IS the persistence).
    store.save_brief(brief)
    cached_id = new_finding_id(ids)
    verdict = _build_verdict(
        ids,
        session,
        [
            VerdictRef(
                lens=LensId.FIRST_PRINCIPLES,
                claim_text_excerpt="no longer present",
                resolved_finding_ids=(cached_id,),
            )
        ],
    )

    report = resolve_verdict(verdict, store)
    assert report.verdict.predictions_validated[0].resolved_finding_ids == (cached_id,)
    # The diagnostic should mark it as cached, not re-resolved.
    assert report.diagnostics == ()


def test_resolver_emits_no_match_diagnostic(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    store.save_brief(brief)
    _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="Something about attention heads.",
    )
    verdict = _build_verdict(
        ids,
        session,
        [VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity")],
    )

    report = resolve_verdict(verdict, store)
    assert report.verdict.predictions_validated[0].resolved_finding_ids == ()
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.status is ResolutionStatus.NO_MATCH
    assert diag.excerpt == "circularity"
    assert diag.candidate_finding_ids == ()


def test_resolver_emits_ambiguous_diagnostic(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    store.save_brief(brief)
    a = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    b = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="Circularity also undermines the validator design.",
    )
    verdict = _build_verdict(
        ids,
        session,
        [VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity")],
    )

    report = resolve_verdict(verdict, store)
    # Ambiguous: cache stays empty so the engineer must disambiguate, but the
    # candidates are surfaced in the diagnostic (not silently dropped).
    assert report.verdict.predictions_validated[0].resolved_finding_ids == ()
    diag = report.diagnostics[0]
    assert diag.status is ResolutionStatus.AMBIGUOUS
    assert set(diag.candidate_finding_ids) == {a.id, b.id}


def test_resolver_fuzzy_matches_minor_typos(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    # An engineer's excerpt typed from memory months later will not be exact.
    store.save_brief(brief)
    finding = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    verdict = _build_verdict(
        ids,
        session,
        [
            VerdictRef(
                lens=LensId.FIRST_PRINCIPLES,
                claim_text_excerpt="the filter agent has a circulrity problem",
            )
        ],
    )

    report = resolve_verdict(verdict, store)
    assert report.verdict.predictions_validated[0].resolved_finding_ids == (finding.id,)


def test_resolver_runs_on_both_validated_and_invalidated(
    store: InMemorySessionStore, ids: SequentialIdGenerator, session: Session, brief: Brief
) -> None:
    store.save_brief(brief)
    validated = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    invalidated = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.ADVERSARIAL,
        brief_version=brief.version,
        claim_text="The proposal is brittle under domain shift.",
    )
    verdict = Verdict(
        id=new_verdict_id(ids),
        session_ref=session.id,
        decision=Decision.PURSUED_MODIFIED,
        outcome_summary="x",
        predictions_validated=(
            VerdictRef(lens=LensId.FIRST_PRINCIPLES, claim_text_excerpt="circularity"),
        ),
        predictions_invalidated=(
            VerdictRef(lens=LensId.ADVERSARIAL, claim_text_excerpt="domain shift"),
        ),
    )

    report = resolve_verdict(verdict, store)
    assert report.verdict.predictions_validated[0].resolved_finding_ids == (validated.id,)
    assert report.verdict.predictions_invalidated[0].resolved_finding_ids == (invalidated.id,)


def test_load_materialize_resolve_round_trip(
    tmp_path: Path,
    store: InMemorySessionStore,
    ids: SequentialIdGenerator,
    session: Session,
    brief: Brief,
) -> None:
    # The full v0 flow: load -> materialize -> resolve. Author wrote the excerpt;
    # the system attached precise pointers without the author touching IDs.
    store.save_brief(brief)
    finding = _save_finding_under_lens(
        store,
        ids,
        session,
        lens_id=LensId.FIRST_PRINCIPLES,
        brief_version=brief.version,
        claim_text="The filter agent has a circularity problem.",
    )
    path = tmp_path / "v.json"
    path.write_text(json.dumps(_draft_payload(session)))

    draft = load_verdict_draft(path)
    verdict = materialize_verdict(draft, ids)
    report = resolve_verdict(verdict, store)

    assert report.verdict.predictions_validated[0].resolved_finding_ids == (finding.id,)
