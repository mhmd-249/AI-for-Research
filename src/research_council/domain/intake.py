"""IntakeOrchestrator — the master agent's intake conversation (Layer 3.1).

Encapsulates the conversation invariants in code so the master prompt cannot
silently drift from them:

* mode is declared explicitly before intake proceeds (story 66);
* the 9 brief fields are elicited in PRD order (story 69);
* in the first 3 turns the master asks open questions; from turn 4 onwards it
  proposes drafts — except for ``success_criteria_for_deliberation``, which is
  *never* drafted because it defines downstream "useful" (stories 72-74);
* deep_dive requires all 9 fields, 2 intake turns of no edits, and the self-test
  is surfaced as a *recommendation* (story 70-71); exploratory holds the
  4-field lower bar with no self-test (story 68);
* three decline gates: not-a-research-question, insufficient-specificity,
  all-claims-failed-verification (story 78);
* two-screen confirmation; edits to a confirmed brief produce a *new* version
  and re-show Screen 2 (stories 75-76).

The orchestrator is a pure state machine; an outer agent loop is responsible for
showing prompts and asking the user to act. ``next_prompt`` returns what to show;
``submit_field`` / ``edit_field`` / ``idle_turn`` / ``decline`` / ``dispatch``
advance state. The store is touched only when a brief is dispatched or edited
post-confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..enums import LensId, Mode
from ..ids import SessionId
from ..models import Brief
from ..store.interface import SessionStore
from .brief_ops import confirm_brief, revise_brief

# --- Field order (story 69) -------------------------------------------------

# ``mode`` is elicited FIRST (story 66) but, in the Brief object, it lives
# alongside the other 8 fields. This tuple is the elicitation order of the 8
# non-mode fields.
INTAKE_FIELD_ORDER: tuple[str, ...] = (
    "problem_statement",
    "proposed_solution",
    "researcher_context",
    "background_claims",
    "success_criteria_for_deliberation",
    "scope_and_non_scope",
    "panel_constraints",
    "prior_panel_consultations",
)

# Fields the master may NOT draft. ``success_criteria_for_deliberation`` is the
# non-negotiable one (story 74). The other "optional" fields (proposed_solution,
# panel_constraints, prior_panel_consultations) are draftable: the master can
# propose an empty/null draft for accept/edit/reject.
NEVER_DRAFT: frozenset[str] = frozenset({"success_criteria_for_deliberation"})

# Turn threshold for draft-vs-ask. Story 72/73: first 3 turns ask, turn 4+ drafts.
DRAFT_AFTER_TURN: int = 3

# Stability requirement (stories 68, 70): 2 intake turns of no edits.
STABILITY_TURNS_REQUIRED: int = 2

# Self-test question count (story 70).
SELF_TEST_QUESTION_COUNT: int = 3


class PromptKind(StrEnum):
    """What kind of UI surface the next prompt should render."""

    ASK_MODE = "ask_mode"
    ASK_OPEN = "ask_open"
    DRAFT = "draft"
    SCREEN1 = "screen1"
    SCREEN2 = "screen2"
    DECLINE = "decline"


class DeclineCategory(StrEnum):
    """The three decline gates (story 78)."""

    NOT_A_RESEARCH_QUESTION = "not_a_research_question"
    INSUFFICIENT_SPECIFICITY = "insufficient_specificity"
    ALL_CLAIMS_FAILED_VERIFICATION = "all_claims_failed_verification"


@dataclass(frozen=True)
class Prompt:
    """What the orchestrator wants the master to render next."""

    kind: PromptKind
    field: str | None = None
    draft: Any | None = None
    decline_category: DeclineCategory | None = None
    decline_reason: str | None = None


@dataclass(frozen=True)
class Screen1:
    """Full brief in schema order, no verification results (story 75)."""

    fields: dict[str, Any]
    verification_results: None = None


@dataclass(frozen=True)
class VerificationSummary:
    unverified: int
    verified: int


@dataclass(frozen=True)
class Screen2:
    """Dispatch readiness: self-test, verification, proposed panel (story 75)."""

    self_test_questions: tuple[str, ...]
    bearing_fields: tuple[str, ...]
    verification_summary: VerificationSummary | None
    panel_included: dict[LensId, str]
    panel_excluded: dict[LensId, str]


@dataclass(frozen=True)
class DeclineState:
    category: DeclineCategory
    reason: str


class IntakeOrchestrator:
    """Stateful conversation runner. Not thread-safe by design — one orchestrator
    per (session, master conversation)."""

    def __init__(
        self,
        session_id: SessionId,
        store: SessionStore,
    ) -> None:
        self._session_id = session_id
        self._store = store
        self._mode: Mode | None = None
        self._values: dict[str, Any] = {}
        self._turn_count: int = 0
        self._last_change_turn: int | None = None
        self._verification: VerificationSummary | None = None
        self._panel_included: dict[LensId, str] = {}
        self._panel_excluded: dict[LensId, str] = {}
        self._decline: DeclineState | None = None
        self._confirmed_version: int | None = None
        self._needs_rescreen2: bool = False

    # --- Public state queries ----------------------------------------------

    @property
    def session_id(self) -> SessionId:
        return self._session_id

    @property
    def mode(self) -> Mode | None:
        return self._mode

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def decline_state(self) -> DeclineState | None:
        return self._decline

    # --- Mode declaration --------------------------------------------------

    def declare_mode(self, mode: Mode) -> None:
        """Story 66: explicit, one-tap, never inferred. May only be called once."""
        if self._mode is not None:
            raise ValueError("mode has already been declared and cannot be re-declared")
        self._mode = mode

    # --- Field submission --------------------------------------------------

    def _require_mode(self) -> Mode:
        if self._mode is None:
            raise ValueError("mode must be declared before intake proceeds")
        return self._mode

    def _next_field_in_order(self) -> str | None:
        for f in INTAKE_FIELD_ORDER:
            if f not in self._values:
                return f
        return None

    def _record_change(self, field_name: str, value: Any) -> None:
        """Apply a field change and advance the stability clock."""
        self._values[field_name] = value
        self._turn_count += 1
        self._last_change_turn = self._turn_count

    def submit_field(self, field_name: str, value: Any) -> None:
        """Record the user's answer for the next-in-order field."""
        self._require_mode()
        if field_name not in INTAKE_FIELD_ORDER:
            raise ValueError(f"unknown intake field: {field_name}")
        expected = self._next_field_in_order()
        if expected != field_name:
            raise ValueError(
                f"field {field_name!r} is out of order; expected {expected!r}"
            )
        self._record_change(field_name, value)

    def edit_field(self, field_name: str, value: Any) -> None:
        """Edit an already-submitted field. Counts as a brief edit (story 86),
        which resets the 2-turn stability clock."""
        self._require_mode()
        if field_name not in INTAKE_FIELD_ORDER:
            raise ValueError(f"unknown intake field: {field_name}")
        if field_name not in self._values:
            raise ValueError(f"cannot edit {field_name!r}: not yet submitted")
        self._record_change(field_name, value)

    def idle_turn(self) -> None:
        """Advance one intake turn without editing any field. Required to satisfy
        the 2-turn stability requirement (stories 68, 70)."""
        self._require_mode()
        self._turn_count += 1

    # --- Decline gates -----------------------------------------------------

    def decline(self, category: DeclineCategory, reason: str) -> None:
        self._decline = DeclineState(category=category, reason=reason)

    def record_verification_summary(self, *, unverified: int, verified: int) -> None:
        """Story 78(c): >3 unverified, 0 verified -> auto-decline."""
        summary = VerificationSummary(unverified=unverified, verified=verified)
        self._verification = summary
        if summary.unverified > 3 and summary.verified == 0:
            self._decline = DeclineState(
                category=DeclineCategory.ALL_CLAIMS_FAILED_VERIFICATION,
                reason=(
                    f"{summary.unverified} background_claims failed verification "
                    "and none verified"
                ),
            )

    # --- Termination -------------------------------------------------------

    def _is_stable(self) -> bool:
        if self._last_change_turn is None:
            return False
        return (self._turn_count - self._last_change_turn) >= STABILITY_TURNS_REQUIRED

    def _all_required_present(self) -> bool:
        mode = self._require_mode()
        if mode is Mode.DEEP_DIVE:
            return all(f in self._values for f in INTAKE_FIELD_ORDER)
        # exploratory: 4-field lower bar (story 68)
        if "problem_statement" not in self._values:
            return False
        solution = self._values.get("proposed_solution")
        context = self._values.get("researcher_context")
        if not solution and not context:
            return False
        if not self._values.get("success_criteria_for_deliberation"):
            return False
        return bool(self._values.get("scope_and_non_scope"))

    def can_terminate(self) -> bool:
        if self._decline is not None:
            return False
        if not self._all_required_present():
            return False
        return self._is_stable()

    # --- Draft logic -------------------------------------------------------

    def _should_draft(self, field_name: str) -> bool:
        if field_name in NEVER_DRAFT:
            return False
        return self._turn_count >= DRAFT_AFTER_TURN

    def _draft_for(self, field_name: str) -> Any:
        """Placeholder draft content for the field. The actual LLM-generated
        draft is produced by the master agent; this method only returns a
        type-appropriate empty value. The discriminator between ASK_OPEN and
        DRAFT is :class:`PromptKind`, not the draft value itself (``None`` is
        a valid draft for ``panel_constraints``, meaning "no constraints")."""
        if field_name in ("background_claims", "prior_panel_consultations"):
            return []
        if field_name == "panel_constraints":
            return None
        return ""  # textual fields default to an empty draft skeleton

    # --- Next-prompt resolution -------------------------------------------

    def next_prompt(self) -> Prompt:
        if self._decline is not None:
            return Prompt(
                kind=PromptKind.DECLINE,
                decline_category=self._decline.category,
                decline_reason=self._decline.reason,
            )
        if self._mode is None:
            return Prompt(kind=PromptKind.ASK_MODE)
        # _needs_rescreen2 is only set after a successful confirm/edit cycle,
        # so it implies _confirmed_version is not None.
        if self._needs_rescreen2:
            return Prompt(kind=PromptKind.SCREEN2)
        next_field = self._next_field_in_order()
        if next_field is not None:
            if self._should_draft(next_field):
                return Prompt(
                    kind=PromptKind.DRAFT,
                    field=next_field,
                    draft=self._draft_for(next_field),
                )
            return Prompt(kind=PromptKind.ASK_OPEN, field=next_field)
        # All fields present. Outer loop decides whether to go to Screen 1 / 2;
        # we keep prompting Screen 1 by default.
        return Prompt(kind=PromptKind.SCREEN1)

    # --- Self-test (story 70) ---------------------------------------------

    def self_test_questions(self) -> tuple[str, ...]:
        """Three questions a lens might ask that can't be answered from the brief.

        Recommendation only — dispatch is never gated on these. Returns an empty
        tuple in exploratory mode (story 68: no self-test)."""
        if self._mode is not Mode.DEEP_DIVE:
            return ()
        # Stand-in: the real master LLM generates these. The placeholder copy
        # references field names so a downstream review can see the brief was
        # walked, not invented.
        ps = self._values.get("problem_statement", "<problem>")
        sol = self._values.get("proposed_solution") or "<solution>"
        sc = self._values.get("success_criteria_for_deliberation", "<success>")
        return (
            f"What evidence makes you believe {sol!r} addresses {ps!r}?",
            f"Which assumption would you abandon first if {sc!r} is unmet?",
            "What's a near-neighbor problem you ruled out — and why?",
        )

    # --- Panel proposal ----------------------------------------------------

    def propose_panel(
        self,
        included: dict[LensId, str],
        excluded: dict[LensId, str],
    ) -> None:
        self._panel_included = dict(included)
        self._panel_excluded = dict(excluded)

    # --- Screen builders ---------------------------------------------------

    def _ordered_brief_dict(self) -> dict[str, Any]:
        ordered: dict[str, Any] = {f: self._values.get(f) for f in INTAKE_FIELD_ORDER}
        ordered["mode"] = self._mode
        return ordered

    def build_screen1(self) -> Screen1:
        return Screen1(fields=self._ordered_brief_dict())

    def build_screen2(self) -> Screen2:
        bearing = tuple(f for f in INTAKE_FIELD_ORDER if self._values.get(f))
        return Screen2(
            self_test_questions=self.self_test_questions(),
            bearing_fields=bearing,
            verification_summary=self._verification,
            panel_included=dict(self._panel_included),
            panel_excluded=dict(self._panel_excluded),
        )

    # --- Dispatch & post-confirm edits -------------------------------------

    def _build_brief(self, *, version: int, confirmed: bool) -> Brief:
        mode = self._require_mode()
        return Brief(
            session_id=self._session_id,
            version=version,
            mode=mode,
            problem_statement=self._values.get("problem_statement") or "",
            researcher_context=self._values.get("researcher_context") or "",
            success_criteria_for_deliberation=(
                self._values.get("success_criteria_for_deliberation") or ""
            ),
            scope_and_non_scope=self._values.get("scope_and_non_scope") or "",
            proposed_solution=self._values.get("proposed_solution"),
            background_claims=tuple(self._values.get("background_claims") or ()),
            panel_constraints=(
                tuple(self._values["panel_constraints"])
                if self._values.get("panel_constraints")
                else None
            ),
            prior_panel_consultations=tuple(
                self._values.get("prior_panel_consultations") or ()
            ),
            confirmed=confirmed,
        )

    def dispatch(self) -> Brief:
        """Confirm and persist Brief v1. Raises if a decline is active, a
        confirmed version already exists, or termination conditions are not met.

        Post-confirmation the only legal path to mutate the brief is
        :meth:`edit_confirmed_brief` (story 76); a second ``dispatch()`` call
        would silently overwrite v1 with whatever field state the orchestrator
        currently holds, so the guard refuses it."""
        if self._decline is not None:
            raise ValueError(
                f"cannot dispatch: intake declined ({self._decline.category.value})"
            )
        if self._confirmed_version is not None:
            raise ValueError(
                f"cannot dispatch: brief v{self._confirmed_version} is already "
                "confirmed; use edit_confirmed_brief() to revise"
            )
        if not self._all_required_present():
            raise ValueError("cannot dispatch: required fields are missing")
        brief = self._build_brief(version=1, confirmed=True)
        self._store.save_brief(brief)
        self._confirmed_version = 1
        self._needs_rescreen2 = False
        return brief

    def edit_confirmed_brief(self, **edits: Any) -> Brief:
        """Story 76: any post-confirmation edit produces a NEW version, never
        mutates the prior version. The new version is auto-confirmed and Screen 2
        is re-surfaced so the user can review."""
        if self._confirmed_version is None:
            raise ValueError("no confirmed brief to edit")
        prior = self._store.get_brief(self._session_id, self._confirmed_version)
        if prior is None:  # pragma: no cover - defensive
            raise ValueError(
                f"confirmed brief v{self._confirmed_version} missing from store"
            )
        revised = revise_brief(prior, **edits)
        revised = confirm_brief(revised)
        self._store.save_brief(revised)
        # Sync local field values so subsequent dispatches mirror the new state.
        for k, v in edits.items():
            if k in INTAKE_FIELD_ORDER:
                self._values[k] = v
        self._confirmed_version = revised.version
        self._needs_rescreen2 = True
        return revised


# Public re-exports keep the import surface tight.
__all__ = [
    "DRAFT_AFTER_TURN",
    "INTAKE_FIELD_ORDER",
    "NEVER_DRAFT",
    "SELF_TEST_QUESTION_COUNT",
    "STABILITY_TURNS_REQUIRED",
    "DeclineCategory",
    "DeclineState",
    "IntakeOrchestrator",
    "Prompt",
    "PromptKind",
    "Screen1",
    "Screen2",
    "VerificationSummary",
]


