"""BriefVerificationStream — background verification during intake (Layer 3.1).

Wraps :class:`~research_council.verifier.ClaimVerifier` with the lifecycle the
master agent needs while intake is in progress:

* claims are :meth:`submit`-ted as the user adds / edits ``background_claims``;
  each submit kicks off a verification task on the asyncio loop and returns
  immediately so the intake conversation never blocks (story 80, "silent during
  the conversation");
* :meth:`await_completion` is called once intake is ready to terminate —
  by then most or all tasks are already done because they ran in parallel
  with the conversation;
* :meth:`problems` enumerates anything the user must triage (anything that
  isn't ``verified`` / ``partially_supported`` / ``unverifiable_by_design``)
  with a fixed 3-option menu: ``fix`` / ``replace`` / ``dispatch_unverified``
  (story 79);
* :meth:`accept_unverified` records the user's ``dispatch_unverified`` choice
  for a problem; :meth:`apply_to` then propagates the chosen tags onto the
  Findings going into the dispatched brief — for ``dispatch_unverified`` the
  status is rewritten to ``unverified`` so every lens sees the tag on the
  brief it receives (story 65).
* The ``fix`` and ``replace`` options are handled by the orchestrator
  re-submitting a corrected :class:`ClaimToVerify` with the same
  :class:`FindingId`; the stream replaces the prior result on the new submit.
* Empty ``background_claims`` is a no-op — ``is_noop`` is ``True``,
  ``await_completion`` returns immediately, and ``problems`` is empty
  (story 68 exploratory mode).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..enums import VerificationStatus
from ..ids import FindingId
from ..models import Finding, VerificationResult
from ..verifier import ClaimToVerify, ClaimVerifier

# A "problem" is a verification outcome the user must triage at the end of
# intake. partially_supported is intentionally NOT a problem: the brief carries
# that status forward and lenses can decide what to do with it. verified and
# unverifiable_by_design are clean outcomes; unverified means a verification
# was never run (e.g. a Finding added to background_claims but not submitted).
_PROBLEM_STATUSES: frozenset[VerificationStatus] = frozenset(
    {
        VerificationStatus.CONTRADICTED,
        VerificationStatus.SOURCE_NOT_FOUND,
        VerificationStatus.VERIFIER_ERROR,
    }
)


class ProblemAction(StrEnum):
    """The 3 options offered per problem at end-of-intake (story 79)."""

    FIX = "fix"
    REPLACE = "replace"
    DISPATCH_UNVERIFIED = "dispatch_unverified"


PROBLEM_OPTIONS: tuple[ProblemAction, ...] = (
    ProblemAction.FIX,
    ProblemAction.REPLACE,
    ProblemAction.DISPATCH_UNVERIFIED,
)


@dataclass(frozen=True)
class Problem:
    """A verification outcome surfaced for user resolution at end of intake."""

    finding_id: FindingId
    result: VerificationResult
    options: tuple[ProblemAction, ...] = PROBLEM_OPTIONS


class BriefVerificationStream:
    """Stateful background-verification stream. One instance per intake session.

    Not thread-safe — designed to live inside the same asyncio loop as the
    intake conversation."""

    def __init__(self, *, verifier: ClaimVerifier) -> None:
        self._verifier = verifier
        self._tasks: dict[FindingId, asyncio.Task[VerificationResult]] = {}
        self._results: dict[FindingId, VerificationResult] = {}
        self._accepted_unverified: set[FindingId] = set()

    # --- State queries -----------------------------------------------------

    @property
    def is_noop(self) -> bool:
        """True if no claim has ever been submitted (story 68: exploratory mode
        commonly leaves background_claims empty)."""
        return not self._tasks

    def results(self) -> Mapping[FindingId, VerificationResult]:
        """Currently-known results. Empty during intake until at least one task
        has resolved AND await_completion has reaped it."""
        return dict(self._results)

    # --- Submission --------------------------------------------------------

    def submit(self, claim: ClaimToVerify) -> None:
        """Kick off background verification for one claim.

        Resubmitting with the same FindingId (the ``fix`` / ``replace`` path)
        cancels the prior in-flight task and discards its result; the next
        :meth:`await_completion` will reap the new task. Any previous user
        acceptance is also cleared — fixed claims start fresh.
        """
        fid = claim.finding_id
        prior_task = self._tasks.get(fid)
        if prior_task is not None and not prior_task.done():
            prior_task.cancel()
        self._results.pop(fid, None)
        self._accepted_unverified.discard(fid)
        self._tasks[fid] = asyncio.create_task(self._verifier.verify(claim))

    # --- Completion --------------------------------------------------------

    async def await_completion(self) -> None:
        """Wait for every in-flight task to resolve. Idempotent.

        Called when intake is ready to terminate. By story 80, the conversation
        has run in parallel with verification, so this typically returns after
        a short tail and only a handful of tasks remain pending.
        """
        pending: dict[FindingId, asyncio.Task[VerificationResult]] = {
            fid: task for fid, task in self._tasks.items() if fid not in self._results
        }
        if not pending:
            return
        results = await asyncio.gather(*pending.values())
        for fid, res in zip(pending, results, strict=True):
            self._results[fid] = res

    # --- Problem surfacing -------------------------------------------------

    def problems(self) -> tuple[Problem, ...]:
        """Verification results requiring user triage. Excludes anything the
        user has already resolved via :meth:`accept_unverified`."""
        return tuple(
            Problem(finding_id=fid, result=res)
            for fid, res in self._results.items()
            if res.status in _PROBLEM_STATUSES and fid not in self._accepted_unverified
        )

    def accept_unverified(self, finding_id: FindingId) -> None:
        """Record the user's ``dispatch_unverified`` choice for one problem."""
        if finding_id not in self._results:
            raise ValueError(
                f"no verification result for finding {finding_id!r}; cannot accept_unverified"
            )
        self._accepted_unverified.add(finding_id)

    # --- Apply to dispatched findings --------------------------------------

    def apply_to(self, findings: Iterable[Finding]) -> tuple[Finding, ...]:
        """Rewrite each Finding's ``verification_status`` based on its result.

        Rules:

        * Finding with no result in this stream → returned unchanged.
        * Result is a non-problem status (verified / partially_supported /
          unverifiable_by_design) → status written through.
        * Result is a problem AND the user accepted ``dispatch_unverified``
          → status rewritten to ``unverified`` (story 65 — lenses see the tag).
        * Result is a problem with no acceptance → returned unchanged (the
          caller should not be dispatching unresolved problems; we don't
          quietly mark them as the verifier's status either).
        """
        out: list[Finding] = []
        for f in findings:
            res = self._results.get(f.id)
            if res is None:
                out.append(f)
            elif res.status not in _PROBLEM_STATUSES:
                out.append(f.model_copy(update={"verification_status": res.status}))
            elif f.id in self._accepted_unverified:
                out.append(
                    f.model_copy(update={"verification_status": VerificationStatus.UNVERIFIED})
                )
            else:
                out.append(f)
        return tuple(out)


__all__ = [
    "PROBLEM_OPTIONS",
    "BriefVerificationStream",
    "Problem",
    "ProblemAction",
]
