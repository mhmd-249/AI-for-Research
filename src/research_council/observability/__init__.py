"""TraceStore and observability reads (PRD stories 162-165).

Two facilities built over data the data model already stores, not a separate
logging system:

* The **backward trace-view** walks the persisted pointer graph for any
  synthesis claim: synthesis claim → constituent Findings → emitting LensRuns
  → brief version → VerificationResult → Source text → raw TraceRecords (story
  162). See :mod:`.trace_view`.
* **Partial replay** re-dispatches a single LensRun or a single verifier check
  against the stored brief version (story 165). Full-session deterministic
  replay is not achievable (LLM nondeterminism) and is deliberately not faked
  (PRD out-of-scope item 174). See :mod:`.replay`.

The append-only raw-I/O capture itself (story 163) is the existing
``TracingLlmClient`` + ``TraceSink`` pair — unbounded for v0 (story 164) by
construction: nothing in the store evicts trace records.
"""

from .replay import (
    ReplayLensRunInputs,
    Verifier,
    reconstruct_lens_run_inputs,
    replay_lens_run,
    replay_verifier_check,
)
from .trace_view import (
    FindingTrace,
    walk_finding,
    walk_findings,
    walk_synthesis_claim,
)

__all__ = [
    "FindingTrace",
    "ReplayLensRunInputs",
    "Verifier",
    "reconstruct_lens_run_inputs",
    "replay_lens_run",
    "replay_verifier_check",
    "walk_finding",
    "walk_findings",
    "walk_synthesis_claim",
]
