"""Real-API tracer (story: tracer bullet, issue #2).

Runs the first-principles lens end-to-end against a fixture brief through a real
Claude call and prints the persisted, schema-valid output. The deterministic
spine is covered by ``tests/`` with a fake client; this script is the manual
confirmation that a real call works.

Usage::

    ANTHROPIC_API_KEY=sk-... uv run council-tracer
"""

from __future__ import annotations

import asyncio
import os

from ..enums import LensId, Mode
from ..ids import UuidGenerator, new_session_id
from ..lenses.roster import get_lens_config
from ..models import Brief
from ..runtime import (
    AnthropicLlmClient,
    LensRunInput,
    TracingLlmClient,
    persist_outcome,
    run_lens,
)
from ..store import InMemorySessionStore


def _fixture_brief(session_id: object) -> Brief:
    return Brief(
        session_id=session_id,  # type: ignore[arg-type]
        version=1,
        mode=Mode.DEEP_DIVE,
        problem_statement=(
            "Long-context language models under-attend to information in the middle of "
            "their context window, degrading multi-hop reasoning over long inputs."
        ),
        researcher_context="An AI engineer researching open ML problems.",
        success_criteria_for_deliberation=(
            "Surface mechanism-level gaps in the proposal I have not already identified."
        ),
        scope_and_non_scope=(
            "In scope: architectural and mechanistic accounts. Out of scope: training a new "
            "model from scratch."
        ),
        proposed_solution=(
            "Active Context Inhibition: a filter agent prunes the context so the main model "
            "attends only to a distilled, relevant subset."
        ),
        confirmed=True,
    )


async def _run() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the real-API tracer. Exiting without a call.")
        return 1

    ids = UuidGenerator()
    store = InMemorySessionStore()
    session_id = new_session_id(ids)
    brief = _fixture_brief(session_id)

    client = TracingLlmClient(AnthropicLlmClient(), store)
    inputs = LensRunInput(
        brief=brief,
        lens_config=get_lens_config(LensId.FIRST_PRINCIPLES),
        round=1,
    )

    outcome = await run_lens(inputs=inputs, client=client, id_generator=ids, caller_ref="tracer")
    run = persist_outcome(
        store,
        outcome=outcome,
        lens_id=LensId.FIRST_PRINCIPLES,
        session_id=session_id,
        brief_version=1,
        round=1,
        id_generator=ids,
    )

    print(f"Outcome: {outcome.status}")
    print(f"Persisted LensRun {run.id} with status {run.status.value}")
    for finding in store.list_findings_for_lens_run(run.id):
        print(f"  - ({finding.claim_type.value}) {finding.claim_text}")
    print(f"Trace records captured: {len(store.list_traces('tracer'))}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
