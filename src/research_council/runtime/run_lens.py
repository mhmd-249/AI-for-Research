"""``run_lens`` — the stateless lens execution loop.

Enforces the sub-agent contract in code (not prompt):

* **Tool-access scope.** Only granted domain tools are advertised, and a call to
  any other tool is refused by the runner (never dispatched).
* **Two-strike validation, no auto-repair.** An invalid ``emit_output`` is sent
  back to the lens once with the validation error; a second failure yields
  ``schema_invalid``. Partial validity is impossible — the whole envelope is
  validated atomically by Pydantic.
* **Code-enforced claim-type restriction.** A lens with ``allowed_claim_types``
  (first-principles) that emits a disallowed claim type fails validation.
* **Refusal** is a first-class terminal outcome, not an error.

Outcomes are a discriminated union (data the controllers branch on), never
exceptions. ``run_lens`` does not touch the store; ``persist_outcome`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from ..enums import LensId, LensRunStatus, Round, ToolName, VerificationStatus
from ..ids import (
    DispatchEventId,
    IdGenerator,
    SessionId,
    new_finding_id,
    new_lens_run_id,
)
from ..models import Finding, FindingDraft, LensRun, LensRunOutputDraft
from ..store.interface import SessionStore
from .inputs import LensRunInput, build_lens_prompt, build_system_prompt
from .llm import (
    LlmClient,
    LlmRequest,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
)
from .tools import EMIT_OUTPUT_TOOL, ToolRegistry, emit_output_tool_spec

DEFAULT_MODEL = "claude-opus-4-8"
MAX_EMIT_ATTEMPTS = 2  # two-strike: original + one retry


@dataclass(frozen=True)
class LensRunSucceeded:
    findings: list[Finding]
    open_questions: list[str]
    disagreements_with_my_own_framing: list[str]
    brief_summary: str
    status: Literal["succeeded"] = "succeeded"


@dataclass(frozen=True)
class LensRunRefused:
    reason: str
    status: Literal["refused"] = "refused"


@dataclass(frozen=True)
class LensRunSchemaInvalid:
    raw_output: str
    validation_error: str
    status: Literal["schema_invalid"] = "schema_invalid"


LensRunOutcome = LensRunSucceeded | LensRunRefused | LensRunSchemaInvalid


def _materialize(draft: FindingDraft, id_generator: IdGenerator, round: Round) -> Finding:
    return Finding(
        id=new_finding_id(id_generator),
        claim_text=draft.claim_text,
        claim_type=draft.claim_type,
        confidence=draft.confidence,
        sources=tuple(draft.sources),
        failure_modes_if_wrong=draft.failure_modes_if_wrong,
        verification_status=VerificationStatus.UNVERIFIED,
        round=round,
        responding_to=tuple(draft.responding_to),
    )


def _validate_emit(
    tool_input: dict[str, object],
    inputs: LensRunInput,
    id_generator: IdGenerator,
) -> tuple[LensRunOutcome | None, str | None]:
    """Return (terminal outcome, None) on a decisive result, or (None, error) when
    the lens should be asked to fix and retry."""
    try:
        draft = LensRunOutputDraft.model_validate(tool_input)
    except ValidationError as exc:
        return None, str(exc)

    if draft.refused:
        # refusal_reason is guaranteed by the envelope validator.
        return LensRunRefused(reason=draft.refusal_reason or ""), None

    allowed = inputs.lens_config.allowed_claim_types
    if allowed is not None:
        disallowed = sorted(
            {f.claim_type.value for f in draft.findings if f.claim_type not in allowed}
        )
        if disallowed:
            permitted = ", ".join(ct.value for ct in allowed)
            return None, (
                f"lens {inputs.lens_config.id.value} may only emit claim_type in "
                f"{{{permitted}}}; got disallowed types: {disallowed}"
            )

    findings = [_materialize(f, id_generator, inputs.round) for f in draft.findings]
    return (
        LensRunSucceeded(
            findings=findings,
            open_questions=list(draft.open_questions),
            disagreements_with_my_own_framing=list(draft.disagreements_with_my_own_framing),
            brief_summary=draft.brief_summary,
        ),
        None,
    )


def _advertised_tools(inputs: LensRunInput, tools: ToolRegistry) -> list[ToolSpec]:
    specs = [
        ToolSpec(
            name=tools[name].name.value,
            description=tools[name].description,
            input_schema=tools[name].input_schema,
        )
        for name in inputs.lens_config.tool_access
        if name in tools
    ]
    specs.append(
        ToolSpec(
            name=EMIT_OUTPUT_TOOL,
            description="Emit your final typed analysis. Call this exactly once.",
            input_schema=emit_output_tool_spec(),
        )
    )
    return specs


async def run_lens(
    *,
    inputs: LensRunInput,
    client: LlmClient,
    id_generator: IdGenerator,
    tools: ToolRegistry | None = None,
    model: str = DEFAULT_MODEL,
    caller_ref: str = "lensrun",
    max_turns: int = 8,
) -> LensRunOutcome:
    tools = tools or {}
    granted = {name.value for name in inputs.lens_config.tool_access if name in tools}

    system = build_system_prompt(inputs.lens_config, inputs.brief.mode.value)
    advertised = _advertised_tools(inputs, tools)
    messages: list[Message] = [Message(role="user", content=[TextBlock(build_lens_prompt(inputs))])]

    emit_attempts = 0
    last_emit_raw = ""

    for _ in range(max_turns):
        response = await client.create_message(
            LlmRequest(
                model=model,
                system=system,
                messages=messages,
                tools=advertised,
                caller_ref=caller_ref,
            )
        )
        messages.append(Message(role="assistant", content=response.content))

        tool_uses = response.tool_uses
        if not tool_uses:
            messages.append(
                Message(
                    role="user",
                    content=[TextBlock("Call the emit_output tool to submit your analysis.")],
                )
            )
            continue

        results: list[ToolResultBlock] = []
        for use in tool_uses:
            if use.name == EMIT_OUTPUT_TOOL:
                last_emit_raw = repr(use.input)
                outcome, error = _validate_emit(use.input, inputs, id_generator)
                if outcome is not None:
                    return outcome
                emit_attempts += 1
                if emit_attempts >= MAX_EMIT_ATTEMPTS:
                    return LensRunSchemaInvalid(
                        raw_output=last_emit_raw, validation_error=error or "invalid emit_output"
                    )
                results.append(
                    ToolResultBlock(
                        tool_use_id=use.id,
                        content=f"emit_output was invalid. Fix and resubmit. Error:\n{error}",
                        is_error=True,
                    )
                )
            elif use.name in granted:
                output = await tools[ToolName(use.name)].handle(use.input)
                results.append(ToolResultBlock(tool_use_id=use.id, content=output))
            else:
                # Tool-access enforcement: never dispatch an ungranted tool.
                results.append(
                    ToolResultBlock(
                        tool_use_id=use.id,
                        content=f"Tool '{use.name}' is not available to this lens.",
                        is_error=True,
                    )
                )

        messages.append(Message(role="user", content=list(results)))

    return LensRunSchemaInvalid(
        raw_output=last_emit_raw,
        validation_error="lens did not produce a valid emit_output within the turn limit",
    )


def persist_outcome(
    store: SessionStore,
    *,
    outcome: LensRunOutcome,
    lens_id: LensId,
    session_id: SessionId,
    brief_version: int,
    round: Round | None,
    id_generator: IdGenerator,
    dispatch_event_id: DispatchEventId | None = None,
) -> LensRun:
    """Persist a lens outcome: save its Findings and the LensRun record, and
    return the saved run. Status is derived from the outcome variant."""
    run_id = new_lens_run_id(id_generator)

    if isinstance(outcome, LensRunSucceeded):
        for finding in outcome.findings:
            store.save_finding(finding)
        run = LensRun(
            id=run_id,
            lens_id=lens_id,
            session_id=session_id,
            brief_version=brief_version,
            round=round,
            dispatch_event_id=dispatch_event_id,
            status=LensRunStatus.SUCCEEDED,
            finding_ids=tuple(f.id for f in outcome.findings),
            open_questions=tuple(outcome.open_questions),
            disagreements_with_my_own_framing=tuple(outcome.disagreements_with_my_own_framing),
            brief_summary=outcome.brief_summary,
        )
    elif isinstance(outcome, LensRunRefused):
        run = LensRun(
            id=run_id,
            lens_id=lens_id,
            session_id=session_id,
            brief_version=brief_version,
            round=round,
            dispatch_event_id=dispatch_event_id,
            status=LensRunStatus.REFUSED,
            refusal_reason=outcome.reason,
        )
    else:
        run = LensRun(
            id=run_id,
            lens_id=lens_id,
            session_id=session_id,
            brief_version=brief_version,
            round=round,
            dispatch_event_id=dispatch_event_id,
            status=LensRunStatus.SCHEMA_INVALID,
            raw_output=outcome.raw_output,
        )

    store.save_lens_run(run)
    return run
