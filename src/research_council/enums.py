"""Controlled vocabularies for the epistemic schema (PRD stories 21-30) and the
domain state machines. All are string enums for JSON-friendly persistence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

# Round membership is a small closed set; a Literal keeps it lighter than an enum.
# ``None`` is used for findings that predate rounds (background claims) or that
# fall outside the round model (challenge revisions).
Round = Literal[1, 2]


class ClaimType(StrEnum):
    """The five claim types, each with distinct verification semantics (story 21)."""

    EMPIRICAL = "empirical"
    PRIOR_ART = "prior_art"
    MECHANISM_HYPOTHESIS = "mechanism_hypothesis"
    GAP = "gap"
    FAILURE_MODE = "failure_mode"


class Confidence(StrEnum):
    """Per-lens per-claim confidence indicator (story 27). Ordinal, never numeric."""

    LOAD_BEARING = "load_bearing"
    SUPPORTING = "supporting"
    EXPLORATORY = "exploratory"


class VerificationStatus(StrEnum):
    """The seven verification states (story 29). ``unverified`` is an explicit
    state, not the absence of one; ``verifier_error`` never silently becomes
    ``verified`` (story 37)."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    SOURCE_NOT_FOUND = "source_not_found"
    UNVERIFIABLE_BY_DESIGN = "unverifiable_by_design"
    VERIFIER_ERROR = "verifier_error"


class VerifierErrorSubReason(StrEnum):
    """Explicit failure sub-reasons (stories 38, 40)."""

    RATE_LIMITED = "rate_limited"
    PAPER_PAYWALLED_NO_ABSTRACT = "paper_paywalled_no_abstract"
    API_DOWN = "api_down"
    PARSE_ERROR = "parse_error"
    PASSAGE_HALLUCINATED = "passage_hallucinated"


class ChangeReason(StrEnum):
    """Why a challenged finding changed (story 108). The single field that makes
    'updated on evidence' distinguishable from 'caved under pressure'."""

    UNCHANGED = "unchanged"
    REVISED_NEW_EVIDENCE = "revised_new_evidence"
    REVISED_RECONSIDERED = "revised_reconsidered"
    WITHDRAWN = "withdrawn"
    REAFFIRMED_AGAINST_CHALLENGE = "reaffirmed_against_challenge"


class Decision(StrEnum):
    """A Verdict's outcome decision (story 128)."""

    PURSUED_AS_PROPOSED = "pursued_as_proposed"
    PURSUED_MODIFIED = "pursued_modified"
    ABANDONED = "abandoned"
    SUPERSEDED_BY_OTHER_WORK = "superseded_by_other_work"
    UNDETERMINED = "undetermined"


class SessionStatus(StrEnum):
    """Top-level session lifecycle (story 8). Transitions are enforced in
    ``domain.session_ops``."""

    INTAKE = "intake"
    CONSULTING = "consulting"
    SYNTHESIZED = "synthesized"
    ITERATING = "iterating"
    CLOSED = "closed"


class LensRunStatus(StrEnum):
    """Terminal states a LensRun can reach (stories 93, 50)."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    SCHEMA_INVALID = "schema_invalid"
    TIMEOUT = "timeout"
    REFUSED = "refused"


class Mode(StrEnum):
    """Intake mode (story 66)."""

    EXPLORATORY = "exploratory"
    DEEP_DIVE = "deep_dive"


class LensId(StrEnum):
    """The nine v0 lenses (story 54). Cognitive science is cut for v0."""

    PRIOR_ART = "prior_art"
    ADVERSARIAL = "adversarial"
    EMPIRICAL_BENCHMARKING = "empirical_benchmarking"
    MECHANISTIC_INTERPRETABILITY = "mechanistic_interpretability"
    INFORMATION_THEORETIC = "information_theoretic"
    TRAINING_DATA_DISTRIBUTION = "training_data_distribution"
    DEPLOYMENT_SERVING = "deployment_serving"
    ARCHITECTURE = "architecture"
    FIRST_PRINCIPLES = "first_principles"


class ToolName(StrEnum):
    """Grantable lens tools in v0 (story 47). No web search, code execution, or
    arbitrary HTTP. ``emit_output`` is the runtime structured-output tool and is
    deliberately not in this enum."""

    VERIFIER_QUERY = "verifier_query"
    SOURCE_FETCH = "source_fetch"


class TensionSource(StrEnum):
    """How the synthesizer detected a tension (story schema, Layer 5.1)."""

    CONTRADICTION = "contradiction"
    SIBLING_IMPACT_FLAG = "sibling_impact_flag"
    OPPOSING_PAIR = "opposing_pair"
