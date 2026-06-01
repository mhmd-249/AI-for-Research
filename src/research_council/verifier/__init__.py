"""ClaimVerifier — the 3-stage per-claim verification pipeline (PRD Layer 2.1).

Stage 1 (source resolution), Stage 2 (locality), Stage 3 (entailment) for
empirical / prior_art claims; a tag-appropriateness short-circuit for
unverifiable-by-design claim types. Failures surface as explicit
``verifier_error`` sub-reasons (story 38); ``verifier_error`` is never silently
turned into ``verified`` (story 37).
"""

from .api import (
    UNVERIFIABLE_BY_DESIGN_TYPES,
    ClaimToVerify,
    ClaimVerifier,
    RetryConfig,
)
from .cache import (
    InMemoryVerifierCache,
    VerifierCache,
    claim_text_hash,
)
from .judge import (
    EntailmentResult,
    EntailmentVerdict,
    Judge,
    LocalityResult,
    TagAppropriatenessResult,
)
from .sources import (
    ApiDownError,
    ParseClientError,
    PaywalledError,
    RateLimitedError,
    SourceCandidate,
    SourceClient,
    SourceClientError,
    SourceHint,
    SourceRegistry,
    tiebreak,
)

__all__ = [
    "UNVERIFIABLE_BY_DESIGN_TYPES",
    "ApiDownError",
    "ClaimToVerify",
    "ClaimVerifier",
    "EntailmentResult",
    "EntailmentVerdict",
    "InMemoryVerifierCache",
    "Judge",
    "LocalityResult",
    "ParseClientError",
    "PaywalledError",
    "RateLimitedError",
    "RetryConfig",
    "SourceCandidate",
    "SourceClient",
    "SourceClientError",
    "SourceHint",
    "SourceRegistry",
    "TagAppropriatenessResult",
    "VerifierCache",
    "claim_text_hash",
    "tiebreak",
]
