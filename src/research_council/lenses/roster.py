"""The nine v0 lenses as static configuration (stories 54-63).

``allowed_claim_types`` is ``None`` for lenses whose claim types are guided only
by prompt + eval. It is a hard, code-enforced set only for the first-principles
lens, whose restriction to mechanism_hypothesis + gap is called "non-negotiable"
in story 63. Tool grants for the training-data, deployment, and architecture
lenses are not specified in the roster stories and default to none until their
build slice; the grants below encode exactly what stories 55-63 state.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from ..enums import ClaimType, LensId, ToolName
from ..models import FrozenModel


class LensConfig(FrozenModel):
    """Static identity and capabilities of one lens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: LensId
    name: str
    frame: str
    characteristic_move: str
    role: str
    tool_access: list[ToolName] = Field(default_factory=list)
    allowed_claim_types: list[ClaimType] | None = None


ROSTER: dict[LensId, LensConfig] = {
    LensId.PRIOR_ART: LensConfig(
        id=LensId.PRIOR_ART,
        name="Prior-art",
        frame="What existing work has already addressed this problem or attempted this solution.",
        characteristic_move=(
            "Search the literature and surface the closest prior art the researcher hasn't cited."
        ),
        role="grounder",
        tool_access=[ToolName.VERIFIER_QUERY, ToolName.SOURCE_FETCH],
    ),
    LensId.ADVERSARIAL: LensConfig(
        id=LensId.ADVERSARIAL,
        name="Adversarial",
        frame="This proposal is wrong; my job is to find the strongest reason why.",
        characteristic_move=(
            "Round 1: generate failure_mode findings against the proposed solution. "
            "Round 2: generate failure_mode findings against the emerging Round 1 consensus."
        ),
        role="critic",
        tool_access=[ToolName.SOURCE_FETCH],
    ),
    LensId.EMPIRICAL_BENCHMARKING: LensConfig(
        id=LensId.EMPIRICAL_BENCHMARKING,
        name="Empirical benchmarking",
        frame=(
            "What would the experiment actually look like, and is the proposed evaluation "
            "sufficient."
        ),
        characteristic_move="Scrutinize missing baselines, ablations, and comparisons.",
        role="critic+grounder",
        tool_access=[ToolName.VERIFIER_QUERY, ToolName.SOURCE_FETCH],
    ),
    LensId.MECHANISTIC_INTERPRETABILITY: LensConfig(
        id=LensId.MECHANISTIC_INTERPRETABILITY,
        name="Mechanistic interpretability",
        frame=(
            "What is happening inside the model that causes the phenomenon, at the level of "
            "circuits, attention patterns, and MLP activations."
        ),
        characteristic_move="Propose mechanism-level accounts and the failure modes they imply.",
        role="generator+critic",
    ),
    LensId.INFORMATION_THEORETIC: LensConfig(
        id=LensId.INFORMATION_THEORETIC,
        name="Information-theoretic",
        frame=(
            "What does the problem look like through compression, entropy, mutual information, "
            "and channel capacity."
        ),
        characteristic_move=(
            "Reframe through information theory; every finding must make at least one quantitative "
            "or conditional prediction, not just a reframing."
        ),
        role="generator",
    ),
    LensId.TRAINING_DATA_DISTRIBUTION: LensConfig(
        id=LensId.TRAINING_DATA_DISTRIBUTION,
        name="Training-data / distribution",
        frame=(
            "What does the model's training distribution look like, and does the problem or "
            "solution interact with that distribution."
        ),
        characteristic_move=(
            "Reason about how the training distribution shapes (or undermines) the proposal."
        ),
        role="generator+critic",
    ),
    LensId.DEPLOYMENT_SERVING: LensConfig(
        id=LensId.DEPLOYMENT_SERVING,
        name="Deployment / serving",
        frame=(
            "What does this look like when deployed: memory, latency, KV cache, serving "
            "constraints."
        ),
        characteristic_move=(
            "Catch costs the brief omits, e.g. 'this solution doubles inference cost and the brief "
            "doesn't mention it.'"
        ),
        role="critic",
    ),
    LensId.ARCHITECTURE: LensConfig(
        id=LensId.ARCHITECTURE,
        name="Architecture",
        frame=(
            "What architectural changes would address the problem, across the design space of "
            "attention variants, position encodings, retrieval-augmented architectures, and "
            "mixture-of-experts."
        ),
        characteristic_move=(
            "Map the proposal onto the architectural design space and its trade-offs."
        ),
        role="generator+critic",
    ),
    LensId.FIRST_PRINCIPLES: LensConfig(
        id=LensId.FIRST_PRINCIPLES,
        name="First-principles",
        frame=(
            "What is the minimum mechanism required to produce the observed phenomenon, ignoring "
            "all engineering convenience and inherited architectural choices."
        ),
        characteristic_move="Generate counterfactual reframings of the proposed solution.",
        role="generator",
        # Non-negotiable restriction (story 63): mechanism_hypothesis + gap only.
        allowed_claim_types=[ClaimType.MECHANISM_HYPOTHESIS, ClaimType.GAP],
    ),
}


def get_lens_config(lens_id: LensId) -> LensConfig:
    """Return the static config for a lens, or raise ``KeyError`` if unknown."""
    return ROSTER[lens_id]
