"""Synthesis support modules. The first deep module is the ClaimClusterer
(same-claim detection), isolated specifically so it is fixture-testable: every
synthesis count depends on it (PRD stories 130-138, 157).
"""

from .clusterer import (
    DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD,
    MIN_CLAIM_TEXT_LENGTH,
    AdjudicationRecord,
    AdjudicationVerdict,
    Cluster,
    ClusterResult,
    Embedder,
    ExcludedFinding,
    SameClaimAdjudicator,
    cluster_findings,
)

__all__ = [
    "DEFAULT_CANDIDATE_SIMILARITY_THRESHOLD",
    "MIN_CLAIM_TEXT_LENGTH",
    "AdjudicationRecord",
    "AdjudicationVerdict",
    "Cluster",
    "ClusterResult",
    "Embedder",
    "ExcludedFinding",
    "SameClaimAdjudicator",
    "cluster_findings",
]
