"""Candidate ranking and final score computation."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass

from behavior_score import compute_behavior_score
from config import ScoringContext
from education_score import compute_education_score
from experience_score import compute_experience_score
from feature_engineering import CandidateFeatures
from penalty import compute_penalty
from resume_score import compute_resume_score
from utils import clamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A scored candidate ready for export."""

    candidate_id: str
    score: float
    reasoning: str


def build_reasoning(features: CandidateFeatures, ctx: ScoringContext) -> str:
    """Generate concise reasoning for a ranked candidate."""
    title = features.current_title or "Candidate"
    reasoning = (
        f"{title} with {features.years_of_experience:.1f} yrs; "
        f"{features.ai_core_skill_count} AI core skills; "
        f"response rate {features.recruiter_response_rate:.2f}."
    )
    return reasoning[: ctx.config.reasoning_max_length]


def compute_final_score(
    features: CandidateFeatures, ctx: ScoringContext
) -> float:
    """Compute the weighted final score minus penalties."""
    config = ctx.config
    weights = config.component_weights

    resume = compute_resume_score(features, ctx)
    experience = compute_experience_score(features, ctx)
    education = compute_education_score(features, ctx)
    behavior = compute_behavior_score(features, ctx)
    penalty = compute_penalty(features, ctx)

    weighted = (
        weights.get("resume", 0.0) * resume
        + weights.get("experience", 0.0) * experience
        + weights.get("education", 0.0) * education
        + weights.get("behavioral", 0.0) * behavior
    )
    return round(
        clamp(
            weighted - penalty,
            config.normalization.min_score,
            config.normalization.max_score,
        ),
        4,
    )


def score_candidate(
    features: CandidateFeatures, ctx: ScoringContext
) -> RankedCandidate:
    """Score a single candidate and build reasoning."""
    return RankedCandidate(
        candidate_id=features.candidate_id,
        score=compute_final_score(features, ctx),
        reasoning=build_reasoning(features, ctx),
    )


def select_top_candidates(
    heap: list[tuple[float, int, str, str]],
    ctx: ScoringContext,
) -> list[RankedCandidate]:
    """
    Sort heap-selected candidates for export.

    Tie-break: higher score first, then candidate_id ascending.
    """
    ordered = sorted(heap, key=lambda item: (-item[0], item[2]))
    return [
        RankedCandidate(candidate_id=cid, score=score, reasoning=reasoning)
        for score, _, cid, reasoning in ordered
    ]


def update_top_heap(
    heap: list[tuple[float, int, str, str]],
    score: float,
    numeric_id: int,
    candidate_id: str,
    reasoning: str,
    top_n: int,
) -> None:
    """Maintain a min-heap of the top-N candidates during streaming."""
    entry = (score, -numeric_id, candidate_id, reasoning)
    if len(heap) < top_n:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)
