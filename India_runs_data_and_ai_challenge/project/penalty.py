"""Deterministic penalty detection."""

from __future__ import annotations

from config import ScoringContext
from feature_engineering import CandidateFeatures


def compute_penalty(features: CandidateFeatures, ctx: ScoringContext) -> float:
    """
    Compute total penalty as a value subtracted from the final score.

    All rules use only candidate profile and signal data.
    """
    cfg = ctx.config
    penalties = cfg.penalties
    thresholds = cfg.penalty_thresholds
    total = 0.0

    if features.timeline_issue:
        total += penalties.get("timeline_inconsistency", 0.0)

    if features.yoe_mismatch:
        total += penalties.get("yoe_mismatch", 0.0)

    if (
        features.ai_core_skill_count >= thresholds.keyword_stuffing_min_ai_skills
        and features.beginner_ai_skill_count >= thresholds.keyword_stuffing_min_beginner
    ):
        total += penalties.get("keyword_stuffing", 0.0)

    if (
        features.title_is_irrelevant
        and features.ai_core_skill_count >= thresholds.contradictory_title_min_ai_skills
    ):
        total += penalties.get("contradictory_title", 0.0)

    if not features.has_essential_fields:
        total += penalties.get("missing_essential", 0.0)

    if features.is_inactive:
        total += penalties.get("inactive_profile", 0.0)

    if features.recruiter_response_rate < thresholds.low_response_rate:
        total += penalties.get("low_response_rate", 0.0)

    if (
        features.keyword_density > thresholds.keyword_density_threshold
        and features.ai_core_skill_count >= thresholds.keyword_density_min_ai_skills
    ):
        total += (
            penalties.get("keyword_stuffing", 0.0)
            * thresholds.keyword_stuffing_partial_multiplier
        )

    return min(total, thresholds.max_total_penalty)
