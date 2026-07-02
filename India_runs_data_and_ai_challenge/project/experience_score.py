"""Experience-based scoring."""

from __future__ import annotations

from config import ScoringContext
from feature_engineering import CandidateFeatures
from utils import clamp, normalize_linear


def compute_experience_score(features: CandidateFeatures, ctx: ScoringContext) -> float:
    """Evaluate total experience, AI tenure, continuity, and role relevance."""
    exp_cfg = ctx.config.experience
    score_cfg = ctx.config.experience_scoring

    years_component = normalize_linear(
        features.years_of_experience,
        low=exp_cfg.min_relevant_years,
        high=exp_cfg.optimal_years,
    )

    ai_component = normalize_linear(
        features.ai_role_months / 12.0,
        low=score_cfg.ai_years_low,
        high=exp_cfg.optimal_years,
    )

    continuity = 1.0
    if features.career_gap_months > 0:
        continuity = normalize_linear(
            float(features.career_gap_months),
            low=0.0,
            high=score_cfg.gap_months_high,
            out_min=1.0,
            out_max=score_cfg.gap_continuity_min,
        )

    if features.title_is_ai_relevant:
        relevance = 1.0
    elif features.title_is_irrelevant:
        relevance = score_cfg.irrelevant_relevance
    elif features.ai_core_skill_count >= score_cfg.skills_relevance_min_count:
        relevance = score_cfg.skills_relevance
    else:
        relevance = score_cfg.default_relevance

    score = (
        score_cfg.years_weight * years_component
        + score_cfg.ai_years_weight * ai_component
        + score_cfg.continuity_weight * continuity
        + score_cfg.relevance_weight * relevance
    )
    return clamp(score, 0.0, 1.0)
