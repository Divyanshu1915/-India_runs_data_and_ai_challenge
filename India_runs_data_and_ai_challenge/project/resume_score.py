"""Resume and skill-based scoring."""

from __future__ import annotations

from config import ScoringContext
from feature_engineering import CandidateFeatures
from utils import clamp, normalize_linear


def compute_resume_score(features: CandidateFeatures, ctx: ScoringContext) -> float:
    """
    Score resume content using configured skill weights and profile signals.

    Evaluates structured skills and certifications from the candidate schema.
    """
    cfg = ctx.config.resume_scoring

    if not features.skill_details:
        return 0.0

    skill_component = normalize_linear(
        features.ai_skill_score_sum,
        low=0.0,
        high=cfg.skill_sum_high,
    )
    coverage = normalize_linear(
        float(features.ai_core_skill_count),
        low=0.0,
        high=cfg.skill_count_high,
    )

    title_bonus = cfg.title_relevant_bonus if features.title_is_ai_relevant else 0.0
    title_penalty = cfg.title_irrelevant_penalty if features.title_is_irrelevant else 0.0

    profile_bonus = (
        cfg.profile_keyword_bonus if features.profile_has_skills else 0.0
    )

    cert_bonus = min(
        features.certification_count,
        cfg.certification_cap,
    ) * cfg.certification_bonus

    score = (
        cfg.skill_sum_weight * skill_component
        + cfg.coverage_weight * coverage
        + title_bonus
        + profile_bonus
        + cert_bonus
        - title_penalty
    )
    return clamp(score, 0.0, 1.0)
