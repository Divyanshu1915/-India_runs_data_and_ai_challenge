"""Education-based scoring from structured profile fields."""

from __future__ import annotations

from config import ScoringContext
from feature_engineering import CandidateFeatures
from utils import clamp


def compute_education_score(features: CandidateFeatures, ctx: ScoringContext) -> float:
    """
    Score education using field of study and degree level only.

    Institution names are intentionally not used as a quality proxy.
    """
    if not features.education_field and features.education_best_score <= 0:
        return ctx.config.education_scoring.missing_education_score

    return clamp(features.education_best_score, 0.0, 1.0)
