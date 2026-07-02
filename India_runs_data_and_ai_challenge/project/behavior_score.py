"""Behavioral signal scoring from Redrob platform data."""

from __future__ import annotations

from config import ScoringContext
from feature_engineering import CandidateFeatures
from utils import clamp


def compute_behavior_score(features: CandidateFeatures, ctx: ScoringContext) -> float:
    """
    Deterministic behavioral score using all documented redrob_signals fields.

    Metrics are normalized once during feature extraction.
    """
    weights = ctx.config.behavioral_weights
    metrics = features.behavioral_metrics

    score = sum(weights.get(key, 0.0) * metrics.get(key, 0.0) for key in weights)
    return clamp(score, 0.0, 1.0)
