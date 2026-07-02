"""Load and expose application configuration from config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from utils import SkillMatcher, TitleMatcher, normalize_text


_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"


@dataclass(frozen=True)
class PathsConfig:
    """File path settings."""

    candidates_file: Path
    schema_file: Path
    output_file: Path


@dataclass(frozen=True)
class NormalizationConfig:
    """Final score normalization bounds."""

    min_score: float
    max_score: float


@dataclass(frozen=True)
class ExperienceConfig:
    """Experience scoring parameters."""

    optimal_years: float
    max_years_cap: float
    min_relevant_years: float


@dataclass(frozen=True)
class ResumeScoringConfig:
    """Resume score component settings."""

    skill_sum_weight: float
    coverage_weight: float
    title_relevant_bonus: float
    title_irrelevant_penalty: float
    profile_keyword_bonus: float
    certification_bonus: float
    certification_cap: int
    skill_sum_high: float
    skill_count_high: float
    endorsement_divisor: float
    endorsement_cap: float
    duration_divisor: float
    duration_cap: float


@dataclass(frozen=True)
class ExperienceScoringConfig:
    """Experience score component settings."""

    years_weight: float
    ai_years_weight: float
    continuity_weight: float
    relevance_weight: float
    ai_years_low: float
    gap_months_high: float
    gap_continuity_min: float
    default_relevance: float
    irrelevant_relevance: float
    skills_relevance: float
    skills_relevance_min_count: int


@dataclass(frozen=True)
class EducationScoringConfig:
    """Education score component settings."""

    field_weight: float
    degree_weight: float
    missing_education_score: float


@dataclass(frozen=True)
class PenaltyThresholds:
    """Thresholds for deterministic penalty rules."""

    yoe_mismatch_years: float
    keyword_stuffing_min_ai_skills: int
    keyword_stuffing_min_beginner: int
    beginner_max_endorsements: int
    beginner_max_duration_months: int
    contradictory_title_min_ai_skills: int
    inactive_days: int
    low_response_rate: float
    keyword_density_threshold: float
    keyword_density_min_ai_skills: int
    keyword_stuffing_partial_multiplier: float
    max_total_penalty: float
    default_inactive_days: int


@dataclass(frozen=True)
class BehavioralNormSpec:
    """Normalization bounds for a single behavioral metric."""

    low: float
    high: float
    invert: bool = False
    missing_value: float | None = None


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    paths: PathsConfig
    component_weights: dict[str, float]
    skill_weights: dict[str, float]
    skill_synonyms: dict[str, str]
    proficiency_multipliers: dict[str, float]
    proficiency_rank: dict[str, int]
    ai_title_keywords: tuple[str, ...]
    irrelevant_title_keywords: tuple[str, ...]
    education_field_scores: dict[str, float]
    degree_level_scores: dict[str, float]
    education_scoring: EducationScoringConfig
    behavioral_weights: dict[str, float]
    behavioral_normalization: dict[str, BehavioralNormSpec]
    preferred_work_mode_scores: dict[str, float]
    resume_scoring: ResumeScoringConfig
    experience_scoring: ExperienceScoringConfig
    penalties: dict[str, float]
    penalty_thresholds: PenaltyThresholds
    experience: ExperienceConfig
    normalization: NormalizationConfig
    reasoning_max_length: int
    top_n: int
    reference_date: date


@dataclass(frozen=True)
class ScoringContext:
    """Precompiled structures reused across all candidates."""

    config: AppConfig
    skill_matcher: SkillMatcher
    title_matcher: TitleMatcher
    education_field_items: tuple[tuple[str, float], ...]
    degree_level_items: tuple[tuple[str, float], ...]


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be an object.")
    return value


def _load_behavioral_norm(raw: dict[str, Any]) -> dict[str, BehavioralNormSpec]:
    specs: dict[str, BehavioralNormSpec] = {}
    for key, value in raw.items():
        if key == "offer_acceptance_missing_value":
            continue
        if not isinstance(value, dict):
            continue
        specs[key] = BehavioralNormSpec(
            low=float(value["low"]),
            high=float(value["high"]),
            invert=bool(value.get("invert", False)),
            missing_value=(
                float(value["missing_value"])
                if value.get("missing_value") is not None
                else None
            ),
        )
    return specs


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from JSON file."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    base = path.parent
    paths_raw = _require_mapping(raw, "paths")
    exp_raw = _require_mapping(raw, "experience")
    norm_raw = _require_mapping(raw, "normalization")
    reasoning_raw = _require_mapping(raw, "reasoning")
    resume_raw = _require_mapping(raw, "resume_scoring")
    exp_score_raw = _require_mapping(raw, "experience_scoring")
    edu_score_raw = _require_mapping(raw, "education_scoring")
    penalty_thresh_raw = _require_mapping(raw, "penalty_thresholds")
    feature_raw = _require_mapping(raw, "feature_extraction")

    component_weights = _require_mapping(raw, "component_weights")
    total_weight = sum(float(v) for v in component_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(
            f"component_weights must sum to 1.0 (got {total_weight:.4f})."
        )

    behavioral_weights = {
        k: float(v) for k, v in _require_mapping(raw, "behavioral_weights").items()
    }
    behavior_total = sum(behavioral_weights.values())
    if abs(behavior_total - 1.0) > 0.01:
        raise ValueError(
            f"behavioral_weights must sum to 1.0 (got {behavior_total:.4f})."
        )

    return AppConfig(
        paths=PathsConfig(
            candidates_file=_resolve_path(base, str(paths_raw["candidates_file"])),
            schema_file=_resolve_path(base, str(paths_raw["schema_file"])),
            output_file=_resolve_path(base, str(paths_raw["output_file"])),
        ),
        component_weights={k: float(v) for k, v in component_weights.items()},
        skill_weights={
            k.lower(): float(v)
            for k, v in _require_mapping(raw, "skill_weights").items()
        },
        skill_synonyms={
            k.lower(): v.lower()
            for k, v in _require_mapping(raw, "skill_synonyms").items()
        },
        proficiency_multipliers={
            k: float(v)
            for k, v in _require_mapping(raw, "proficiency_multipliers").items()
        },
        proficiency_rank={
            k: int(v) for k, v in _require_mapping(raw, "proficiency_rank").items()
        },
        ai_title_keywords=tuple(
            s.lower() for s in raw.get("ai_title_keywords", [])
        ),
        irrelevant_title_keywords=tuple(
            s.lower() for s in raw.get("irrelevant_title_keywords", [])
        ),
        education_field_scores={
            k.lower(): float(v)
            for k, v in _require_mapping(raw, "education_field_scores").items()
        },
        degree_level_scores={
            k.lower(): float(v)
            for k, v in _require_mapping(raw, "degree_level_scores").items()
        },
        education_scoring=EducationScoringConfig(
            field_weight=float(edu_score_raw["field_weight"]),
            degree_weight=float(edu_score_raw["degree_weight"]),
            missing_education_score=float(edu_score_raw["missing_education_score"]),
        ),
        behavioral_weights=behavioral_weights,
        behavioral_normalization=_load_behavioral_norm(
            _require_mapping(raw, "behavioral_normalization")
        ),
        preferred_work_mode_scores={
            k: float(v)
            for k, v in _require_mapping(raw, "preferred_work_mode_scores").items()
        },
        resume_scoring=ResumeScoringConfig(
            skill_sum_weight=float(resume_raw["skill_sum_weight"]),
            coverage_weight=float(resume_raw["coverage_weight"]),
            title_relevant_bonus=float(resume_raw["title_relevant_bonus"]),
            title_irrelevant_penalty=float(resume_raw["title_irrelevant_penalty"]),
            profile_keyword_bonus=float(resume_raw["profile_keyword_bonus"]),
            certification_bonus=float(resume_raw["certification_bonus"]),
            certification_cap=int(resume_raw["certification_cap"]),
            skill_sum_high=float(resume_raw["skill_sum_high"]),
            skill_count_high=float(resume_raw["skill_count_high"]),
            endorsement_divisor=float(resume_raw["endorsement_divisor"]),
            endorsement_cap=float(resume_raw["endorsement_cap"]),
            duration_divisor=float(resume_raw["duration_divisor"]),
            duration_cap=float(resume_raw["duration_cap"]),
        ),
        experience_scoring=ExperienceScoringConfig(
            years_weight=float(exp_score_raw["years_weight"]),
            ai_years_weight=float(exp_score_raw["ai_years_weight"]),
            continuity_weight=float(exp_score_raw["continuity_weight"]),
            relevance_weight=float(exp_score_raw["relevance_weight"]),
            ai_years_low=float(exp_score_raw["ai_years_low"]),
            gap_months_high=float(exp_score_raw["gap_months_high"]),
            gap_continuity_min=float(exp_score_raw["gap_continuity_min"]),
            default_relevance=float(exp_score_raw["default_relevance"]),
            irrelevant_relevance=float(exp_score_raw["irrelevant_relevance"]),
            skills_relevance=float(exp_score_raw["skills_relevance"]),
            skills_relevance_min_count=int(exp_score_raw["skills_relevance_min_count"]),
        ),
        penalties={k: float(v) for k, v in _require_mapping(raw, "penalties").items()},
        penalty_thresholds=PenaltyThresholds(
            yoe_mismatch_years=float(penalty_thresh_raw["yoe_mismatch_years"]),
            keyword_stuffing_min_ai_skills=int(
                penalty_thresh_raw["keyword_stuffing_min_ai_skills"]
            ),
            keyword_stuffing_min_beginner=int(
                penalty_thresh_raw["keyword_stuffing_min_beginner"]
            ),
            beginner_max_endorsements=int(penalty_thresh_raw["beginner_max_endorsements"]),
            beginner_max_duration_months=int(
                penalty_thresh_raw["beginner_max_duration_months"]
            ),
            contradictory_title_min_ai_skills=int(
                penalty_thresh_raw["contradictory_title_min_ai_skills"]
            ),
            inactive_days=int(penalty_thresh_raw["inactive_days"]),
            low_response_rate=float(penalty_thresh_raw["low_response_rate"]),
            keyword_density_threshold=float(
                penalty_thresh_raw["keyword_density_threshold"]
            ),
            keyword_density_min_ai_skills=int(
                penalty_thresh_raw["keyword_density_min_ai_skills"]
            ),
            keyword_stuffing_partial_multiplier=float(
                penalty_thresh_raw["keyword_stuffing_partial_multiplier"]
            ),
            max_total_penalty=float(penalty_thresh_raw["max_total_penalty"]),
            default_inactive_days=int(penalty_thresh_raw["default_inactive_days"]),
        ),
        experience=ExperienceConfig(
            optimal_years=float(exp_raw["optimal_years"]),
            max_years_cap=float(exp_raw["max_years_cap"]),
            min_relevant_years=float(exp_raw["min_relevant_years"]),
        ),
        normalization=NormalizationConfig(
            min_score=float(norm_raw["min_score"]),
            max_score=float(norm_raw["max_score"]),
        ),
        reasoning_max_length=int(reasoning_raw.get("max_length", 200)),
        top_n=int(raw.get("top_n", 100)),
        reference_date=datetime.strptime(
            str(feature_raw["reference_date"]), "%Y-%m-%d"
        ).date(),
    )


def build_scoring_context(config: AppConfig) -> ScoringContext:
    """Build precompiled matchers and lookup tables for scoring."""
    return ScoringContext(
        config=config,
        skill_matcher=SkillMatcher(config.skill_weights, config.skill_synonyms),
        title_matcher=TitleMatcher(
            config.ai_title_keywords, config.irrelevant_title_keywords
        ),
        education_field_items=tuple(config.education_field_scores.items()),
        degree_level_items=tuple(config.degree_level_scores.items()),
    )
