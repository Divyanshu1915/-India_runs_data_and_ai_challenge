"""Extract ranking features from normalized candidate records."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from config import BehavioralNormSpec, ScoringContext
from parser import CandidateRecord
from utils import clamp, normalize_linear, normalize_text, parse_date, safe_float, safe_int

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateFeatures:
    """Precomputed features used by scoring modules."""

    candidate_id: str
    numeric_id: int
    current_title: str
    years_of_experience: float
    headline: str
    summary: str
    skill_details: tuple[dict, ...]
    ai_core_skill_count: int
    ai_skill_score_sum: float
    certification_count: int
    career_months_total: int
    ai_role_months: int
    career_gap_months: int
    title_is_ai_relevant: bool
    title_is_irrelevant: bool
    education_best_score: float
    education_field: str
    profile_text: str
    profile_has_skills: bool
    keyword_density: float
    has_essential_fields: bool
    timeline_issue: bool
    yoe_mismatch: bool
    beginner_ai_skill_count: int
    is_inactive: bool
    behavioral_metrics: dict[str, float]
    recruiter_response_rate: float


def _dedupe_skills(
    skills: list[dict], proficiency_rank: dict[str, int]
) -> tuple[dict, ...]:
    """Deduplicate skills by normalized name, keeping the strongest entry."""
    by_name: dict[str, dict] = {}
    for skill in skills:
        key = skill.get("normalized_name") or normalize_text(skill.get("name", ""))
        if not key:
            continue
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = skill
            continue
        new_rank = proficiency_rank.get(skill.get("proficiency", "beginner"), 0)
        old_rank = proficiency_rank.get(existing.get("proficiency", "beginner"), 0)
        if new_rank > old_rank:
            by_name[key] = skill
        elif new_rank == old_rank and skill.get("endorsements", 0) > existing.get(
            "endorsements", 0
        ):
            by_name[key] = skill
    return tuple(by_name.values())


def _compute_ai_skill_metrics(
    skills: tuple[dict, ...], ctx: ScoringContext
) -> tuple[int, float, int]:
    """Return AI skill count, weighted sum, and shallow beginner count."""
    cfg = ctx.config
    resume_cfg = cfg.resume_scoring
    thresholds = cfg.penalty_thresholds
    matcher = ctx.skill_matcher

    ai_count = 0
    score_sum = 0.0
    beginner_count = 0

    for skill in skills:
        base = matcher.match(skill.get("normalized_name") or skill["name"])
        if base <= 0:
            continue

        proficiency = skill.get("proficiency", "beginner")
        prof_mult = cfg.proficiency_multipliers.get(proficiency, 0.4)
        endorse_boost = min(
            skill.get("endorsements", 0) / resume_cfg.endorsement_divisor,
            resume_cfg.endorsement_cap,
        )
        duration_boost = min(
            skill.get("duration_months", 0) / resume_cfg.duration_divisor,
            resume_cfg.duration_cap,
        )
        score_sum += base * prof_mult * (1.0 + endorse_boost + duration_boost)
        ai_count += 1

        if (
            proficiency == "beginner"
            and skill.get("endorsements", 0) <= thresholds.beginner_max_endorsements
            and skill.get("duration_months", 0) < thresholds.beginner_max_duration_months
        ):
            beginner_count += 1

    return ai_count, score_sum, beginner_count


def _career_metrics(
    record: CandidateRecord, ctx: ScoringContext
) -> tuple[int, int, int, bool]:
    """Compute career tenure, AI months, gaps, and timeline flags."""
    reference_date = ctx.config.reference_date
    matcher = ctx.skill_matcher
    title_matcher = ctx.title_matcher

    total_months = 0
    ai_months = 0
    gap_months = 0
    timeline_issue = False
    previous_end: date | None = None
    roles = record.career_history
    if len(roles) > 1:
        roles = sorted(roles, key=lambda role: role.get("start") or date.min)

    for role in roles:
        months = max(int(role.get("duration_months", 0)), 0)
        total_months += months

        role_text = f"{role.get('title', '')} {role.get('description', '')}".lower()
        if title_matcher.text_has_ai_keyword(role_text) or matcher.text_contains_skill(
            role_text
        ):
            ai_months += months

        start = role.get("start")
        end = role.get("end") if role.get("end") else reference_date

        if start and end and end < start:
            timeline_issue = True

        if start and previous_end and start > previous_end:
            gap_months += max((start - previous_end).days // 30, 0)

        if end:
            previous_end = end
        elif start:
            previous_end = start

    return total_months, ai_months, gap_months, timeline_issue


def _education_score(
    record: CandidateRecord, ctx: ScoringContext
) -> tuple[float, str]:
    """Score education from field of study and degree only."""
    edu_cfg = ctx.config.education_scoring
    best_score = 0.0
    best_field = ""

    for edu in record.education:
        field_name = normalize_text(edu.get("field_of_study", ""))
        degree = normalize_text(edu.get("degree", ""))

        field_score = 0.0
        for key, value in ctx.education_field_items:
            if key in field_name:
                field_score = max(field_score, value)

        degree_score = 0.0
        for key, value in ctx.degree_level_items:
            if key in degree:
                degree_score = max(degree_score, value)

        combined = (
            edu_cfg.field_weight * field_score + edu_cfg.degree_weight * degree_score
        )
        if combined > best_score:
            best_score = combined
            best_field = str(edu.get("field_of_study", ""))

    return best_score, best_field


def _normalize_metric(
    value: float, spec: BehavioralNormSpec | None, default: float = 0.0
) -> float:
    if spec is None:
        return default
    return normalize_linear(
        value,
        low=spec.low,
        high=spec.high,
        invert=spec.invert,
    )


def _compute_behavioral_metrics(
    record: CandidateRecord, ctx: ScoringContext, days_since_active: int
) -> dict[str, float]:
    """Normalize all documented Redrob behavioral signals once."""
    cfg = ctx.config
    signals = record.redrob_signals
    norms = cfg.behavioral_normalization
    reference_date = cfg.reference_date

    signup = signals.get("signup")
    signup_days = (reference_date - signup).days if signup else 0

    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = safe_float(salary.get("min"), 0.0)
    salary_max = safe_float(salary.get("max"), 0.0)
    salary_mid = (salary_min + salary_max) / 2.0 if salary_max >= salary_min else 0.0

    github_raw = safe_float(signals.get("github_activity_score"), -1.0)
    github_spec = norms.get("github_activity_score")
    if github_raw < 0:
        github_score = (
            github_spec.missing_value if github_spec and github_spec.missing_value is not None else 0.3
        )
    else:
        github_score = _normalize_metric(github_raw, github_spec)

    offer_rate = safe_float(signals.get("offer_acceptance_rate"), -1.0)
    offer_spec = norms.get("offer_acceptance_rate")
    if offer_rate < 0:
        offer_score = (
            offer_spec.missing_value
            if offer_spec and offer_spec.missing_value is not None
            else 0.5
        )
    else:
        offer_score = clamp01(offer_rate)

    work_mode = normalize_text(signals.get("preferred_work_mode", ""))
    work_mode_score = cfg.preferred_work_mode_scores.get(work_mode, 0.5)

    assessments = signals.get("skill_assessment_scores", {})
    assessment_values: list[float] = []
    if isinstance(assessments, dict):
        for skill_name, score in assessments.items():
            if ctx.skill_matcher.match(str(skill_name)) > 0:
                assessment_values.append(safe_float(score, 0.0))
        if not assessment_values:
            assessment_values = [
                safe_float(value, 0.0) for value in assessments.values()
            ]
    assessment_mean = (
        sum(assessment_values) / len(assessment_values) if assessment_values else 0.0
    )

    return {
        "profile_completeness_score": _normalize_metric(
            safe_float(signals.get("profile_completeness_score"), 0.0),
            norms.get("profile_completeness_score"),
        ),
        "signup_tenure": _normalize_metric(
            float(signup_days), norms.get("signup_tenure_days")
        ),
        "activity_recency": _normalize_metric(
            float(days_since_active), norms.get("activity_recency_days")
        ),
        "open_to_work_flag": 1.0 if signals.get("open_to_work_flag") else 0.0,
        "profile_views_received_30d": _normalize_metric(
            float(safe_int(signals.get("profile_views_received_30d"), 0)),
            norms.get("profile_views_received_30d"),
        ),
        "applications_submitted_30d": _normalize_metric(
            float(safe_int(signals.get("applications_submitted_30d"), 0)),
            norms.get("applications_submitted_30d"),
        ),
        "recruiter_response_rate": clamp01(
            safe_float(signals.get("recruiter_response_rate"), 0.0)
        ),
        "avg_response_time_hours": _normalize_metric(
            safe_float(signals.get("avg_response_time_hours"), 0.0),
            norms.get("avg_response_time_hours"),
        ),
        "skill_assessment_scores": _normalize_metric(
            assessment_mean, norms.get("skill_assessment_scores")
        ),
        "connection_count": _normalize_metric(
            float(safe_int(signals.get("connection_count"), 0)),
            norms.get("connection_count"),
        ),
        "endorsements_received": _normalize_metric(
            float(safe_int(signals.get("endorsements_received"), 0)),
            norms.get("endorsements_received"),
        ),
        "notice_period_days": _normalize_metric(
            float(safe_int(signals.get("notice_period_days"), 0)),
            norms.get("notice_period_days"),
        ),
        "expected_salary_range_inr_lpa": _normalize_metric(
            salary_mid, norms.get("expected_salary_mid_lpa")
        ),
        "preferred_work_mode": work_mode_score,
        "willing_to_relocate": 1.0 if signals.get("willing_to_relocate") else 0.0,
        "github_activity_score": github_score,
        "search_appearance_30d": _normalize_metric(
            float(safe_int(signals.get("search_appearance_30d"), 0)),
            norms.get("search_appearance_30d"),
        ),
        "saved_by_recruiters_30d": _normalize_metric(
            float(safe_int(signals.get("saved_by_recruiters_30d"), 0)),
            norms.get("saved_by_recruiters_30d"),
        ),
        "interview_completion_rate": clamp01(
            safe_float(signals.get("interview_completion_rate"), 0.0)
        ),
        "offer_acceptance_rate": offer_score,
        "verified_email": 1.0 if signals.get("verified_email") else 0.0,
        "verified_phone": 1.0 if signals.get("verified_phone") else 0.0,
        "linkedin_connected": 1.0 if signals.get("linkedin_connected") else 0.0,
    }


def clamp01(value: float) -> float:
    """Clamp a ratio to [0, 1]."""
    return clamp(value, 0.0, 1.0)


def extract_features(record: CandidateRecord, ctx: ScoringContext) -> CandidateFeatures:
    """Extract all ranking features for a single candidate."""
    cfg = ctx.config
    profile = record.profile
    thresholds = cfg.penalty_thresholds

    skills = _dedupe_skills(record.skills, cfg.proficiency_rank)
    ai_count, ai_skill_sum, beginner_count = _compute_ai_skill_metrics(skills, ctx)
    career_months, ai_months, gap_months, timeline_issue = _career_metrics(
        record, ctx
    )

    current_title = str(profile.get("current_title", ""))
    title_is_ai = ctx.title_matcher.is_ai_relevant(current_title)
    title_is_irrelevant = ctx.title_matcher.is_irrelevant(current_title)

    years = safe_float(profile.get("years_of_experience"), 0.0)
    yoe_mismatch = False
    if career_months > 0:
        if abs((career_months / 12.0) - years) > thresholds.yoe_mismatch_years:
            yoe_mismatch = True

    education_score, education_field = _education_score(record, ctx)

    headline = str(profile.get("headline", ""))
    summary = str(profile.get("summary", ""))
    profile_text = normalize_text(f"{headline} {summary}")

    keyword_hits, profile_has_skills = ctx.skill_matcher.profile_keyword_stats(
        profile_text
    )
    word_count = max(profile_text.count(" ") + 1, 1)
    keyword_density = keyword_hits / word_count

    has_essential = bool(
        current_title and summary and skills and record.career_history
    )

    last_active = record.redrob_signals.get("last_active")
    if last_active:
        days_since_active = max((cfg.reference_date - last_active).days, 0)
    else:
        days_since_active = thresholds.default_inactive_days

    is_inactive = days_since_active > thresholds.inactive_days

    behavioral_metrics = _compute_behavioral_metrics(
        record, ctx, days_since_active
    )
    response_rate = behavioral_metrics["recruiter_response_rate"]

    numeric_id = int(record.candidate_id[5:])

    return CandidateFeatures(
        candidate_id=record.candidate_id,
        numeric_id=numeric_id,
        current_title=current_title,
        years_of_experience=years,
        headline=headline,
        summary=summary,
        skill_details=skills,
        ai_core_skill_count=ai_count,
        ai_skill_score_sum=ai_skill_sum,
        certification_count=len(record.certifications),
        career_months_total=career_months,
        ai_role_months=ai_months,
        career_gap_months=gap_months,
        title_is_ai_relevant=title_is_ai,
        title_is_irrelevant=title_is_irrelevant,
        education_best_score=education_score,
        education_field=education_field,
        profile_text=profile_text,
        profile_has_skills=profile_has_skills,
        keyword_density=keyword_density,
        has_essential_fields=has_essential,
        timeline_issue=timeline_issue,
        yoe_mismatch=yoe_mismatch,
        beginner_ai_skill_count=beginner_count,
        is_inactive=is_inactive,
        behavioral_metrics=behavioral_metrics,
        recruiter_response_rate=response_rate,
    )
