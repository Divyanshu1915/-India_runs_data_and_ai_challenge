"""Parse and validate candidate records against the provided schema."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import normalize_text, parse_date, safe_float, safe_int

logger = logging.getLogger(__name__)

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
REQUIRED_TOP_LEVEL = (
    "candidate_id",
    "profile",
    "career_history",
    "education",
    "skills",
    "redrob_signals",
)


@dataclass
class CandidateRecord:
    """Validated and normalized candidate record."""

    candidate_id: str
    profile: dict[str, Any]
    career_history: list[dict[str, Any]]
    education: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    certifications: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    redrob_signals: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)
    validation_warnings: list[str] = field(default_factory=list)


def load_schema_required_fields(schema_path: Path) -> tuple[str, ...]:
    """Read required top-level fields from candidate_schema.json."""
    if not schema_path.is_file():
        logger.warning("Schema file not found; using built-in required fields.")
        return REQUIRED_TOP_LEVEL

    with open(schema_path, encoding="utf-8") as handle:
        schema = json.load(handle)

    required = schema.get("required")
    if not isinstance(required, list):
        raise ValueError("Schema 'required' field must be a list.")
    return tuple(str(item) for item in required)


def _normalize_skills(skills: Any) -> list[dict[str, Any]]:
    if not isinstance(skills, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in skills:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": str(item.get("name", "")).strip(),
                "normalized_name": normalize_text(str(item.get("name", ""))),
                "proficiency": normalize_text(str(item.get("proficiency", "beginner")))
                or "beginner",
                "endorsements": safe_int(item.get("endorsements"), 0),
                "duration_months": safe_int(item.get("duration_months"), 0),
            }
        )
    return normalized


def _normalize_education(education: Any) -> list[dict[str, Any]]:
    if not isinstance(education, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in education:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "institution": str(item.get("institution", "")).strip(),
                "degree": str(item.get("degree", "")).strip(),
                "field_of_study": str(item.get("field_of_study", "")).strip(),
                "start_year": safe_int(item.get("start_year"), 0),
                "end_year": safe_int(item.get("end_year"), 0),
                "grade": item.get("grade"),
                "tier": str(item.get("tier", "unknown")).strip(),
            }
        )
    return normalized


def _normalize_career_history(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        start_raw = str(item.get("start_date", "")).strip()
        end_raw = item.get("end_date")
        end_str = str(end_raw).strip() if end_raw else ""
        normalized.append(
            {
                "company": str(item.get("company", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "start_date": start_raw,
                "end_date": end_raw,
                "start": parse_date(start_raw),
                "end": parse_date(end_str) if end_str else None,
                "duration_months": safe_int(item.get("duration_months"), 0),
                "is_current": bool(item.get("is_current", False)),
                "industry": str(item.get("industry", "")).strip(),
                "company_size": str(item.get("company_size", "")).strip(),
                "description": str(item.get("description", "")).strip(),
            }
        )
    return normalized


def _normalize_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {
            "headline": "",
            "summary": "",
            "years_of_experience": 0.0,
            "current_title": "",
            "current_company": "",
            "current_industry": "",
            "location": "",
            "country": "",
        }
    return {
        "anonymized_name": str(profile.get("anonymized_name", "")).strip(),
        "headline": str(profile.get("headline", "")).strip(),
        "summary": str(profile.get("summary", "")).strip(),
        "location": str(profile.get("location", "")).strip(),
        "country": str(profile.get("country", "")).strip(),
        "years_of_experience": safe_float(profile.get("years_of_experience"), 0.0),
        "current_title": str(profile.get("current_title", "")).strip(),
        "current_company": str(profile.get("current_company", "")).strip(),
        "current_company_size": str(profile.get("current_company_size", "")).strip(),
        "current_industry": str(profile.get("current_industry", "")).strip(),
    }


def _normalize_signals(signals: Any) -> dict[str, Any]:
    if not isinstance(signals, dict):
        return {}
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if not isinstance(salary, dict):
        salary = {}
    assessments = signals.get("skill_assessment_scores", {})
    if not isinstance(assessments, dict):
        assessments = {}
    return {
        "profile_completeness_score": safe_float(
            signals.get("profile_completeness_score"), 0.0
        ),
        "signup_date": str(signals.get("signup_date", "")).strip(),
        "signup": parse_date(str(signals.get("signup_date", "")).strip()),
        "last_active_date": str(signals.get("last_active_date", "")).strip(),
        "last_active": parse_date(str(signals.get("last_active_date", "")).strip()),
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "profile_views_received_30d": safe_int(
            signals.get("profile_views_received_30d"), 0
        ),
        "applications_submitted_30d": safe_int(
            signals.get("applications_submitted_30d"), 0
        ),
        "recruiter_response_rate": safe_float(
            signals.get("recruiter_response_rate"), 0.0
        ),
        "avg_response_time_hours": safe_float(
            signals.get("avg_response_time_hours"), 0.0
        ),
        "skill_assessment_scores": {
            str(k): safe_float(v, 0.0) for k, v in assessments.items()
        },
        "connection_count": safe_int(signals.get("connection_count"), 0),
        "endorsements_received": safe_int(signals.get("endorsements_received"), 0),
        "notice_period_days": safe_int(signals.get("notice_period_days"), 0),
        "expected_salary_range_inr_lpa": {
            "min": safe_float(salary.get("min"), 0.0),
            "max": safe_float(salary.get("max"), 0.0),
        },
        "preferred_work_mode": str(signals.get("preferred_work_mode", "")).strip(),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "github_activity_score": safe_float(signals.get("github_activity_score"), -1.0),
        "search_appearance_30d": safe_int(signals.get("search_appearance_30d"), 0),
        "saved_by_recruiters_30d": safe_int(signals.get("saved_by_recruiters_30d"), 0),
        "interview_completion_rate": safe_float(
            signals.get("interview_completion_rate"), 0.0
        ),
        "offer_acceptance_rate": safe_float(
            signals.get("offer_acceptance_rate"), -1.0
        ),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
    }


def parse_candidate(
    raw: dict[str, Any],
    required_fields: tuple[str, ...] | None = None,
) -> CandidateRecord | None:
    """
    Validate and normalize a raw candidate dict.

    Returns None for records that fail hard validation.
    """
    required = required_fields or REQUIRED_TOP_LEVEL
    warnings: list[str] = []

    candidate_id = str(raw.get("candidate_id", "")).strip()
    if not CANDIDATE_ID_PATTERN.match(candidate_id):
        logger.debug("Invalid candidate_id skipped: %r", candidate_id)
        return None

    for field_name in required:
        if field_name not in raw or raw[field_name] is None:
            warnings.append(f"Missing required field: {field_name}")

    profile = _normalize_profile(raw.get("profile"))
    career_history = _normalize_career_history(raw.get("career_history"))
    education = _normalize_education(raw.get("education"))
    skills = _normalize_skills(raw.get("skills"))
    signals = _normalize_signals(raw.get("redrob_signals"))

    certifications = raw.get("certifications", [])
    if not isinstance(certifications, list):
        certifications = []

    languages = raw.get("languages", [])
    if not isinstance(languages, list):
        languages = []

    if not career_history:
        warnings.append("Empty career_history.")

    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile,
        career_history=career_history,
        education=education,
        skills=skills,
        certifications=certifications,
        languages=languages,
        redrob_signals=signals,
        raw=raw,
        validation_warnings=warnings,
    )
