"""Shared utilities for logging, normalization, and text processing."""

from __future__ import annotations

import logging
from datetime import date


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to [low, high]."""
    return max(low, min(high, value))


def normalize_linear(
    value: float,
    low: float,
    high: float,
    out_min: float = 0.0,
    out_max: float = 1.0,
    invert: bool = False,
) -> float:
    """Map value from [low, high] to [out_min, out_max]."""
    if high <= low:
        return out_min
    ratio = clamp((value - low) / (high - low), 0.0, 1.0)
    if invert:
        ratio = 1.0 - ratio
    return out_min + ratio * (out_max - out_min)


def parse_date(value: str | None) -> date | None:
    """Parse ISO date strings safely."""
    if not value or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def normalize_text(text: str | None) -> str:
    """Lowercase and collapse whitespace."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert value to float with fallback."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    """Convert value to int with fallback."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class SkillMatcher:
    """Precompiled case-insensitive skill matching with synonym support."""

    __slots__ = ("_exact", "_sorted_keywords", "_weights")

    def __init__(
        self,
        skill_weights: dict[str, float],
        skill_synonyms: dict[str, str],
    ) -> None:
        self._weights = skill_weights
        self._sorted_keywords = tuple(
            sorted(skill_weights.keys(), key=len, reverse=True)
        )
        exact: dict[str, float] = dict(skill_weights)
        for variant, canonical in skill_synonyms.items():
            if canonical in skill_weights:
                exact[variant] = skill_weights[canonical]
        self._exact = exact

    def match(self, skill_name: str) -> float:
        """Return the best configured weight for a skill name."""
        normalized = normalize_text(skill_name)
        if not normalized:
            return 0.0

        exact = self._exact.get(normalized)
        if exact is not None:
            return exact

        best = 0.0
        for keyword in self._sorted_keywords:
            if keyword in normalized or normalized in keyword:
                weight = self._weights[keyword]
                if weight > best:
                    best = weight
        return best

    def profile_keyword_stats(self, text: str) -> tuple[int, bool]:
        """Return keyword hit count and presence in a single pass."""
        lowered = normalize_text(text)
        if not lowered:
            return 0, False
        hits = 0
        for keyword in self._sorted_keywords:
            if keyword in lowered:
                hits += 1
        return hits, hits > 0

    def text_contains_skill(self, text: str) -> bool:
        """Return True when any configured skill keyword appears in text."""
        return self.profile_keyword_stats(text)[1]


class TitleMatcher:
    """Precompiled title relevance checks."""

    __slots__ = ("_ai_keywords", "_irrelevant_keywords")

    def __init__(
        self,
        ai_keywords: tuple[str, ...],
        irrelevant_keywords: tuple[str, ...],
    ) -> None:
        self._ai_keywords = ai_keywords
        self._irrelevant_keywords = irrelevant_keywords

    def is_ai_relevant(self, title: str) -> bool:
        """Return True when the title matches AI role keywords."""
        lowered = title.lower()
        return any(keyword in lowered for keyword in self._ai_keywords)

    def is_irrelevant(self, title: str) -> bool:
        """Return True when the title matches non-AI role keywords."""
        lowered = title.lower()
        return any(keyword in lowered for keyword in self._irrelevant_keywords)

    def text_has_ai_keyword(self, text: str) -> bool:
        """Return True when AI title keywords appear in text."""
        lowered = text.lower()
        return any(keyword in lowered for keyword in self._ai_keywords)
