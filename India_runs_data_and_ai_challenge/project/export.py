"""Export ranked candidates to submission CSV."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ranking import RankedCandidate

logger = logging.getLogger(__name__)

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def export_submission(
    ranked: list[RankedCandidate],
    output_path: Path,
) -> None:
    """
    Write the top-ranked candidates to a competition-compliant CSV.

    Scores are written in non-increasing rank order. Equal scores retain
    candidate_id ascending tie-break from the ranking step.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(ranked, key=lambda item: (-item.score, item.candidate_id))

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REQUIRED_HEADER)
        for rank, candidate in enumerate(ordered, start=1):
            writer.writerow(
                [
                    candidate.candidate_id,
                    rank,
                    f"{candidate.score:.4f}",
                    candidate.reasoning,
                ]
            )

    logger.info("Exported %d ranked candidates to %s", len(ordered), output_path)
