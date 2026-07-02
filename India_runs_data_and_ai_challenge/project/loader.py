"""Stream candidate records from JSONL or gzipped JSONL files."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TextIO

logger = logging.getLogger(__name__)


def _open_candidate_stream(path: Path) -> TextIO:
    """Open a text stream for plain or gzipped JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"Candidate dataset not found: {path}")

    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        logger.info("Opening gzipped candidate file: %s", path)
        return gzip.open(path, mode="rt", encoding="utf-8")  # type: ignore[return-value]

    logger.info("Opening candidate file: %s", path)
    return open(path, encoding="utf-8", buffering=1_048_576)


def iter_candidates(path: Path) -> Iterator[dict]:
    """
    Yield parsed candidate records one at a time.

    Skips blank lines and logs malformed JSON without stopping the pipeline.
    """
    line_number = 0
    malformed = 0

    with _open_candidate_stream(path) as handle:
        for line in handle:
            line_number += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                malformed += 1
                logger.warning(
                    "Skipping malformed JSON at line %d: %s", line_number, exc
                )
                continue
            if not isinstance(record, dict):
                malformed += 1
                logger.warning(
                    "Skipping non-object record at line %d.", line_number
                )
                continue
            yield record

    if line_number == 0:
        raise ValueError(f"Candidate dataset is empty: {path}")

    logger.info(
        "Finished loading candidates from %s (%d malformed lines skipped).",
        path,
        malformed,
    )


def count_candidates(path: Path) -> int:
    """Count non-empty lines in the candidate file."""
    count = 0
    with _open_candidate_stream(path) as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count
