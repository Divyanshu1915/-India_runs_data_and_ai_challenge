#!/usr/bin/env python3
"""Main entry point for the Senior AI Engineer candidate ranking pipeline."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from config import build_scoring_context, load_config
from export import export_submission
from feature_engineering import extract_features
from loader import iter_candidates
from parser import load_schema_required_fields, parse_candidate
from ranking import build_reasoning, compute_final_score, select_top_candidates, update_top_heap
from utils import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(config_path: Path | None = None) -> Path:
    """Execute the full ranking pipeline and return the output CSV path."""
    config = load_config(config_path)
    ctx = build_scoring_context(config)
    setup_logging()

    logger.info("Starting candidate ranking pipeline")
    logger.info("Candidates file: %s", config.paths.candidates_file)
    logger.info("Output file: %s", config.paths.output_file)

    required_fields = load_schema_required_fields(config.paths.schema_file)
    top_heap: list[tuple[float, int, str, str]] = []
    parsed_count = 0
    skipped_count = 0
    top_n = config.top_n

    start = time.perf_counter()

    for raw in iter_candidates(config.paths.candidates_file):
        record = parse_candidate(raw, required_fields=required_fields)
        if record is None:
            skipped_count += 1
            continue

        parsed_count += 1
        features = extract_features(record, ctx)
        score = compute_final_score(features, ctx)
        reasoning = build_reasoning(features, ctx)
        update_top_heap(
            top_heap, score, features.numeric_id, features.candidate_id, reasoning, top_n
        )

        if parsed_count % 10000 == 0:
            logger.info("Scored %d candidates...", parsed_count)

    if parsed_count < top_n:
        raise ValueError(
            f"Insufficient candidates for top {top_n}; only {parsed_count} scored."
        )

    logger.info(
        "Scoring complete: %d candidates scored, %d skipped (%.2fs).",
        parsed_count,
        skipped_count,
        time.perf_counter() - start,
    )

    top_candidates = select_top_candidates(top_heap, ctx)
    export_submission(top_candidates, config.paths.output_file)

    validator = Path(__file__).resolve().parent.parent / "validate_submission.py"
    if validator.is_file():
        result = subprocess.run(
            [sys.executable, str(validator), str(config.paths.output_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Submission validation passed.")
        else:
            logger.error(
                "Submission validation failed:\n%s", result.stdout + result.stderr
            )
            raise ValueError("Generated submission failed validation.")

    elapsed = time.perf_counter() - start
    logger.info("Pipeline finished in %.2f seconds.", elapsed)
    logger.info("Submission written to %s", config.paths.output_file)

    return config.paths.output_file


def main() -> None:
    """CLI entry point."""
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        output_path = run_pipeline(config_arg)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Submission generated: {output_path}")


if __name__ == "__main__":
    main()
