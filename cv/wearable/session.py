"""
StrideIQ — CV Pipeline Session Entrypoint

Runs the complete analysis pipeline for one filmed running session.

Flow
----
Video
    ↓
Retrieve wearable summary
    ↓
run_cv_pipeline()
    ↓
Predictions
    ↓
Scoring
    ↓
Coaching (CV + Wearable Fusion)
    ↓
Database
"""

from __future__ import annotations

import logging

from backend.database_manager import DatabaseManager

from cv.pipeline import run_cv_pipeline
from cv.wearable.fusion import build_fused_context
from cv.wearable.feedback import store_cv_report

logger = logging.getLogger("strideiq.cv.session")


def process_filmed_session(
    video_path: str,
    run_id: int,
    db: DatabaseManager | None = None,
):
    """
    Complete entrypoint for a filmed running session.

    Steps
    -----
    1. Load wearable summary for this run.
    2. Run CV pipeline.
    3. Fuse wearable + CV.
    4. Generate coaching.
    5. Store report.
    """

    if db is None:
        db = DatabaseManager()

    logger.info(
        "Processing session | run_id=%s | video=%s",
        run_id,
        video_path,
    )

    try:

        # --------------------------------------------------
        # Build wearable context FIRST
        # --------------------------------------------------

        wearable_data = build_fused_context(
            cv_output=None,
            run_id=run_id,
            db=db,
        )

        # --------------------------------------------------
        # Run CV Pipeline
        #
        # Once pipeline.py officially accepts wearable_data,
        # simply uncomment the second version below.
        # --------------------------------------------------

        cv_output = run_cv_pipeline(video_path)

        # Future:
        #
        # cv_output = run_cv_pipeline(
        #     video_path,
        #     wearable_data=wearable_data,
        # )

        if cv_output.report is None:

            logger.warning(
                "No CV report generated for %s",
                video_path,
            )

            return cv_output

        # --------------------------------------------------
        # Rebuild fused context now that CV scores exist
        # --------------------------------------------------

        wearable_data = build_fused_context(
            cv_output=cv_output,
            run_id=run_id,
            db=db,
        )

        # --------------------------------------------------
        # Generate fused coaching
        # --------------------------------------------------

        if wearable_data.get("has_both_sources", False):

            from cv.coaching_engine import CoachingEngine

            coach = CoachingEngine()

            findings = coach.generate(
                metrics=cv_output.report.metrics,
                wearable_data=wearable_data,
            )

            # ------------------------------------------
            # Convert findings into report fields
            # ------------------------------------------

            cv_output.report.strengths = [
                f.summary
                for f in findings
                if getattr(f, "is_positive", False)
            ]

            cv_output.report.improvements = [
                f.summary
                for f in findings
                if not getattr(f, "is_positive", False)
            ]

            logger.info(
                "Run %s: generated fused coaching (%d findings)",
                run_id,
                len(findings),
            )

        else:

            logger.info(
                "Run %s: wearable data unavailable (CV-only report)",
                run_id,
            )

        # --------------------------------------------------
        # Store report
        # --------------------------------------------------

        store_cv_report(
            cv_output,
            run_id,
            db,
        )

        logger.info(
            "Run %s successfully processed.",
            run_id,
        )

        return cv_output

    except Exception:

        logger.exception(
            "Failed to process run %s",
            run_id,
        )

        raise


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:

        print(
            "Usage: python -m cv.session <video_path> <run_id>"
        )

        sys.exit(1)

    process_filmed_session(
        sys.argv[1],
        int(sys.argv[2]),
    ) 