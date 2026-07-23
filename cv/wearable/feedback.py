"""
StrideIQ — CV/Wearable: Feedback
Takes the finished CVPipelineOutput (already fused with wearable context
via fusion.py) and stores it as a recommendation entry, same table your
live wearable-only recommendations use — so the dashboard's Coaching
section can display CV-derived insights alongside sensor-only ones
without needing a separate display path.

Unlike the wearable-only feedback loop (backend/mqtt/feedback_loop.py),
this does NOT trigger real-time actuation — CV processing happens after
the run is already over, there's no live device to buzz.
"""

import logging
import time

from backend.database_manager import DatabaseManager
from cv.models import CVPipelineOutput

logger = logging.getLogger("strideiq.cv.wearable.feedback")


def store_cv_report(cv_output: CVPipelineOutput, run_id: int, db: DatabaseManager):
    """
    Writes one recommendation row per metric in the CV report, plus an
    overall summary row. Mirrors recommend.py's insert_recommendation
    pattern so CV and wearable-only recommendations share one table
    and one dashboard view.
    """
    if not cv_output.report:
        logger.warning(f"No CV report to store for run {run_id} (pipeline returned no report)")
        return

    report = cv_output.report
    now = time.time()

    db.insert_recommendation( 
        run_id=run_id,
        timestamp=now,
        related_metric="cv_overall",
        recommendation_text=f"Overall CV score: {report.overall_score} ({report.overall_grade})",
        llm_explanation=getattr(report, "llm_summary", None), 
    )

    for metric_result in report.metrics:
        db.insert_recommendation(
            run_id=run_id,
            timestamp=now,
            related_metric=str(metric_result.metric),
            recommendation_text=metric_result.explanation, 
            llm_explanation=None,  # per-metric LLM phrasing isn't generated separately yet
        )

    logger.info(f"Stored CV report for run {run_id}: {len(report.metrics)} metric entries")