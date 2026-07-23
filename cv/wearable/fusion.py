"""
StrideIQ — CV/Wearable Fusion

Combines the CV analysis with wearable-derived metrics into the
context consumed by CoachingEngine.generate().

Responsibilities
----------------
- Merge CV scores
- Merge wearable summary
- Keep both sources independent
- Never query the database directly

NOT responsible for:
- BLE
- SQLite lookups
- Signal processing
"""

from __future__ import annotations

import logging
from typing import Optional

from cv.models import CVPipelineOutput

logger = logging.getLogger("strideiq.cv.wearable.fusion")


class WearableFusion:
    """
    Merges wearable metrics and CV outputs into one context
    for the Coaching Engine.
    """

    def build_context(
        self,
        cv_output: CVPipelineOutput,
        wearable_summary: Optional[dict] = None,
    ) -> dict:
        """
        Creates the context dictionary consumed by
        CoachingEngine.generate().

        Parameters
        ----------
        cv_output
            Output of the CV pipeline.

        wearable_summary
            Output from wearable/aggregator.py.
            May be None if no wearable data exists.
        """

        # -----------------------------
        # Extract CV metric scores
        # -----------------------------

        cv_scores = {}

        if cv_output is not None and cv_output.report:

            for metric in cv_output.report.metrics:

                cv_scores[metric.metric] = {

                    "score": metric.score,

                    "grade": metric.grade,

                    "confidence": metric.confidence,

                }

        # -----------------------------
        # No wearable data
        # -----------------------------

        if wearable_summary is None:

            logger.info(
                "Fusion running with CV-only data."
            )

            wearable_summary = {}

        # -----------------------------
        # Final context
        # -----------------------------

        return {

            "wearable": wearable_summary,

            "cv_scores": cv_scores,

            "has_cv": bool(cv_scores),

            "has_wearables": bool(wearable_summary),

            "has_both_sources": bool(
                cv_scores and wearable_summary
            ),

        }