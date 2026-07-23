"""
StrideIQ — CV/Wearable: Processor
Pulls wearable feature rows for a given run and summarizes them into
the shape CoachingEngine.sensor_fusion_rules() expects. This is post-hoc
(CV runs on a recorded video matched to a run_id), not live BLE — the
video and the wearable session both belong to the same `runs` row,
since CV sessions run both sensors and camera simultaneously per the
project's design.
"""

import logging
from typing import Optional

from backend.database_manager import DatabaseManager  

logger = logging.getLogger("strideiq.cv.wearable.processor")


def summarize_run_features(run_id: int, db: DatabaseManager) -> Optional[dict]:
    """
    Returns a summary dict of the wearable-derived metrics for a run —
    averaged/aggregated across all windows, since the CV report is a
    single per-video result, not per-window like the live dashboard.
    Returns None if no feature rows exist for this run (e.g. waist/feet
    weren't connected during this particular session).
    """
    features = db.get_features(run_id)
    if not features:
        logger.warning(f"No wearable feature rows found for run {run_id}")
        return None

    rows = [dict(f) for f in features]

    def _avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "run_id": run_id,
        "cadence": _avg("cadence"),
        "vertical_oscillation": _avg("vertical_oscillation"),
        "running_smoothness": _avg("running_smoothness"),
        "pelvic_tilt": _avg("pelvic_tilt"),
        "pelvic_rotation": _avg("pelvic_rotation"),
        "impact_left": _avg("impact_left"),
        "impact_right": _avg("impact_right"),
        "symmetry_score": _avg("symmetry_score"),
        "cadence_variability": _avg("cadence_variability"),
        "window_count": len(rows),
    }