"""
StrideIQ — CV/Wearable: Receiver
Despite the name (kept consistent with the folder's existing pattern),
this is NOT a live BLE receiver — the CV pipeline processes a recorded
video after the fact, so there's no live stream to receive here. This
is the lookup step: given a run_id (or a video filepath, via the runs
table's video_filepath column), find the matching wearable data.
"""

import logging
from typing import Optional

from backend.database_manager import DatabaseManager
from cv.wearable.processor import summarize_run_features

logger = logging.getLogger("strideiq.cv.wearable.receiver")


def get_wearable_data_for_video(video_filepath: str, db: DatabaseManager) -> Optional[dict]:
    """
    Looks up which run a given filmed video belongs to (via runs.video_filepath,
    set by database_manager.set_sync_marker() when the session was logged),
    then returns that run's summarized wearable data.
    """
    all_runs = db.get_all_runs()
    matching = [r for r in all_runs if r["video_filepath"] == video_filepath]

    if not matching:
        logger.warning(f"No run found with video_filepath={video_filepath}")
        return None

    if len(matching) > 1:
        logger.warning(
            f"Multiple runs reference {video_filepath} — using the most recent. "
            f"This shouldn't happen if each filmed session gets a unique filepath."
        )

    run = matching[0]
    return summarize_run_features(run["run_id"], db)


def get_wearable_data_for_run(run_id: int, db: DatabaseManager) -> Optional[dict]:
    """Direct lookup when you already have the run_id, skipping the filepath match."""
    return summarize_run_features(run_id, db)