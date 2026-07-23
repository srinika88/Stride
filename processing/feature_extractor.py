"""
StrideIQ Feature Extraction Pipeline

Master orchestrator for feature extraction.

Pipeline

Raw Sensor Data
        │
        ▼
Windowing
        │
        ├─────────────┐
        ▼             ▼
Foot Features     Waist Features
        │             │
        └──────┬──────┘
               ▼
        Feature Dictionary
               ▼
          SQLite Features Table

This file intentionally contains NO signal-processing
algorithms.

Those live in:

processing/features_foot.py

processing/features_waist.py
"""

import logging

from backend import config
from backend.database_manager import DatabaseManager

from processing.features_foot import (
    compute_footstrike_features,
    compute_symmetry_score,
    compute_cadence,
)

from processing.features_waist import (
    compute_waist_features,
)

logger = logging.getLogger("StrideIQ-FeatureExtractor")


# ==========================================================
# Window helper
# ==========================================================

def get_window(readings, start, end):
    """
    Returns all sensor readings inside one window.
    """

    return [
        r
        for r in readings
        if start <= r["timestamp"] < end
    ]


# ==========================================================
# Main pipeline
# ==========================================================

def process_run_windows(
    run_id: int,
    db: DatabaseManager
):
    """
    Computes all engineered features for one run.

    Returns
    -------
    Number of feature windows written.
    """

    logger.info(
        f"Processing run {run_id}"
    )

    # ------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------

    left_raw = db.get_raw_readings(
        run_id,
        "foot_left"
    )

    right_raw = db.get_raw_readings(
        run_id,
        "foot_right"
    )

    waist_raw = db.get_raw_readings(
        run_id,
        "waist"
    )

    if not left_raw:

        raise ValueError(
            "Missing left foot data"
        )

    if not right_raw:

        raise ValueError(
            "Missing right foot data"
        )

    if not waist_raw:

        logger.warning(
            "Waist data missing. "
            "Waist-derived features will be NULL."
        )

    # ------------------------------------------------------
    # Determine run duration
    # ------------------------------------------------------

    timestamps = []

    timestamps.extend(
        [
            r["timestamp"]
            for r in left_raw
        ]
    )

    timestamps.extend(
        [
            r["timestamp"]
            for r in right_raw
        ]
    )

    if waist_raw:

        timestamps.extend(
            [
                r["timestamp"]
                for r in waist_raw
            ]
        )

    run_start = min(timestamps)

    run_end = max(timestamps)

    # ------------------------------------------------------
    # Sliding windows
    # ------------------------------------------------------

    window_size = config.WINDOW_SIZE

    step = (
        window_size
        -
        config.WINDOW_OVERLAP
    )

    feature_rows = []

    t = run_start

    while t + window_size <= run_end:

        # ----------------------------------------------
        # Sensor windows
        # ----------------------------------------------

        left_window = get_window(
            left_raw,
            t,
            t + window_size
        )

        right_window = get_window(
            right_raw,
            t,
            t + window_size
        )

        waist_window = get_window(
            waist_raw,
            t,
            t + window_size
        )

        # ----------------------------------------------
        # Foot features
        # ----------------------------------------------

        left_features = (
            compute_footstrike_features(
                left_window
            )
            if left_window
            else {
                "impact_intensity": None,
                "footstrike_timing": None,
                "strike_count": 0,
            }
        )

        right_features = (
            compute_footstrike_features(
                right_window
            )
            if right_window
            else {
                "impact_intensity": None,
                "footstrike_timing": None,
                "strike_count": 0,
            }
        )

        cadence = compute_cadence(

            left_features["strike_count"],

            right_features["strike_count"],

            window_size

        )

        symmetry = compute_symmetry_score(

            left_features,

            right_features

        )

        # ----------------------------------------------
        # Waist features
        # ----------------------------------------------

        if waist_window:

            waist_features = compute_waist_features(
                waist_window
            )

        else:

            waist_features = {

                "vertical_oscillation": None,

                "running_smoothness": None,

                "pelvic_tilt": None,

                "pelvic_rotation": None,

                "mean_accel": None,

                "variance_accel": None,

                "rms": None,

                "sma": None,

                "dominant_freq": None,

            }

        # ----------------------------------------------
        # Merge
        # ----------------------------------------------

        feature_rows.append(

            {

                "run_id": run_id,

                "window_start": t,

                "window_end": t + window_size,

                # -------------------

                "cadence": cadence,

                "vertical_oscillation":
                    waist_features["vertical_oscillation"],

                "running_smoothness":
                    waist_features["running_smoothness"],

                "pelvic_tilt":
                    waist_features["pelvic_tilt"],

                "pelvic_rotation":
                    waist_features["pelvic_rotation"],

                # -------------------

                "impact_left":
                    left_features["impact_intensity"],

                "impact_right":
                    right_features["impact_intensity"],

                "footstrike_timing_left":
                    left_features["footstrike_timing"],

                "footstrike_timing_right":
                    right_features["footstrike_timing"],

                "symmetry_score":
                    symmetry,

                # -------------------

                "mean_accel":
                    waist_features["mean_accel"],

                "variance_accel":
                    waist_features["variance_accel"],

                "rms":
                    waist_features["rms"],

                "sma":
                    waist_features["sma"],

                "dominant_freq":
                    waist_features["dominant_freq"],

            }

        )

        t += step

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    db.insert_features(
        feature_rows
    )

    logger.info(

        f"{len(feature_rows)} windows saved."

    )

    return len(feature_rows)


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    db = DatabaseManager()

    runs = db.get_all_runs()

    if not runs:

        print("No runs found.")

    else:

        latest = runs[0]["run_id"]

        n = process_run_windows(

            latest,

            db

        )

        print(

            f"Processed {n} windows."

        )