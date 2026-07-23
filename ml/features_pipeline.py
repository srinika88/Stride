"""
StrideIQ — Combined Feature Pipeline
Runs foot AND waist feature extraction per window and merges them into
one row per window in the features table — this supersedes
features_foot.py's process_run_windows(), which only ever wrote foot
data and left every waist column as None.

Merge precedence, stated explicitly so it's not a silent choice:
- cadence: prefer FOOT-derived (from real footstrike counts — more
  accurate) when foot data exists for the window; fall back to the
  waist's FFT-based estimate only if foot data is missing for that
  window (e.g. a foot node dropped briefly).
- running_smoothness: waist-derived (SPARC) is the only real source —
  features_foot.py never actually computed this, despite the column
  existing; no conflict in practice.
- Everything else: each side only ever supplies its own columns
  (impact/timing/symmetry from feet, pelvic/SPARC/jerk/etc from waist),
  no overlap to resolve.
"""

import logging

from backend import config
from backend.database_manager import DatabaseManager
import processing.features_foot
import processing.features_waist

logger = logging.getLogger("strideiq.features_pipeline")


def _merge_window(foot_feat: dict, waist_feat: dict) -> dict:
    merged = dict(waist_feat)  # start with waist's columns

    # Cadence precedence: foot wins if available
    if foot_feat.get("cadence") is not None:
        merged["cadence"] = foot_feat["cadence"]
    # else: keep waist's cadence (already in merged from waist_feat)

    # Foot-only columns always come from foot_feat
    for key in ["impact_left", "impact_right", "footstrike_timing_left",
                "footstrike_timing_right", "symmetry_score",
                "mean_accel", "variance_accel", "rms", "sma", "dominant_freq"]:
        merged[key] = foot_feat.get(key)

    return merged


def process_run_windows(run_id: int, db: DatabaseManager):
    left_raw = db.get_raw_readings(run_id, "foot_left")
    right_raw = db.get_raw_readings(run_id, "foot_right")

    try:
        waist_raw = db.get_raw_readings(run_id, "waist")
    except ValueError:
        waist_raw = []  # waist node not in this run — pre-integration sessions

    has_feet = bool(left_raw and right_raw)
    has_waist = bool(waist_raw)

    if not has_feet and not has_waist:
        raise ValueError(f"Run {run_id}: no usable sensor data found for any node")

    tilt_offset = 0.0
    if has_waist:
        calib_end = waist_raw[0]["timestamp"] + config.CALIBRATION_DURATION_SEC
        calib_readings = [r for r in waist_raw if r["timestamp"] < calib_end]
        try:
            tilt_offset = processing.features_waist.compute_tilt_baseline(calib_readings)
            logger.info(f"Run {run_id}: tilt baseline = {tilt_offset:.2f} deg (from first {config.CALIBRATION_DURATION_SEC}s)")
        except ValueError as e:
            logger.warning(
                f"Run {run_id}: couldn't compute tilt baseline ({e}) — "
                f"pelvic_tilt will be uncalibrated for this run"
            )
            tilt_offset = 0.0

    all_timestamps = []
    if has_feet:
        all_timestamps += [left_raw[0]["timestamp"], right_raw[0]["timestamp"],
                            left_raw[-1]["timestamp"], right_raw[-1]["timestamp"]]
    if has_waist:
        all_timestamps += [waist_raw[0]["timestamp"], waist_raw[-1]["timestamp"]]

    run_start, run_end = min(all_timestamps), max(all_timestamps)

    window_size = config.WINDOW_SIZE
    step = window_size - config.WINDOW_OVERLAP

    feature_rows = []
    cadence_values_for_run = []  # for computing cadence_variability afterward
    t = run_start

    while t + window_size <= run_end:
        window_end = t + window_size

        # ---- Foot side ----
        if has_feet:
            window_left = [r for r in left_raw if t <= r["timestamp"] < window_end]
            window_right = [r for r in right_raw if t <= r["timestamp"] < window_end]

            left_feat = processing.features_foot.compute_footstrike_features(window_left) if window_left else \
                {"impact_intensity": None, "footstrike_timing": None, "strike_count": 0}
            right_feat = processing.features_foot.compute_footstrike_features(window_right) if window_right else \
                {"impact_intensity": None, "footstrike_timing": None, "strike_count": 0}

            symmetry = processing.features_foot.compute_symmetry_score(left_feat, right_feat)
            cadence = processing.features_foot.compute_cadence(
                left_feat.get("strike_count", 0), right_feat.get("strike_count", 0), window_size
            )

            foot_feat = {
                "cadence": cadence,
                "impact_left": left_feat["impact_intensity"],
                "impact_right": right_feat["impact_intensity"],
                "footstrike_timing_left": left_feat["footstrike_timing"],
                "footstrike_timing_right": right_feat["footstrike_timing"],
                "symmetry_score": symmetry,
                "mean_accel": None, "variance_accel": None, "rms": None,
                "sma": None, "dominant_freq": None,
            }
        else:
            foot_feat = {"cadence": None, "impact_left": None, "impact_right": None,
                         "footstrike_timing_left": None, "footstrike_timing_right": None,
                         "symmetry_score": None, "mean_accel": None, "variance_accel": None,
                         "rms": None, "sma": None, "dominant_freq": None}

        # ---- Waist side ----
        if has_waist:
            window_waist = [r for r in waist_raw if t <= r["timestamp"] < window_end]
            waist_feat = processing.features_waist.extract_waist_features(window_waist, tilt_offset=tilt_offset) if window_waist else \
                processing.features_waist._empty_waist_features()
        else:
            waist_feat = processing.features_waist._empty_waist_features()

        merged = _merge_window(foot_feat, waist_feat)
        merged["run_id"] = run_id
        merged["window_start"] = t
        merged["window_end"] = window_end
        merged["cadence_variability"] = None  # filled in after the loop, run-level metric

        cadence_values_for_run.append(merged.get("cadence"))
        feature_rows.append(merged)

        t += step

    # Run-level cadence variability, backfilled into every row
    cadence_var = processing.features_waist.compute_cadence_variability(cadence_values_for_run)
    for row in feature_rows:
        row["cadence_variability"] = cadence_var

    db.insert_features(feature_rows)
    logger.info(
        f"Run {run_id}: wrote {len(feature_rows)} windows "
        f"(feet={has_feet}, waist={has_waist})"
    )
    return len(feature_rows)


if __name__ == "__main__":
    db = DatabaseManager()
    runs = db.get_all_runs()
    if not runs:
        print("No runs found.")
    else:
        latest_run_id = runs[0]["run_id"]
        n = process_run_windows(latest_run_id, db)
        print(f"Computed {n} combined feature windows for run {latest_run_id}") 