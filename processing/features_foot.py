"""
StrideIQ — Feature Extraction (Foot Metrics)
Converts raw accelerometer windows from both foot nodes into:
  - footstrike impact intensity (per foot)
  - footstrike timing (per foot)
  - left-right symmetry score

Waist-derived features (cadence, vertical oscillation, pelvic tilt/rotation)
are NOT in this file — they get added once the waist node is integrated.
"""

import numpy as np

from backend import config
from backend.database_manager import DatabaseManager 


def _accel_magnitude(ax, ay, az):
    """Combine 3 axes into one magnitude signal — footstrike impact
    doesn't care about orientation, just how hard the foot hit."""
    return np.sqrt(np.array(ax) ** 2 + np.array(ay) ** 2 + np.array(az) ** 2)


def detect_footstrikes(timestamps: list, ax: list, ay: list, az: list) -> list[dict]:
    """
    Simple threshold/peak detection on acceleration magnitude.
    Returns a list of {timestamp, impact_intensity} per detected footstrike.

    Threshold is config.FOOTSTRIKE_PEAK_THRESHOLD_G — tune this against
    real Day 3-4 test run data before trusting it on real sessions.
    """
    if len(timestamps) < 3:
        return []

    magnitude = _accel_magnitude(ax, ay, az)
    threshold = config.FOOTSTRIKE_PEAK_THRESHOLD_G

    strikes = []
    i = 1
    while i < len(magnitude) - 1:
        is_local_peak = magnitude[i] > magnitude[i - 1] and magnitude[i] > magnitude[i + 1]
        if is_local_peak and magnitude[i] > threshold:
            strikes.append({
                "timestamp": timestamps[i],
                "impact_intensity": float(magnitude[i]),
            })
            i += config.MIN_PEAK_DISTANCE  # skip ahead to avoid double-counting the same footstrike
        else:
            i += 1
    return strikes


def compute_footstrike_features(readings: list[dict]) -> dict:
    """
    readings: list of raw_sensor_data rows for ONE foot, ONE window,
    each a dict with timestamp, ax, ay, az.

    Returns average impact intensity and footstrike timing (mean interval
    between strikes) for that window. Returns None values if no strikes
    were detected in the window (e.g. window fell between strides).
    """
    timestamps = [r["timestamp"] for r in readings]
    ax = [r["ax"] for r in readings]
    ay = [r["ay"] for r in readings]
    az = [r["az"] for r in readings]

    strikes = detect_footstrikes(timestamps, ax, ay, az)

    if not strikes:
        return {"impact_intensity": None, "footstrike_timing": None, "strike_count": 0}

    avg_impact = float(np.mean([s["impact_intensity"] for s in strikes]))

    if len(strikes) >= 2:
        intervals = np.diff([s["timestamp"] for s in strikes])
        avg_timing = float(np.mean(intervals))
    else:
        avg_timing = None  # can't compute an interval from a single strike

    return {
        "impact_intensity": avg_impact,
        "footstrike_timing": avg_timing,
        "strike_count": len(strikes),
    }


def compute_symmetry_score(left_features: dict, right_features: dict) -> float | None:
    """
    Compares matched left/right window features to produce a symmetry score.
    0 = perfectly symmetric, higher = more asymmetric.
    Returns None if either side has no detected strikes in this window —
    an asymmetry score computed from a one-sided empty window is meaningless,
    not just noisy.
    """
    if left_features["impact_intensity"] is None or right_features["impact_intensity"] is None:
        return None

    impact_l = left_features["impact_intensity"]
    impact_r = right_features["impact_intensity"]
    denom = max(impact_l, impact_r)
    if denom == 0:
        return 0.0
    impact_diff = abs(impact_l - impact_r) / denom

    timing_diff = 0.0
    if left_features["footstrike_timing"] and right_features["footstrike_timing"]:
        timing_l = left_features["footstrike_timing"]
        timing_r = right_features["footstrike_timing"]
        timing_diff = abs(timing_l - timing_r) / max(timing_l, timing_r)

    # Weighted combination — impact asymmetry weighted higher, it's the
    # more injury-relevant signal per your proposal's framing
    return round(0.7 * impact_diff + 0.3 * timing_diff, 4)


def compute_cadence(left_strike_count: int, right_strike_count: int, window_size_sec: float) -> float | None:
    """
    Cadence = total steps (both feet combined) per minute, derived directly
    from footstrike counts. More accurate than estimating it from the waist's
    periodic acceleration signal, since this comes straight from actual
    detected strikes rather than an indirect proxy.
    """
    total_strikes = left_strike_count + right_strike_count
    if total_strikes == 0:
        return None
    return round((total_strikes / window_size_sec) * 60, 1)


def process_run_windows(run_id: int, db: DatabaseManager):
    """
    Pulls raw readings for both feet, windows them per config.WINDOW_SIZE /
    WINDOW_OVERLAP, computes features per window, and writes results to
    the features table.
    """
    left_raw = db.get_raw_readings(run_id, "foot_left")
    right_raw = db.get_raw_readings(run_id, "foot_right")

    if not left_raw or not right_raw:
        raise ValueError(f"Run {run_id}: missing raw data for one or both feet")

    run_start = min(left_raw[0]["timestamp"], right_raw[0]["timestamp"])
    run_end = max(left_raw[-1]["timestamp"], right_raw[-1]["timestamp"])

    window_size = config.WINDOW_SIZE
    step = window_size - config.WINDOW_OVERLAP

    feature_rows = []
    t = run_start
    while t + window_size <= run_end:
        window_left = [r for r in left_raw if t <= r["timestamp"] < t + window_size]
        window_right = [r for r in right_raw if t <= r["timestamp"] < t + window_size]

        left_feat = compute_footstrike_features(window_left) if window_left else \
            {"impact_intensity": None, "footstrike_timing": None, "strike_count": 0}
        right_feat = compute_footstrike_features(window_right) if window_right else \
            {"impact_intensity": None, "footstrike_timing": None, "strike_count": 0}

        symmetry = compute_symmetry_score(left_feat, right_feat)
        cadence = compute_cadence(
            left_feat.get("strike_count", 0), right_feat.get("strike_count", 0), window_size
        )

        feature_rows.append({
            "run_id": run_id,
            "window_start": t,
            "window_end": t + window_size,
            "cadence": cadence,            # computed directly from foot strikes, not waist-derived
            "vertical_oscillation": None,  # waist-derived, added later
            "running_smoothness": None,    # waist-derived, added later
            "pelvic_tilt": None,           # waist-derived, added later
            "pelvic_tilt_range": None,     # waist-derived, added later
            "pelvic_rotation": None,       # waist-derived, added later
            "pelvic_rotation_range": None, # waist-derived, added later
            "impact_left": left_feat["impact_intensity"],
            "impact_right": right_feat["impact_intensity"],
            "footstrike_timing_left": left_feat["footstrike_timing"],
            "footstrike_timing_right": right_feat["footstrike_timing"],
            "symmetry_score": symmetry,
            "mean_accel": None, "variance_accel": None, "rms": None,
            "sma": None, "dominant_freq": None,
            "sparc": None, "rms_jerk": None, "harmonic_ratio": None,
            "step_regularity": None, "trunk_stability_index": None,
            "cadence_variability": None,
        })

        t += step

    db.insert_features(feature_rows)
    return len(feature_rows)


if __name__ == "__main__":
    # Standalone test: run feature extraction on the most recent test run
    db = DatabaseManager()
    runs = db.get_all_runs()
    if not runs:
        print("No runs found — run ble_receiver.py first to collect test data.")
    else:
        latest_run_id = runs[0]["run_id"]
        n = process_run_windows(latest_run_id, db)
        print(f"Computed features for {n} windows in run {latest_run_id}")