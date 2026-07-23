"""
StrideIQ — Heuristic Labeler
Assigns one of config.FORM_LABELS to a computed feature row using
threshold rules, for use as label_source='heuristic'.

This exists specifically for public/pre-existing IMU datasets: there's
no video to run MediaPipe on, and no researcher manually induced bad
form on command, so neither 'mediapipe' nor 'induced_manual' applies.
Thresholds are a starting point (config.HEURISTIC_*), not validated
ground truth — retune them once you have real MediaPipe-labeled
sessions of your own to check against.

Only labels windows where the underlying features exist. A foot-only
dataset (no waist node) can never surface "Excessive Pelvic Tilt" —
that requires pelvic_tilt, which stays None until the waist node is
integrated. Rather than guessing, unlabelable windows are skipped.
"""

import logging

from backend import config

logger = logging.getLogger("strideiq.ml.heuristic_labeler")


def label_window(feature_row: dict) -> tuple[str, float] | None:
    """
    feature_row: one row from the `features` table (as a dict).
    Returns (label, confidence) or None if this window can't be
    labeled from the features actually present (e.g. no footstrikes
    detected on either foot in this window).

    Priority: High Impact > Asymmetry > Excessive Pelvic Tilt > Normal.
    Impact and asymmetry are checked first because they're the more
    injury-relevant signals per the project proposal's framing
    (features_foot.compute_symmetry_score weights impact higher too).
    """
    impact_left = feature_row.get("impact_left")
    impact_right = feature_row.get("impact_right")
    symmetry = feature_row.get("symmetry_score")
    pelvic_tilt = feature_row.get("pelvic_tilt")

    max_impact = None
    if impact_left is not None or impact_right is not None:
        candidates = [v for v in (impact_left, impact_right) if v is not None]
        max_impact = max(candidates) if candidates else None

    if max_impact is None and symmetry is None and pelvic_tilt is None:
        # nothing usable in this window (e.g. it fell between strides
        # on both feet) — don't fabricate a label
        return None

    if max_impact is not None and max_impact >= config.HEURISTIC_HIGH_IMPACT_G:
        margin = (max_impact - config.HEURISTIC_HIGH_IMPACT_G) / config.HEURISTIC_HIGH_IMPACT_G
        confidence = min(0.55 + margin, 0.95)
        return "High Impact", round(confidence, 3)

    if symmetry is not None and symmetry >= config.HEURISTIC_ASYMMETRY_SCORE:
        margin = (symmetry - config.HEURISTIC_ASYMMETRY_SCORE) / config.HEURISTIC_ASYMMETRY_SCORE
        confidence = min(0.55 + margin, 0.95)
        return "Asymmetry", round(confidence, 3)

    if pelvic_tilt is not None and abs(pelvic_tilt) >= config.HEURISTIC_PELVIC_TILT_DEG:
        margin = (abs(pelvic_tilt) - config.HEURISTIC_PELVIC_TILT_DEG) / config.HEURISTIC_PELVIC_TILT_DEG
        confidence = min(0.55 + margin, 0.95)
        return "Excessive Pelvic Tilt", round(confidence, 3)

    # below every threshold checked -> Normal. Confidence reflects how
    # far below the nearest threshold it sits, capped so "Normal" never
    # claims more certainty than the thresholds themselves warrant.
    return "Normal", 0.6


def label_run(run_id: int, db) -> dict:
    """
    Labels every feature window in a run and writes results via
    db.insert_label(..., label_source='heuristic'). Returns counts per
    label plus how many windows were skipped as unlabelable, so you can
    see class balance immediately instead of discovering it later in
    train.py's sparse-class warning. 
    """
    features = db.get_features(run_id)
    if not features:
        raise ValueError(f"Run {run_id}: no feature rows found — run feature extraction first")

    counts = {label: 0 for label in config.FORM_LABELS}
    skipped = 0

    for row in features:
        row = dict(row)
        result = label_window(row)
        if result is None:
            skipped += 1
            continue

        label, confidence = result
        db.insert_label(
            run_id=run_id,
            window_start=row["window_start"], 
            window_end=row["window_end"],
            label=label,
            label_source="heuristic",
            confidence=confidence,
        )
        counts[label] += 1

    logger.info(f"Run {run_id}: labeled {sum(counts.values())} windows, skipped {skipped}. Breakdown: {counts}")
    return {"counts": counts, "skipped": skipped} 