"""
StrideIQ — Recommendation Engine
Phase 5's "rule engine" + "recommendation mapping": a threshold-based
classifier that needs zero labeled data, plus a label -> coaching text
mapping.

Two jobs:
1. classify_window(feature_row) -> (label, confidence) — a rule-based
   bootstrap classifier, usable from day one before enough labeled
   sessions exist to train ml/train.py's real model. predict.py's
   docstring calls this out directly: "recommend.py's classify_window()
   was always meant to be swapped for [the trained model] ... same
   (label, confidence) output shape, so nothing downstream needs to
   change." This file is that piece — swap ml.predict.predict_window
   in once a model exists; until then, this keeps the MQTT feedback
   loop (backend/mqtt/feedback_loop.py) usable end-to-end.
2. recommend(label, feature_row) -> short templated coaching text — a
   non-LLM fallback/companion to llm/langchain_pipeline.generate_coaching,
   for when no LLM is configured (config.LLM_MODEL unset) or the LLM
   call fails.

Thresholds below are starting points, same spirit as
config.FOOTSTRIKE_PEAK_THRESHOLD_G — tune against real test-run data,
don't treat them as final.
"""

from __future__ import annotations

import logging

from backend import config

logger = logging.getLogger("strideiq.ml.recommendation_engine")

# Rule thresholds — deliberately conservative starting points; each maps
# to one of config.FORM_LABELS so output is compatible with everything
# downstream that expects those exact label strings (actuation logic,
# the labels table's CHECK constraint, etc).
IMPACT_HIGH_THRESHOLD_G = 3.0          # avg footstrike impact intensity
SYMMETRY_ASYMMETRY_THRESHOLD = 0.25    # 0=symmetric, higher=more asymmetric
PELVIC_TILT_RANGE_THRESHOLD_DEG = 12.0  # degrees of pelvic tilt swing per window

RECOMMENDATION_TEMPLATES = {
    "High Impact": (
        "Your footstrike impact is running high this window. Try a "
        "shorter, quicker stride and aim to land more underneath your "
        "body rather than reaching out in front."
    ),
    "Asymmetry": (
        "There's a noticeable left/right imbalance in this window. "
        "Check for one-sided fatigue, tightness, or an old niggle — "
        "and consider a short mobility check-in on the weaker side."
    ),
    "Excessive Pelvic Tilt": (
        "Your pelvis is tilting more than usual this window, often a "
        "sign of core/hip fatigue. Light core engagement and a slight "
        "cadence bump can help steady this."
    ),
    "Normal": (
        "Form looks solid this window — cadence, impact, and symmetry "
        "are all in a comfortable range. Keep doing what you're doing."
    ),
}

DEFAULT_RECOMMENDATION = (
    "Not enough signal in this window to give a confident read — "
    "keep an eye on how you're feeling and we'll pick it up next window."
)


def classify_window(feature_row: dict) -> tuple[str, float]:
    """
    Rule-based fallback classifier. Returns (label, confidence) with the
    same shape as ml.predict.predict_window(), so callers (the MQTT
    feedback loop, a dashboard preview) can use either interchangeably.

    Confidence here isn't a real predicted probability — it's a fixed,
    conservative stand-in reflecting "this is a simple threshold rule,
    not a calibrated model." Don't feed it into anything that expects a
    genuine probability distribution.
    """
    impact_left = feature_row.get("impact_left")
    impact_right = feature_row.get("impact_right")
    symmetry = feature_row.get("symmetry_score")
    pelvic_tilt_range = feature_row.get("pelvic_tilt_range")

    impacts = [v for v in (impact_left, impact_right) if v is not None]

    # Priority order matters: asymmetry and pelvic tilt are checked before
    # raw impact, since a high impact reading is often a downstream
    # symptom of one of those rather than an independent issue — no
    # point flagging "High Impact" and missing the more actionable cause.
    if symmetry is not None and symmetry >= SYMMETRY_ASYMMETRY_THRESHOLD:
        return "Asymmetry", 0.6

    if pelvic_tilt_range is not None and pelvic_tilt_range >= PELVIC_TILT_RANGE_THRESHOLD_DEG:
        return "Excessive Pelvic Tilt", 0.6

    if impacts and max(impacts) >= IMPACT_HIGH_THRESHOLD_G:
        return "High Impact", 0.6

    if not impacts and symmetry is None and pelvic_tilt_range is None:
        # No usable signal at all in this window (e.g. a dropped node) —
        # don't confidently call it "Normal" when there's nothing to base
        # that on.
        return "Normal", 0.3

    return "Normal", 0.55


def recommend(label: str, feature_row: dict = None) -> str:
    """
    Maps a form label to a short templated coaching message. This is the
    non-LLM counterpart to llm.langchain_pipeline.generate_coaching —
    use this when config.LLM_MODEL isn't set, or as a guaranteed fallback
    if the LLM call raises/times out, so a recommendation always gets
    stored even when the LLM path is unavailable.
    """
    if label not in RECOMMENDATION_TEMPLATES:
        logger.warning(f"Unknown label '{label}' — using default recommendation text")
        return DEFAULT_RECOMMENDATION

    return RECOMMENDATION_TEMPLATES[label]


def classify_and_recommend(feature_row: dict) -> dict:
    """
    Convenience combo of the two functions above — one call, one dict
    result, matching the shape the MQTT feedback loop and any dashboard
    preview code would want: label, confidence, and ready-to-display text.
    """
    label, confidence = classify_window(feature_row)
    return {
        "predicted_label": label,
        "confidence": confidence,
        "recommendation_text": recommend(label, feature_row),
    }


if __name__ == "__main__":
    # Standalone smoke test — no database or trained model required.
    sample_windows = [
        {"impact_left": 1.2, "impact_right": 1.1, "symmetry_score": 0.05, "pelvic_tilt_range": 4.0},
        {"impact_left": 4.1, "impact_right": 1.3, "symmetry_score": 0.4, "pelvic_tilt_range": 5.0},
        {"impact_left": 3.5, "impact_right": 3.6, "symmetry_score": 0.1, "pelvic_tilt_range": 3.0},
        {"impact_left": None, "impact_right": None, "symmetry_score": None, "pelvic_tilt_range": None},
    ]
    for window in sample_windows:
        result = classify_and_recommend(window)
        print(f"{window} -> {result['predicted_label']} ({result['confidence']}): {result['recommendation_text']}")
