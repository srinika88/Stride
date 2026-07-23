"""
StrideIQ — Shared ML Feature Columns
Single source of truth for FEATURE_COLUMNS. Previously duplicated
independently in train.py, clustering.py, and hyperparameter_search.py,
each with a comment admitting it was manually kept in sync — a real
risk that a renamed/added feature column gets updated in one file and
silently missed in the others. Import FEATURE_COLUMNS from here instead.
"""

from __future__ import annotations

FEATURE_COLUMNS = [
    "cadence", "vertical_oscillation", "running_smoothness",
    "pelvic_tilt", "pelvic_tilt_range", "pelvic_rotation", "pelvic_rotation_range",
    "impact_left", "impact_right", "footstrike_timing_left", "footstrike_timing_right",
    "symmetry_score", "sparc", "rms_jerk", "harmonic_ratio",
    "step_regularity", "trunk_stability_index", "cadence_variability",
] 