"""
StrideIQ — CV Module Config
Mirrors backend/config.py's approach: paths built from __file__, not
relative strings, so this works regardless of what directory a script
is run from.
"""

import os

# ============================================================
# Paths
# ============================================================
CV_DIR = os.path.dirname(__file__)
TRAINED_MODELS_DIR = os.path.join(CV_DIR, "trained_models")

# predictor.py currently hardcodes MODEL_DIR = Path("trained_models"),
# a relative path — fragile depending on working directory. Point it
# at this instead if you update predictor.py's MODEL_DIR to import
# from here.

# ============================================================
# MediaPipe Pose settings — passed to PoseEstimator
# ============================================================
POSE_STATIC_IMAGE_MODE = False
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================
# Video loading
# ============================================================
RESIZE_WIDTH = 640  # None to disable resizing; smaller = faster processing

# ============================================================
# Frame sampling
# ============================================================
# Process every Nth frame rather than every frame — running video at
# 30fps doesn't need pose estimation on every single frame to get a
# reliable biomechanics read, and this meaningfully speeds up processing.
FRAME_SAMPLE_INTERVAL = 2

# ============================================================
# Scoring — consistency placeholder
# ============================================================
# ScoringEngine.evaluate() takes a "consistency" argument that should
# ideally come from frame-to-frame variance in BiomechanicsAnalyzer's
# per-frame measurements. VideoBiomechanics.analyze_video() currently
# only returns the aggregated FeatureVector, discarding that per-frame
# data. Until biomechanics.py is extended to expose it, pipeline.py uses
# this as a placeholder — real consistency scoring is a known gap, not
# a finished feature.
DEFAULT_CONSISTENCY_PLACEHOLDER = 0.75

# ============================================================
# LLM summary
# ============================================================
LLM_MODEL = os.getenv("STRIDEIQ_CV_LLM_MODEL", "")