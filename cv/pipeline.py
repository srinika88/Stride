"""
StrideIQ — CV Pipeline
Orchestrates the full video -> report chain using the REAL classes/
signatures already in this codebase (video_loader, pose_estimator,
biomechanics, predictor, scoring, coaching_engine) — nothing here
invents new logic, it wires what already exists.

Known gaps, stated plainly rather than papered over:
- Predictor's per-metric models (shoulder.pkl, arm_swing.pkl, etc.) don't
  exist yet — only an empty trained_models/strideiq_model.pkl placeholder
  is present. Predictions will come back as None/low-confidence until
  real models are trained. This pipeline handles that gracefully rather
  than crashing, but it does NOT fix the missing training step.
- "Consistency" fed into ScoringEngine.evaluate() is a placeholder
  (config.DEFAULT_CONSISTENCY_PLACEHOLDER), since VideoBiomechanics
  only returns an aggregated FeatureVector, not per-frame variance.
  Real consistency scoring needs biomechanics.py extended to expose
  frame-level stats — flagged as future work, not solved here.
"""

import logging
from typing import Optional

import cv.config as cv_config
from cv.video_loader import VideoLoader
from cv.pose_estimator import PoseEstimator
from cv.biomechanics import VideoBiomechanics
from cv.predictor import Predictor
from cv.scoring import ScoringEngine
from cv.coaching_engine import CoachingEngine
from cv.models import (
    CVPipelineOutput,
    RunningReport,
    MetricResult as ModelsMetricResult,
    Grade,
)

logger = logging.getLogger("strideiq.cv.pipeline")


def run_cv_pipeline(video_path: str, wearable_data: Optional[dict] = None) -> CVPipelineOutput:
    """
    Full pipeline: video file -> pose frames -> biomechanics -> ML
    predictions -> scored metrics -> coaching findings -> report.

    wearable_data: optional dict of sensor-derived context (e.g. from
    the current run's feature rows) — passed through to CoachingEngine's
    sensor_fusion_rules(), which is the real integration point between
    this CV module and the wearable pipeline. None by default, since
    that fusion wiring (cv/wearable/*) isn't built yet.
    """
    # ------------------------------------------------------------
    # 1. Load video, extract pose per sampled frame
    # ------------------------------------------------------------
    loader = VideoLoader(video_path, resize_width=cv_config.RESIZE_WIDTH)
    estimator = PoseEstimator(
        static_image_mode=cv_config.POSE_STATIC_IMAGE_MODE,
        model_complexity=cv_config.POSE_MODEL_COMPLEXITY,
        min_detection_confidence=cv_config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=cv_config.POSE_MIN_TRACKING_CONFIDENCE,
    )

    pose_frames = []
    try:
        for i, frame_packet in enumerate(loader.frames()):
            if i % cv_config.FRAME_SAMPLE_INTERVAL != 0:
                continue
            result = estimator.process(frame_packet)
            if result is not None:
                pose_frames.append(result)
    finally:
        estimator.close()
        loader.close()

    logger.info(f"Processed {len(pose_frames)} pose-detected frames from {video_path}")

    if not pose_frames:
        logger.warning("No pose detected in any frame — cannot produce a report")
        return CVPipelineOutput(metadata=loader.metadata, feature_vector=None,
                                 predictions=None, report=None)

    # ------------------------------------------------------------
    # 2. Frames -> aggregated biomechanics feature vector
    # ------------------------------------------------------------
    biomechanics = VideoBiomechanics()
    feature_vector = biomechanics.analyze_video(pose_frames)

    # ------------------------------------------------------------
    # 3. Feature vector -> per-metric ML predictions
    # ------------------------------------------------------------
    predictions = []
    try:
        predictor = Predictor()
        predictions = predictor.predict(feature_vector)
    except Exception:
        logger.exception("Predictor failed — likely missing trained per-metric models")

    if not predictions:
        logger.warning(
            "No predictions produced (models not trained yet) — "
            "returning feature vector without a scored report"
        )
        return CVPipelineOutput(metadata=loader.metadata, feature_vector=feature_vector,
                                 predictions=None, report=None)

    # ------------------------------------------------------------
    # 4. Predictions -> scored metric results
    # ------------------------------------------------------------
    scorer = ScoringEngine()
    metric_results = []
    for pred in predictions:
        result = scorer.evaluate(
            metric=pred.metric,
            score=pred.score,
            confidence=pred.confidence,
            consistency=cv_config.DEFAULT_CONSISTENCY_PLACEHOLDER,  # see module docstring
        )
        metric_results.append(result)

    # ------------------------------------------------------------
    # 5. Scored metrics -> coaching findings (with optional wearable fusion)
    # ------------------------------------------------------------
    coach = CoachingEngine()
    findings = coach.generate(metrics=metric_results, wearable_data=wearable_data)

    # ------------------------------------------------------------
    # 6. Assemble the report
    # ------------------------------------------------------------
    overall_score = scorer.overall_score(metric_results) if hasattr(scorer, "overall_score") else \
        (sum(r.score for r in metric_results) / len(metric_results) if metric_results else 0.0)

    overall_grade = _grade_from_score(overall_score)

    report = RunningReport(
        video=loader.metadata,
        overall_score=overall_score,
        overall_grade=overall_grade,
        metrics=[_to_models_metric_result(r) for r in metric_results],
        strengths=[f.summary for f in findings if getattr(f, "is_positive", False)] if findings else [],
        improvements=[f.summary for f in findings if not getattr(f, "is_positive", True)] if findings else [],
    )

    return CVPipelineOutput(
        metadata=loader.metadata,
        feature_vector=feature_vector,
        predictions=predictions,
        report=report,
    )


def _grade_from_score(score: float) -> Grade:
    if score >= 90:
        return Grade.EXCELLENT
    if score >= 75:
        return Grade.GOOD
    if score >= 60:
        return Grade.FAIR
    if score >= 40:
        return Grade.NEEDS_IMPROVEMENT
    return Grade.HIGH_PRIORITY


def _to_models_metric_result(scoring_result) -> ModelsMetricResult:
    """
    scoring.py defines its own MetricResult dataclass, separate from
    models.MetricResult (a pydantic model). Both exist in the codebase
    under the same name in different modules — this converts between
    them for the final report, which is typed against models.MetricResult.
    Worth renaming one of the two classes eventually to avoid the
    same-name confusion; not fixed here since it touches multiple files.
    """
    return ModelsMetricResult(
        metric=scoring_result.metric,
        score=scoring_result.score,
        grade=_grade_from_score(scoring_result.score),
        confidence=getattr(scoring_result, "confidence", 0.5),
        explanation=getattr(scoring_result, "explanation", ""),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m cv.pipeline <video_path>")
        sys.exit(1)
    output = run_cv_pipeline(sys.argv[1])
    if output.report:
        print(f"Overall score: {output.report.overall_score} ({output.report.overall_grade})")
        for m in output.report.metrics:
            print(f"  {m.metric}: {m.score} ({m.grade})")
    else:
        print("No report generated — see logs for why (no pose detected / no trained models).")
