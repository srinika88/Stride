"""
StrideIQ — Annotated Video Output

Produces a copy of the input video with the MediaPipe skeleton overlay
and live biomechanics readouts (trunk angle, knee angles, shoulder
levelness) burned into each frame. Meant for showing a runner/coach
what the CV pipeline actually saw, not as an input to anything else in
the pipeline — nothing else in the codebase currently calls this.

This is a SEPARATE pass from cv/pipeline.py, not a wrapper around it.
Reason: pose_estimator.py's output FramePacket (cv/models.py) only
carries landmarks, not the raw image -- pipeline.py discards the raw
frame after pose estimation since it only needs the landmarks. Drawing
an overlay needs both the raw pixels AND the landmarks together, so
this re-runs video loading + pose estimation in one loop rather than
trying to reuse pipeline.py's already-landmarks-only output.

Skeleton drawing itself isn't reimplemented here -- PoseEstimator
already draws directly onto the frame's pixel buffer via MediaPipe's
own drawing_utils when constructed with draw=True. This module adds
the text readout on top of that.
"""

import logging

import cv2

import cv.config as cv_config
from cv.video_loader import VideoLoader
from cv.pose_estimator import PoseEstimator
from cv.biomechanics import BiomechanicsAnalyzer

logger = logging.getLogger("strideiq.cv.annotated_video")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_COLOR = (0, 255, 0)  # BGR -- green, readable on most footage
FONT_THICKNESS = 2
LINE_HEIGHT = 22
TEXT_ORIGIN = (10, 25)

# codec choice: mp4v is broadly available without extra system codec
# packs, unlike some higher-efficiency codecs that need ffmpeg built
# with specific flags. Good enough for a coaching-review video, not
# meant to be a broadcast-quality export.
FOURCC = "mp4v"


def _overlay_text(frame, lines):
    """Draws each line of text as a separate row, top-left corner."""
    x, y = TEXT_ORIGIN
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x, y + i * LINE_HEIGHT),
            FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
        )


def create_annotated_video(video_path: str, output_path: str) -> dict:
    """
    Reads video_path, writes output_path with skeleton + biomechanics
    text overlay on every frame where pose was detected (frames with no
    detection are written through unmodified, not dropped, so playback
    timing stays intact).

    Returns a small summary dict rather than raising on a fully
    pose-less video -- matches cv/pipeline.py's approach of degrading
    gracefully instead of crashing on unusable footage.
    """
    loader = VideoLoader(video_path, resize_width=cv_config.RESIZE_WIDTH)
    estimator = PoseEstimator(
        static_image_mode=cv_config.POSE_STATIC_IMAGE_MODE,
        model_complexity=cv_config.POSE_MODEL_COMPLEXITY,
        min_detection_confidence=cv_config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=cv_config.POSE_MIN_TRACKING_CONFIDENCE,
        draw=True,  # draws the skeleton directly onto frame_packet.frame
    )
    analyzer = BiomechanicsAnalyzer()

    writer = None
    frames_written = 0
    frames_with_pose = 0

    try:
        for frame_packet in loader.frames():
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*FOURCC)
                writer = cv2.VideoWriter(
                    output_path, fourcc, loader.metadata.fps,
                    (frame_packet.width, frame_packet.height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not open VideoWriter for {output_path} -- "
                        f"check the output directory exists and the '{FOURCC}' "
                        f"codec is available in this OpenCV build."
                    )

            result = estimator.process(frame_packet)  # draws skeleton in place if pose found

            if result is not None:
                frames_with_pose += 1
                measurements = analyzer.analyze_frame(result)
                _overlay_text(frame_packet.frame, [
                    f"trunk: {measurements.trunk_angle:.1f} deg",
                    f"knee L/R: {measurements.knee_angle_left:.1f} / {measurements.knee_angle_right:.1f} deg",
                    f"shoulder diff: {measurements.shoulder_height_difference:.3f}",
                ])
            else:
                _overlay_text(frame_packet.frame, ["no pose detected"])

            writer.write(frame_packet.frame)
            frames_written += 1

    finally:
        estimator.close()
        loader.close()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        logger.warning(f"No frames read from {video_path} -- output video not created")
    elif frames_with_pose == 0:
        logger.warning(
            f"Wrote {frames_written} frames to {output_path}, but pose was "
            f"never detected in any of them -- output has no skeleton overlay"
        )
    else:
        logger.info(
            f"Wrote {frames_written} frames to {output_path} "
            f"({frames_with_pose} with pose detected)"
        )

    return {
        "output_path": output_path,
        "frames_written": frames_written,
        "frames_with_pose": frames_with_pose,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = create_annotated_video(args.video, args.output)
    print(result)