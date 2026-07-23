"""
StrideIQ Computer Vision Module

video_loader.py

---------------------------------------------------------

Production-grade video loading utility.

Responsibilities
----------------
✓ Validate input video

✓ OpenCV initialization

✓ Metadata extraction

✓ Orientation handling (future)

✓ Frame streaming (generator)

✓ Timestamp generation

✓ Optional resizing

✓ Resource management

✓ Webcam-ready architecture

---------------------------------------------------------

Pipeline

Video
    ↓
VideoLoader
    ↓
Frame Generator
    ↓
Pose Estimator
    ↓
Biomechanics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator
from typing import Optional

import cv2
import numpy as np

from cv.models import VideoMetadata


# ==========================================================
# Logger
# ==========================================================

logger = logging.getLogger("StrideIQ.VideoLoader")


# ==========================================================
# Exceptions
# ==========================================================


class VideoLoaderError(Exception):
    """Base exception for video loader."""


class VideoNotFound(VideoLoaderError):
    """Video file does not exist."""


class UnsupportedVideo(VideoLoaderError):
    """Video format not supported."""


class UnableToOpenVideo(VideoLoaderError):
    """OpenCV failed to open the video."""


# ==========================================================
# Frame Packet
# ==========================================================


class FramePacket:
    """
    One streamed frame.

    Passed into MediaPipe.

    Attributes
    ----------

    frame_index

    timestamp

    image (numpy array)

    original_width

    original_height
    """

    __slots__ = (

        "frame",

        "frame_index",

        "timestamp",

        "width",

        "height",

    )

    def __init__(

        self,

        frame: np.ndarray,

        frame_index: int,

        timestamp: float,

    ):

        self.frame = frame

        self.frame_index = frame_index

        self.timestamp = timestamp

        self.height, self.width = frame.shape[:2]


# ==========================================================
# Video Loader
# ==========================================================


class VideoLoader:
    """
    Streams a video frame-by-frame.

    Nothing is loaded entirely into RAM.

    Example
    -------

    loader = VideoLoader("run.mp4")

    for frame in loader:

        ...

    loader.close()
    """

    SUPPORTED_EXTENSIONS = {

        ".mp4",

        ".mov",

        ".avi",

        ".mkv",

        ".mpeg",

        ".mpg",

    }

    # ------------------------------------------------------

    def __init__(

        self,

        video_path: str | Path,

        resize_width: Optional[int] = None,

    ):

        self.video_path = Path(video_path)

        self.resize_width = resize_width

        self.cap: Optional[cv2.VideoCapture] = None

        self.metadata: Optional[VideoMetadata] = None

        self._validate_path()

        self._open_video()

        self.metadata = self._extract_metadata()

        logger.info(

            f"Loaded video: {self.metadata.filename}"

        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    def _validate_path(self):

        if not self.video_path.exists():

            raise VideoNotFound(

                f"{self.video_path} not found."

            )

        if self.video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:

            raise UnsupportedVideo(

                f"{self.video_path.suffix} not supported."

            )

    # ------------------------------------------------------
    # OpenCV
    # ------------------------------------------------------

    def _open_video(self):

        self.cap = cv2.VideoCapture(

            str(self.video_path)

        )

        if not self.cap.isOpened():

            raise UnableToOpenVideo(

                str(self.video_path)

            )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def _extract_metadata(self) -> VideoMetadata:

        assert self.cap is not None

        fps = self.cap.get(

            cv2.CAP_PROP_FPS

        )

        frame_count = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_COUNT

            )

        )

        width = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_WIDTH

            )

        )

        height = int(

            self.cap.get(

                cv2.CAP_PROP_FRAME_HEIGHT

            )

        )

        duration = (

            frame_count / fps

            if fps > 0

            else 0

        )

        fourcc = int(

            self.cap.get(

                cv2.CAP_PROP_FOURCC

            )

        )

        codec = "".join(

            [

                chr((fourcc >> 8 * i) & 0xFF)

                for i in range(4)

            ]

        )

        return VideoMetadata(

            filename=self.video_path.name,

            filepath=str(self.video_path),

            fps=fps,

            frame_count=frame_count,

            width=width,

            height=height,

            duration_seconds=duration,

            codec=codec,

        )

    # ------------------------------------------------------
    # Properties
    # ------------------------------------------------------

    @property
    def fps(self) -> float:

        return self.metadata.fps

    @property
    def duration(self) -> float:

        return self.metadata.duration_seconds

    @property
    def total_frames(self) -> int:

        return self.metadata.frame_count

    # ------------------------------------------------------
    # Resize helper
    # ------------------------------------------------------

    def _resize_frame(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Resize while preserving aspect ratio.

        If resize_width is None, returns
        the original frame.
        """

        if self.resize_width is None:
            return frame

        h, w = frame.shape[:2]

        scale = self.resize_width / w

        new_height = int(h * scale)

        return cv2.resize(
            frame,
            (self.resize_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # ------------------------------------------------------
    # Frame generator
    # ------------------------------------------------------

    def frames(self) -> Iterator[FramePacket]:
        """
        Streams frames one at a time.

        Nothing is buffered into RAM.

        Yields
        ------
        FramePacket
        """

        assert self.cap is not None

        frame_index = 0

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            frame = self._resize_frame(frame)

            timestamp = frame_index / self.fps

            yield FramePacket(

                frame=frame,

                frame_index=frame_index,

                timestamp=timestamp,

            )

            frame_index += 1

    # ------------------------------------------------------
    # Python Iterator Interface
    # ------------------------------------------------------

    def __iter__(self):
        """
        Allows

        for frame in loader
        """

        return self.frames()

    # ------------------------------------------------------
    # Random Access
    # ------------------------------------------------------

    def seek(
        self,
        timestamp_seconds: float
    ):
        """
        Jump to a timestamp.

        Useful later for replay and debugging.
        """

        assert self.cap is not None

        frame = int(

            timestamp_seconds * self.fps

        )

        self.cap.set(

            cv2.CAP_PROP_POS_FRAMES,

            frame

        )

    # ------------------------------------------------------
    # Reset
    # ------------------------------------------------------

    def reset(self):
        """
        Restart video from beginning.
        """

        self.seek(0)

    # ------------------------------------------------------
    # Close
    # ------------------------------------------------------

    def close(self):
        """
        Releases OpenCV resources.
        """

        if self.cap is not None:

            self.cap.release()

            self.cap = None

            logger.info(

                "Video released."

            )

    # ------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(

        self,

        exc_type,

        exc,

        traceback

    ):

        self.close()

    # ------------------------------------------------------
    # String Representation
    # ------------------------------------------------------

    def __repr__(self):

        return (

            f"VideoLoader("
            f"{self.metadata.filename}, "
            f"{self.metadata.frame_count} frames, "
            f"{self.metadata.fps:.1f} FPS)"
        )


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    VIDEO = "sample.mp4"

    try:

        with VideoLoader(VIDEO) as loader:

            print()

            print("Video Loaded")

            print("------------------------")

            print(loader.metadata)

            print()

            for frame in loader:

                print(

                    f"Frame "

                    f"{frame.frame_index} "

                    f"Time "

                    f"{frame.timestamp:.2f}s"

                )

                cv2.imshow(

                    "StrideIQ",

                    frame.frame

                )

                key = cv2.waitKey(1)

                if key == ord("q"):

                    break

            cv2.destroyAllWindows()

    except Exception as e:

        print(e) 