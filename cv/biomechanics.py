"""
StrideIQ

biomechanics.py

Extracts biomechanical measurements from
MediaPipe landmarks.

NO MACHINE LEARNING HERE.

Only geometry. 
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from cv.models import PoseFrame
from cv.models import FeatureVector


# ============================================================
# Math Utilities
# ============================================================


def distance(a, b):

    return math.sqrt(

        (a.x - b.x) ** 2 +

        (a.y - b.y) ** 2
    )


def midpoint(a, b):

    return (

        (a.x + b.x) / 2,

        (a.y + b.y) / 2,

    )


def vector(a, b):

    return np.array(

        [

            b.x - a.x,

            b.y - a.y

        ]

    )


def angle_three_points(a, b, c):

    """
    Returns angle ABC.

    Used everywhere.
    """

    ba = vector(b, a)

    bc = vector(b, c)

    denom = (

        np.linalg.norm(ba)

        *

        np.linalg.norm(bc)

    )

    if denom == 0:

        return 0.0

    cosine = np.dot(

        ba,

        bc

    ) / denom

    cosine = np.clip(

        cosine,

        -1,

        1

    )

    return math.degrees(

        math.acos(cosine)

    )


# ============================================================
# One Frame Measurements
# ============================================================


@dataclass

@dataclass
class FrameMeasurements:

    timestamp: float

    shoulder_height_difference: float

    shoulder_width: float

    arm_angle_left: float

    arm_angle_right: float 

    crossover_left: float

    crossover_right: float

    trunk_angle: float

    knee_angle_left: float

    knee_angle_right: float

# ============================================================
# Main Analyzer
# ============================================================


class BiomechanicsAnalyzer:

    """
    Converts PoseFrames

    →

    numerical measurements.
    """

    def analyze_frame(

        self,

        pose: PoseFrame,

    ) -> FrameMeasurements:

        body = pose.body

        # -------------------------
        # Shoulders
        # -------------------------

        shoulder_height_difference = abs(

            body.left_shoulder.y -

            body.right_shoulder.y

        )

        shoulder_width = distance(

            body.left_shoulder,

            body.right_shoulder

        )

        # -------------------------
        # Arm Swing
        # -------------------------

        arm_angle_left = angle_three_points(

            body.left_shoulder,

            body.left_elbow,

            body.left_wrist

        )

        arm_angle_right = angle_three_points(

            body.right_shoulder,

            body.right_elbow,

            body.right_wrist

        )

        # -------------------------
        # Arm crossover
        # -------------------------

        shoulder_mid_x = (

            body.left_shoulder.x +

            body.right_shoulder.x

        ) / 2

        crossover_left = (

            shoulder_mid_x -

            body.left_wrist.x

        )

        crossover_right = (

            body.right_wrist.x -

            shoulder_mid_x

        )

        # -------------------------
        # Trunk
        # -------------------------

        sx, sy = midpoint(

            body.left_shoulder,

            body.right_shoulder

        )

        hx, hy = midpoint(

            body.left_hip,

            body.right_hip

        )

        trunk_angle = math.degrees(

            math.atan2(

                sx - hx,

                hy - sy

            )

        )

        # -------------------------
        # Knee drive
        # -------------------------

        knee_angle_left = angle_three_points(

            body.left_hip,

            body.left_knee,

            body.left_ankle

        )

        knee_angle_right = angle_three_points(

            body.right_hip,

            body.right_knee,

            body.right_ankle

        )

        return FrameMeasurements(

         timestamp=pose.timestamp,

         shoulder_height_difference=shoulder_height_difference,

         shoulder_width=shoulder_width,

         arm_angle_left=arm_angle_left,

         arm_angle_right=arm_angle_right,

         crossover_left=crossover_left,

         crossover_right=crossover_right,

         trunk_angle=trunk_angle,

         knee_angle_left=knee_angle_left,

         knee_angle_right=knee_angle_right,

        ) 
# ============================================================
# Temporal Aggregation
# ============================================================


class VideoBiomechanics:

    """
    Converts an entire running video into
    one numerical FeatureVector.
    """

    def __init__(self):

        self.frame_analyzer = BiomechanicsAnalyzer()

    # --------------------------------------------------------

    def analyze_video(

        self,

        pose_frames: List[PoseFrame]

    ) -> FeatureVector:

        measurements = []

        for frame in pose_frames:

            if not frame.pose_detected:

                continue

            measurements.append(

                self.frame_analyzer.analyze_frame(

                    frame

                )

            )

        if len(measurements) == 0:

            return FeatureVector(features={})

        return self._aggregate(

            measurements

        )

    # --------------------------------------------------------

    def _aggregate(

        self,

        data: List[FrameMeasurements]

    ) -> FeatureVector:

        """
        Converts frame-wise measurements

        →

        video-level features.
        """

        features = {}

        self._stats(

            features,

            "shoulder_height",

            [

                d.shoulder_height_difference

                for d in data

            ]

        )

        self._stats(

            features,

            "shoulder_width",

            [

                d.shoulder_width

                for d in data

            ]

        )

        self._stats(

            features,

            "arm_angle_left",

            [

                d.arm_angle_left

                for d in data

            ]

        )

        self._stats(

            features,

            "arm_angle_right",

            [

                d.arm_angle_right

                for d in data

            ]

        )

        self._stats(

            features,

            "crossover_left",

            [

                d.crossover_left

                for d in data

            ]

        )

        self._stats(

            features,

            "crossover_right",

            [

                d.crossover_right

                for d in data

            ]

        )

        self._stats(

            features,

            "trunk_angle",

            [

                d.trunk_angle

                for d in data

            ]

        )

        self._stats(

            features,

            "knee_angle_left",

            [

                d.knee_angle_left

                for d in data

            ]

        )

        self._stats(

            features,

            "knee_angle_right",

            [

                d.knee_angle_right

                for d in data

            ]

        )

        return FeatureVector(

            features=features

        )
# ============================================================
# Statistics
# ============================================================

    def _stats(

    self,

    feature_dict,

    prefix,

    values

    ):

     arr = np.asarray(

        values,

        dtype=np.float32

    )

     feature_dict[prefix + "_mean"] = float(np.mean(arr))

     feature_dict[prefix + "_std"] = float(np.std(arr))

     feature_dict[prefix + "_median"] = float(np.median(arr))

     feature_dict[prefix + "_min"] = float(np.min(arr))

     feature_dict[prefix + "_max"] = float(np.max(arr))

     feature_dict[prefix + "_range"] = float(np.max(arr) - np.min(arr))

     feature_dict[prefix + "_q1"] = float(

        np.percentile(arr, 25)

    )

     feature_dict[prefix + "_q3"] = float(

        np.percentile(arr, 75)

    )

     feature_dict[prefix + "_iqr"] = float(

        feature_dict[prefix + "_q3"]

        -

        feature_dict[prefix + "_q1"]

    )

     feature_dict[prefix + "_p05"] = float(

        np.percentile(arr, 5)

    )

     feature_dict[prefix + "_p95"] = float( 

        np.percentile(arr, 95)

    )
# ============================================================
# Feature Preview
# ============================================================

if __name__ == "__main__":

    print(

        "Biomechanics module loaded."

    )

    print(

        "Waiting for PoseFrames..."

    )