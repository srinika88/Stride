"""
StrideIQ

predictor.py

Loads trained ML models and predicts
running quality from biomechanical
features.
"""

from __future__ import annotations

from pathlib import Path

import joblib

import numpy as np

from cv.models import (
    FeatureVector,
    Prediction,
    Grade,
    MetricName,
)

# ============================================================
# Default Model Locations
# ============================================================

MODEL_DIR = Path("trained_models")

MODEL_FILES = {

    MetricName.SHOULDER_RELAXATION:
        MODEL_DIR / "shoulder.pkl",

    MetricName.ARM_SWING:
        MODEL_DIR / "arm_swing.pkl",

    MetricName.ARM_CROSSOVER:
        MODEL_DIR / "arm_crossover.pkl",

    MetricName.TRUNK_LEAN:
        MODEL_DIR / "trunk_lean.pkl",

    MetricName.KNEE_DRIVE:
        MODEL_DIR / "knee_drive.pkl",

}

# ============================================================
# Predictor
# ============================================================


class Predictor:

    """
    Loads one model for each
    biomechanical metric.
    """

    def __init__(self):

        self.models = {}

        self._load_models()

    # --------------------------------------------------------

    def _load_models(self):

        for metric, path in MODEL_FILES.items():

            if path.exists():

                self.models[metric] = joblib.load(path)

            else:

                self.models[metric] = None
# ============================================================
# Prediction
# ============================================================

    def predict(

        self,

        feature_vector: FeatureVector

    ) -> list[Prediction]:

        predictions = []

        x = np.array(

            list(feature_vector.features.values()),

            dtype=np.float32

        ).reshape(1, -1)

        for metric in MetricName:

            model = self.models[metric]

            # -----------------------------------
            # No trained model yet
            # -----------------------------------

            if model is None:

                predictions.append(

                    Prediction(

                        metric=metric,

                        score=75.0,

                        grade=Grade.GOOD,

                        confidence=0.0

                    )

                )

                continue

            # -----------------------------------
            # Real prediction
            # -----------------------------------

            score = float(

                model.predict(x)[0]

            )

            confidence = float(

                np.max(

                    model.predict_proba(x)

                )

            )

            predictions.append(

                Prediction(

                    metric=metric,

                    score=score,

                    grade=self._score_to_grade(score),

                    confidence=confidence

                )

            )

        return predictions
# ============================================================
# Grade Mapping
# ============================================================

    def _score_to_grade(

        self,

        score: float

    ) -> Grade:

        if score >= 90:

            return Grade.EXCELLENT

        elif score >= 75:

            return Grade.GOOD

        elif score >= 60:

            return Grade.FAIR

        elif score >= 40:

            return Grade.NEEDS_IMPROVEMENT

        else:

            return Grade.HIGH_PRIORITY
# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print(

        "StrideIQ Predictor"

    )

    print(

        "Waiting for trained models..."

    )
