"""
StrideIQ

feature_vector.py

Represents one complete feature vector extracted
from a running video.

This is the interface between:

Biomechanics
        ↓
Machine Learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# ============================================================
# Feature Vector
# ============================================================


@dataclass
class FeatureVector:
    """
    Stores all numerical features extracted from a run.

    Example:

    shoulder_height_mean
    trunk_angle_std
    knee_angle_left_max

    etc.
    """

    features: Dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------

    def __getitem__(self, key: str) -> float:
        return self.features[key]

    # --------------------------------------------------------

    def __setitem__(self, key: str, value: float):

        self.features[key] = float(value)

    # --------------------------------------------------------

    def __contains__(self, key: str):

        return key in self.features

    # --------------------------------------------------------

    def keys(self):

        return self.features.keys()

    # --------------------------------------------------------

    def values(self):

        return self.features.values()

    # --------------------------------------------------------

    def items(self):

        return self.features.items()

    # --------------------------------------------------------

    def update(self, other: Dict[str, float]):

        for k, v in other.items():

            self.features[k] = float(v)

    # --------------------------------------------------------

    def as_dict(self) -> Dict[str, float]:

        return dict(self.features)

    # --------------------------------------------------------

    def as_list(self) -> List[float]:
        """
        Returns values in deterministic order.

        IMPORTANT:
        Predictor depends on this order.
        """

        return [

            self.features[k]

            for k in sorted(self.features.keys())

        ]

    # --------------------------------------------------------

    def as_numpy(self) -> np.ndarray:
        """
        Converts to numpy row vector.

        Shape:

        (1, N)
        """

        return np.array(

            self.as_list(),

            dtype=np.float32

        ).reshape(1, -1)

    # --------------------------------------------------------

    def feature_names(self) -> List[str]:

        return sorted(

            self.features.keys()

        )

    # --------------------------------------------------------

    def size(self) -> int:

        return len(self.features)

    # --------------------------------------------------------

    def is_empty(self) -> bool:

        return len(self.features) == 0

    # --------------------------------------------------------

    def clear(self):

        self.features.clear()

    # --------------------------------------------------------

    def summary(self) -> str:

        return (
            f"FeatureVector("
            f"{self.size()} features)"
        )


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":

    fv = FeatureVector()

    fv["cadence"] = 168

    fv["trunk_angle"] = 7.4

    fv["knee_drive"] = 81.3

    print(fv.summary())

    print(fv.as_numpy())