"""
StrideIQ

processor.py

Converts raw IMU packets into meaningful running metrics.

Responsibilities
----------------
- Buffer incoming packets
- Maintain sliding windows
- Apply signal filtering
- Compute running metrics

NO BLE
NO MACHINE LEARNING
NO COACHING
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List

import numpy as np

from cv.wearable.receiver import RawPacket 


# ============================================================
# Metrics
# ============================================================


@dataclass
class FootMetrics:
    """
    Metrics computed independently
    for each foot.
    """

    cadence: float = 0.0

    impact: float = 0.0

    impact_variability: float = 0.0

    contact_time: float = 0.0

    stride_variability: float = 0.0


@dataclass
class RunningMetrics:
    """
    Combined running metrics.
    """

    left: FootMetrics = field(default_factory=FootMetrics)

    right: FootMetrics = field(default_factory=FootMetrics)

    symmetry: float = 0.0

    vertical_oscillation: float = 0.0

    smoothness: float = 0.0


# ============================================================
# Processor
# ============================================================


class WearableProcessor:

    """
    Converts packet streams into
    running metrics.
    """

    SAMPLE_RATE = 50

    WINDOW_SECONDS = 2

    WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

    def __init__(self):

        # Raw packet buffers

        self.left_buffer = deque(maxlen=self.WINDOW_SIZE)

        self.right_buffer = deque(maxlen=self.WINDOW_SIZE)

        self.waist_buffer = deque(maxlen=self.WINDOW_SIZE)

        self.latest_metrics = RunningMetrics()

        print("[Processor] Initialized.")
    
        # --------------------------------------------------------
    # Add Packet
    # --------------------------------------------------------

    def add_packet(

        self,

        packet: RawPacket,

    ):

        """
        Routes packet into the
        correct sliding window.
        """

        if packet.node == "L":

            self.left_buffer.append(packet)

        elif packet.node == "R":

            self.right_buffer.append(packet)

        elif packet.node == "W":

            self.waist_buffer.append(packet)

    # --------------------------------------------------------
    # Enough Data?
    # --------------------------------------------------------

    def ready(self):

        """
        Wait until all available sensors
        have enough samples.
        """

        left_ready = len(self.left_buffer) >= self.WINDOW_SIZE

        right_ready = len(self.right_buffer) >= self.WINDOW_SIZE

        waist_ready = len(self.waist_buffer) >= self.WINDOW_SIZE

        return left_ready or right_ready or waist_ready

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def _acceleration_array(

        self,

        packets,

    ):

        """
        Converts packet buffer

        →

        Nx3 numpy array.
        """

        return np.array(

            [

                [

                    p.ax,

                    p.ay,

                    p.az,

                ]

                for p in packets

            ],

            dtype=np.float32,

        )

    # --------------------------------------------------------
    # Magnitude
    # --------------------------------------------------------

    def _magnitude(

        self,

        packets,

    ):

        acc = self._acceleration_array(

            packets

        )

        return np.linalg.norm(

            acc,

            axis=1,

        )
        # --------------------------------------------------------
    # Peak Detection
    # --------------------------------------------------------

    def _find_peaks(

        self,

        signal,

        prominence=150,

        distance=20,

    ):
        """
        Detect foot-strike peaks.

        prominence:
            Minimum peak height above neighbours.

        distance:
            Minimum samples between peaks.
        """

        try:

            from scipy.signal import find_peaks

            peaks, _ = find_peaks(

                signal,

                prominence=prominence,

                distance=distance,

            )

            return peaks

        except Exception:

            # scipy missing

            return np.array([])


    # --------------------------------------------------------
    # Cadence
    # --------------------------------------------------------

    def _compute_cadence(

        self,

        packets,

    ):

        if len(packets) < self.WINDOW_SIZE:

            return 0.0

        magnitude = self._magnitude(

            packets

        )

        peaks = self._find_peaks(

            magnitude

        )

        duration_seconds = (

            len(magnitude)

            / self.SAMPLE_RATE

        )

        if duration_seconds <= 0:

            return 0.0

        steps = len(peaks)

        cadence = (

            steps

            / duration_seconds

        ) * 60.0

        return float(cadence)


    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    def _compute_impact(

        self,

        packets,

    ):

        if len(packets) == 0:

            return 0.0

        magnitude = self._magnitude(

            packets

        )

        peaks = self._find_peaks(

            magnitude

        )

        if len(peaks) == 0:

            return 0.0

        impacts = magnitude[peaks]

        return float(

            np.mean(

                impacts

            )

        )


    # --------------------------------------------------------
    # Impact Variability
    # --------------------------------------------------------

    def _impact_variability(

        self,

        packets,

    ):

        magnitude = self._magnitude(

            packets

        )

        peaks = self._find_peaks(

            magnitude

        )

        if len(peaks) < 2:

            return 0.0

        impacts = magnitude[peaks]

        return float(

            np.std(

                impacts

            )

        )
        # --------------------------------------------------------
    # Left / Right Symmetry
    # --------------------------------------------------------

    def _compute_symmetry(

        self,

        left: FootMetrics,

        right: FootMetrics,

    ):

        values = [

            ("cadence", left.cadence, right.cadence),

            ("impact", left.impact, right.impact),

            ("contact", left.contact_time, right.contact_time),

        ]

        scores = []

        for _, l, r in values:

            if l <= 0 or r <= 0:

                continue

            diff = abs(l - r)

            avg = (l + r) / 2

            score = max(

                0.0,

                100.0 * (1 - diff / avg)

            )

            scores.append(score)

        if len(scores) == 0:

            return 0.0

        return float(np.mean(scores))


    # --------------------------------------------------------
    # Running Smoothness
    # --------------------------------------------------------

    def _compute_smoothness(

        self,

        packets,

    ):

        if len(packets) < self.WINDOW_SIZE:

            return 0.0

        magnitude = self._magnitude(

            packets

        )

        jerk = np.diff(magnitude)

        std = np.std(jerk)

        score = max(

            0.0,

            100.0 - std

        )

        return float(score)


    # --------------------------------------------------------
    # Vertical Oscillation
    # --------------------------------------------------------

    def _vertical_oscillation(

        self,

        packets,

    ):

        if len(packets) < self.WINDOW_SIZE:

            return 0.0

        acc = self._acceleration_array(

            packets

        )

        z = acc[:, 2]

        return float(

            np.std(z)

        ) 