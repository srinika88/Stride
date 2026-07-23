"""
gait_analyzer.py — StrideIQ Real-Time Foot Gait Analysis

Real-time processing path.

Consumes:
    FootPacket streams from ble_receiver.py

Computes:
    - Left/right cadence
    - Combined cadence
    - Footstrike impact magnitude
    - Footstrike timing asymmetry
    - Left/right impact asymmetry

This module is NOT the ML training pipeline.
Offline feature extraction belongs in ml/feature_extraction.py.

Flow:

BLE Receiver
      |
      ↓
GaitAnalyzer.ingest_packet()
      |
      ↓
Rolling window analysis
      |
      ↓
features callback
      |
      ↓
Dashboard / Database / Feedback
"""

import logging
import math
import time

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


logger = logging.getLogger("strideiq.gait")


# ============================================================
# Parameters
# ============================================================

WINDOW_SECONDS = 5.0

SAMPLE_RATE_HZ = 50

# micro:bit acceleration is mg
# 1000mg = gravity
IMPACT_PEAK_THRESHOLD_MG = 1300

# avoid counting vibration as another step
MIN_PEAK_SPACING_SEC = 0.25

ASYMMETRY_WARN_THRESHOLD_PCT = 15.0

FEEDBACK_COOLDOWN_SEC = 10.0



# ============================================================
# Data structures
# ============================================================

@dataclass
class FootSample:

    timestamp: float
    magnitude: float



# ============================================================
# Rolling buffer
# ============================================================

class RollingFootBuffer:


    def __init__(
        self,
        window_seconds=WINDOW_SECONDS
    ):

        self.window_seconds = window_seconds
        self.samples = deque()



    def add(self, packet):

        magnitude = math.sqrt(

            packet.acc_x ** 2 +

            packet.acc_y ** 2 +

            packet.acc_z ** 2

        )


        self.samples.append(

            FootSample(

                timestamp=packet.pi_arrival_ts,

                magnitude=magnitude

            )

        )


        self._trim()



    def _trim(self):

        cutoff = time.time() - self.window_seconds


        while (

            self.samples and

            self.samples[0].timestamp < cutoff

        ):

            self.samples.popleft()



    def detect_peaks(self):

        """
        Detect real acceleration peaks.

        Conditions:
        1. local maximum
        2. above impact threshold
        3. minimum spacing between strikes
        """

        self._trim()


        peaks = []

        last_peak_time = -999


        samples = list(self.samples)


        for i in range(1, len(samples)-1):

            prev_sample = samples[i-1]

            current = samples[i]

            next_sample = samples[i+1]


            is_peak = (

                current.magnitude >
                prev_sample.magnitude

                and

                current.magnitude >
                next_sample.magnitude

                and

                current.magnitude >=
                IMPACT_PEAK_THRESHOLD_MG

            )


            if is_peak:

                if (

                    current.timestamp -
                    last_peak_time

                ) >= MIN_PEAK_SPACING_SEC:


                    peaks.append(current)

                    last_peak_time = current.timestamp



        return peaks




# ============================================================
# Main analyzer
# ============================================================

class GaitAnalyzer:


    def __init__(

        self,

        on_features:
        Optional[Callable[[dict], None]]=None,

        on_feedback:
        Optional[Callable[[dict], None]]=None

    ):


        self.buffers = {

            "L": RollingFootBuffer(),

            "R": RollingFootBuffer()

        }


        self.on_features = on_features

        self.on_feedback = on_feedback


        self.last_feedback_time = 0



    # --------------------------------------------------------

    def ingest_packet(self, packet):

        """
        Called by BLE receiver for every foot packet.
        """

        if packet.node in self.buffers:

            self.buffers[packet.node].add(packet)



    # --------------------------------------------------------

    def analyze(self):

        """
        Run once every ~1 second from main.py
        """

        left_peaks = (
            self.buffers["L"].detect_peaks()
        )

        right_peaks = (
            self.buffers["R"].detect_peaks()
        )


        left_cadence = (
            self.calculate_cadence(left_peaks)
        )

        right_cadence = (
            self.calculate_cadence(right_peaks)
        )


        left_impact = (
            self.average_peak(left_peaks)
        )

        right_impact = (
            self.average_peak(right_peaks)
        )


        features = {


            "timestamp":
            time.time(),


            "cadence_left_spm":
            left_cadence,


            "cadence_right_spm":
            right_cadence,


            "cadence_combined_spm":

            (

                left_cadence + right_cadence

                if

                left_cadence and right_cadence

                else None

            ),



            "impact_left_mg":
            left_impact,


            "impact_right_mg":
            right_impact,



            "impact_asymmetry_pct":

            self.calculate_asymmetry(

                left_impact,

                right_impact

            ),



            "timing_asymmetry_ms":

            self.timing_asymmetry(

                left_peaks,

                right_peaks

            )

        }



        if self.on_features:

            self.on_features(features)



        self.check_feedback(features)


        return features



    # --------------------------------------------------------

    def calculate_cadence(self, peaks):

        if len(peaks) < 2:

            return None


        duration = (

            peaks[-1].timestamp -

            peaks[0].timestamp

        )


        if duration <= 0:

            return None


        steps_per_second = (

            (len(peaks)-1)

            /

            duration

        )


        return round(

            steps_per_second * 60,

            1

        )



    # --------------------------------------------------------

    def average_peak(self, peaks):

        if not peaks:

            return None


        return round(

            sum(

                p.magnitude

                for p in peaks

            )

            /

            len(peaks),

            1

        )



    # --------------------------------------------------------

    def calculate_asymmetry(
        self,
        left,
        right
    ):


        if left is None or right is None:

            return None


        avg = (

            left + right

        ) / 2


        if avg == 0:

            return None


        return round(

            abs(left-right)

            /

            avg

            *

            100,

            1

        )



    # --------------------------------------------------------

    def timing_asymmetry(
        self,
        left_peaks,
        right_peaks
    ):


        if not left_peaks or not right_peaks:

            return None


        differences = []


        for left in left_peaks:

            closest = min(

                right_peaks,

                key=lambda r:

                abs(

                    r.timestamp -
                    left.timestamp

                )

            )


            differences.append(

                abs(

                    left.timestamp -

                    closest.timestamp

                )

                *

                1000

            )


        return round(

            sum(differences)

            /

            len(differences),

            1

        )



    # --------------------------------------------------------

    def check_feedback(self, features):


        asymmetry = (

            features["impact_asymmetry_pct"]

        )


        if (

            asymmetry is None

            or

            asymmetry < ASYMMETRY_WARN_THRESHOLD_PCT

        ):

            return



        now = time.time()


        if (

            now -

            self.last_feedback_time

            <

            FEEDBACK_COOLDOWN_SEC

        ):

            return



        self.last_feedback_time = now


        feedback = {


            "type":
            "ASYMMETRY",


            "severity":
            "WARNING",


            "value":
            asymmetry,


            "timestamp":
            now

        }


        logger.warning(
            "Feedback triggered: %s",
            feedback
        )


        if self.on_feedback:

            self.on_feedback(feedback)