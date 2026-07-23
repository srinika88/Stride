"""
StrideIQ Sensor Synchronization

Synchronizes

Left Foot
Right Foot
Waist

using Raspberry Pi timestamps.

Outputs synchronized windows ready for feature extraction.
"""

import numpy as np
import pandas as pd


TARGET_RATE = 50


class SensorSynchronizer:

    def __init__(

            self,

            target_rate=50

    ):

        self.target_rate = target_rate

        self.dt = 1 / target_rate


    # ---------------------------------------------

    def estimate_latency(

            self,

            df

    ):

        if "device_timestamp" not in df.columns:

            return 0

        latency = (

            df.timestamp

            -

            df.device_timestamp

        )

        return latency.median()


    # ---------------------------------------------

    def correct_latency(

            self,

            df

    ):

        if "device_timestamp" not in df.columns:

            return df

        latency = self.estimate_latency(df)

        df = df.copy()

        df.timestamp = (

            df.device_timestamp

            +

            latency

        )

        return df


    # ---------------------------------------------

    def interpolate_sensor(

            self,

            df

    ):

        df = df.sort_values("timestamp")

        start = df.timestamp.min()

        end = df.timestamp.max()

        timeline = np.arange(

            start,

            end,

            self.dt

        )

        output = pd.DataFrame()

        output["timestamp"] = timeline

        numeric = df.select_dtypes(

            include=np.number

        )

        for column in numeric.columns:

            if column == "timestamp":

                continue

            output[column] = np.interp(

                timeline,

                df.timestamp,

                df[column]

            )

        return output


    # ---------------------------------------------

    def synchronize(

            self,

            left,

            right,

            waist

    ):

        left = self.correct_latency(left)

        right = self.correct_latency(right)

        waist = self.correct_latency(waist)

        left = self.interpolate_sensor(left)

        right = self.interpolate_sensor(right)

        waist = self.interpolate_sensor(waist)

        start = max(

            left.timestamp.min(),

            right.timestamp.min(),

            waist.timestamp.min()

        )

        end = min(

            left.timestamp.max(),

            right.timestamp.max(),

            waist.timestamp.max()

        )

        left = left[

            (left.timestamp >= start)

            &

            (left.timestamp <= end)

        ].reset_index(drop=True)

        right = right[

            (right.timestamp >= start)

            &

            (right.timestamp <= end)

        ].reset_index(drop=True)

        waist = waist[

            (waist.timestamp >= start)

            &

            (waist.timestamp <= end)

        ].reset_index(drop=True)

        n = min(

            len(left),

            len(right),

            len(waist)

        )

        left = left.iloc[:n]

        right = right.iloc[:n]

        waist = waist.iloc[:n]

        return {

            "left": left,

            "right": right,

            "waist": waist

        }