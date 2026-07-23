"""
StrideIQ Dataset Preprocessor

Supports preprocessing multiple public gait datasets into the
StrideIQ schema.

Pipeline

Inspect
    ↓
Reshape
    ↓
Normalize Units
    ↓
Validate
    ↓
Export

Supported datasets

- GaitMotion
- Camargo EPIC Lab
- Custom CSV

Output schema matches raw_sensor_data so public data can be used
for pretraining, benchmarking and experimentation.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("strideiq.dataset")

logging.basicConfig(level=logging.INFO)


class DatasetPreprocessor:

    """
    Generic dataset preprocessing pipeline.
    """

    SUPPORTED_DATASETS = [

        "gaitmotion",

        "camargo",

        "nonan",

        "custom"

    ]

    # --------------------------------------------------

    def __init__(

        self,

        dataset,

        input_path,

        output_dir="data/processed"

    ):

        dataset = dataset.lower()

        if dataset not in self.SUPPORTED_DATASETS:

            raise ValueError(

                f"Unsupported dataset: {dataset}"

            )

        self.dataset = dataset

        self.input_path = input_path

        self.output_dir = output_dir

        self.df = None

        self.processed = None

        self.report = {}

    # ==================================================
    # LOAD
    # ==================================================

    def load(self):

        logger.info(

            f"Loading {self.input_path}"

        )

        self.df = pd.read_csv(self.input_path)

        logger.info(

            f"{len(self.df)} rows loaded"

        )

        return self.df

    # ==================================================
    # INSPECT
    # ==================================================

    def inspect(self):

        if self.df is None:

            self.load()

        print("\n========== DATASET ==========")

        print(f"Dataset : {self.dataset}")

        print(f"Rows    : {len(self.df)}")

        print(f"Columns : {len(self.df.columns)}")

        print()

        print(self.df.columns.tolist())

        print()

        print(self.df.head())

        print("\n=============================\n")

        self.report["rows"] = len(self.df)

        self.report["columns"] = len(self.df.columns)

        return self.report

    # ==================================================
    # COLUMN MAPS
    # ==================================================

    def column_mapping(self):

        """
        Map dataset columns into StrideIQ schema.
        """

        if self.dataset == "gaitmotion":

            return {

                "timestamp": "time",

                "ax": "acc_x",

                "ay": "acc_y",

                "az": "acc_z",

                "gx": "gyro_x",

                "gy": "gyro_y",

                "gz": "gyro_z"

            }

        elif self.dataset == "camargo":

            return {

                "timestamp": "Time",

                "ax": "AccelX",

                "ay": "AccelY",

                "az": "AccelZ",

                "gx": "GyroX",

                "gy": "GyroY",

                "gz": "GyroZ"

            }

        elif self.dataset == "nonan":

            # Pelvis segment only — this is StrideIQ's waist node analog.
            # AXIS REMAP, not a straight rename: NONAN's local sensor
            # frame has X as the vertical/gravity axis (confirmed from
            # the raw data at rest: X~0.96g, Z~0.31g — NOT the Z-is-
            # vertical convention features_waist.py's accelerometer_pitch/
            # roll assume, which matches StrideIQ's own MPU9250 waist
            # mounting). Feeding X straight into "ax" produced pelvic_tilt
            # values around 71-74 degrees — physiologically impossible
            # for running/walking (real range is roughly 5-20 degrees).
            # Remapping NONAN-X -> az (vertical), NONAN-Y -> ax,
            # NONAN-Z -> ay fixes this. The gyro triad is rotated the
            # same way to keep the frame internally consistent.
            #
            # Caveat: this permutation is inferred from the data itself,
            # not from NONAN's own documentation/sensor-placement diagram
            # — worth a spot-check against their docs if available, since
            # only the vertical axis was empirically confirmable from a
            # single static-ish frame (the ax/ay lateral-vs-forward
            # assignment is a reasonable guess, not verified).
            return {

                "timestamp": "time",

                "az": "Pelvis Accel Sensor X (mG)",

                "ax": "Pelvis Accel Sensor Y (mG)",

                "ay": "Pelvis Accel Sensor Z (mG)",

                "gz": "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-x (deg/s)",

                "gx": "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-y (deg/s)",

                "gy": "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-z (deg/s)",

            }

        else:

            raise NotImplementedError(

                "Custom datasets must override column_mapping()."

            )

    # ==================================================
    # RESHAPE
    # ==================================================

    def reshape(

        self,

        assumed_node="foot_left",

        source_label=None

    ):

        if self.df is None:

            self.load()

        mapping = self.column_mapping()

        missing = [

            source

            for source in mapping.values()

            if source not in self.df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns: {missing}"

            )

        out = pd.DataFrame()

        for target, source in mapping.items():

            out[target] = self.df[source]

        # fields absent from many datasets

        out["mx"] = np.nan
        out["my"] = np.nan
        out["mz"] = np.nan

        out["packet_seq"] = np.arange(len(out))

        out["assumed_node"] = assumed_node

        if source_label is None:

            source_label = self.dataset

        out["source"] = source_label

        self.processed = out

        logger.info(

            "Reshaping complete."

        )

        return out 
        # ==================================================
    # SAMPLING RATE
    # ==================================================

    def detect_sampling_rate(self):

        """
        Estimate sampling frequency from timestamps.
        """

        if self.processed is None:
            raise RuntimeError("Call reshape() first.")

        ts = self.processed["timestamp"].values

        dt = np.diff(ts)

        dt = dt[dt > 0]

        if len(dt) == 0:
            raise ValueError("Unable to estimate sampling rate.")

        median_dt = np.median(dt)

        fs = 1.0 / median_dt

        self.report["sampling_rate"] = fs

        logger.info(
            f"Detected sampling rate: {fs:.2f} Hz"
        )

        return fs

    # ==================================================
    # TIMESTAMP NORMALIZATION
    # ==================================================

    def normalize_timestamps(self):

        """
        Make timestamps start at zero.
        """

        if self.processed is None:
            raise RuntimeError("Call reshape() first.")

        first = self.processed["timestamp"].iloc[0]

        self.processed["timestamp"] -= first

        logger.info("Normalized timestamps.")

    # ==================================================
    # RESAMPLE
    # ==================================================

    def resample(self, target_rate: float):
        """
        Downsamples to target_rate via simple decimation (keep every
        Nth row). Needed because features_waist.py's Butterworth filter
        cutoffs are calculated against a single fixed assumed sampling
        rate (config.DEFAULT_SAMPLING_RATE) — feeding it data recorded
        at a different rate without resampling first would silently
        miscalculate every filter cutoff.

        Decimation (not interpolation) is used deliberately: it's a
        low-pass-then-downsample operation in effect for smoothly
        varying IMU signals at this ratio, and simpler to reason about
        than resampling for the purposes of just getting the rate to
        match. If the ratio isn't a clean divisor, drops the remainder.
        """
        if self.processed is None:
            raise RuntimeError("Call reshape() first.")

        current_rate = self.report.get("sampling_rate")
        if current_rate is None:
            current_rate = self.detect_sampling_rate()

        ratio = current_rate / target_rate
        if ratio < 1:
            raise ValueError(
                f"Cannot resample from {current_rate:.1f}Hz up to {target_rate}Hz "
                f"via decimation — source rate must be higher than target."
            )

        step = round(ratio)
        if abs(ratio - step) > 0.05:
            logger.warning(
                f"Sampling rate ratio {ratio:.3f} isn't close to a whole "
                f"number ({current_rate:.1f}Hz -> {target_rate}Hz) — "
                f"decimating by {step} anyway, but the resulting rate "
                f"({current_rate/step:.1f}Hz) won't exactly match target."
            )

        self.processed = self.processed.iloc[::step].reset_index(drop=True)
        self.report["resampled_from_hz"] = current_rate
        self.report["resampled_to_hz"] = current_rate / step
        logger.info(f"Resampled {current_rate:.1f}Hz -> {current_rate/step:.1f}Hz (decimation step={step})")

    # ==================================================
    # UNIT CONVERSION
    # ==================================================

    def convert_units(

        self,

        accel_unit="g",

        gyro_unit="deg"

    ):

        """
        Convert IMU units into SI units.

        Acceleration
            g -> m/s²

        Gyroscope
            deg/s -> rad/s
        """

        if self.processed is None:
            raise RuntimeError("Call reshape() first.")

        accel_cols = [

            "ax",

            "ay",

            "az"

        ]

        gyro_cols = [

            "gx",

            "gy",

            "gz"

        ]

        if accel_unit.lower() == "g":

            for c in accel_cols:

                self.processed[c] *= 9.80665

        elif accel_unit.lower() == "mg":

            for c in accel_cols:

                self.processed[c] *= 9.80665 / 1000.0

        if gyro_unit.lower() == "deg":

            for c in gyro_cols:

                self.processed[c] = np.deg2rad(

                    self.processed[c]

                )

        logger.info("Converted units.")

    # ==================================================
    # REMOVE DUPLICATES
    # ==================================================

    def remove_duplicates(self):

        before = len(self.processed)

        self.processed.drop_duplicates(

            subset="timestamp",

            inplace=True

        )

        removed = before - len(self.processed)

        logger.info(

            f"Removed {removed} duplicate rows."

        )

    # ==================================================
    # SORT
    # ==================================================

    def sort(self):

        self.processed.sort_values(

            "timestamp",

            inplace=True

        )

        self.processed.reset_index(

            drop=True,

            inplace=True

        )

    # ==================================================
    # INTERPOLATE
    # ==================================================

    def interpolate_missing(self):

        """
        Fill missing sensor values.

        Linear interpolation works well for IMU data.
        """

        sensor_cols = [

            "ax",

            "ay",

            "az",

            "gx",

            "gy",

            "gz"

        ]

        self.processed[sensor_cols] = (

            self.processed[sensor_cols]

            .interpolate(

                method="linear"

            )

            .bfill()

            .ffill()

        )

        logger.info(

            "Interpolated missing values."

        )

    # ==================================================
    # NORMALIZATION
    # ==================================================

    def normalize(

        self,

        method="zscore"

    ):

        """
        Normalize sensor channels.

        zscore
        minmax
        """

        sensor_cols = [

            "ax",

            "ay",

            "az",

            "gx",

            "gy",

            "gz"

        ]

        if method == "zscore":

            for c in sensor_cols:

                mean = self.processed[c].mean()

                std = self.processed[c].std()

                if std != 0:

                    self.processed[c] = (

                        self.processed[c] - mean

                    ) / std

        elif method == "minmax":

            for c in sensor_cols:

                mn = self.processed[c].min()

                mx = self.processed[c].max()

                if mx != mn:

                    self.processed[c] = (

                        self.processed[c] - mn

                    ) / (mx - mn)

        else:

            raise ValueError(

                "Unknown normalization method."

            )

        logger.info(

            f"Applied {method} normalization."

        )
            # ==================================================
    # VALIDATION
    # ==================================================

    def validate(self):

        """
        Performs sanity checks on the processed dataset.
        """

        if self.processed is None:
            raise RuntimeError("Call reshape() first.")

        report = {}

        report["rows"] = len(self.processed)

        report["missing_values"] = int(
            self.processed.isna().sum().sum()
        )

        report["duplicate_timestamps"] = int(
            self.processed["timestamp"].duplicated().sum()
        )

        report["timestamp_monotonic"] = bool(
            self.processed["timestamp"].is_monotonic_increasing
        )

        sensor_cols = [

            "ax","ay","az",

            "gx","gy","gz"

        ]

        report["sensor_ranges"] = {}

        for c in sensor_cols:

            report["sensor_ranges"][c] = {

                "min": float(self.processed[c].min()),

                "max": float(self.processed[c].max())

            }

        self.report.update(report)

        logger.info("Validation complete.")

        return report

    # ==================================================
    # TRAIN / VAL / TEST SPLIT
    # ==================================================

    def split(

        self,

        train_ratio=0.70,

        val_ratio=0.15

    ):

        if self.processed is None:

            raise RuntimeError("Call reshape() first.")

        n = len(self.processed)

        train_end = int(n * train_ratio)

        val_end = train_end + int(n * val_ratio)

        train = self.processed.iloc[:train_end]

        val = self.processed.iloc[train_end:val_end]

        test = self.processed.iloc[val_end:]

        logger.info(

            f"Split dataset "

            f"{len(train)} / "

            f"{len(val)} / "

            f"{len(test)}"

        )

        return train, val, test

    # ==================================================
    # EXPORT
    # ==================================================

    def export(

        self,

        filename=None

    ):

        if self.processed is None:

            raise RuntimeError("Nothing to export.")

        os.makedirs(

            self.output_dir,

            exist_ok=True

        )

        if filename is None:

            filename = (

                f"{self.dataset}_processed.csv"

            )

        path = os.path.join(

            self.output_dir,

            filename

        )

        self.processed.to_csv(

            path,

            index=False

        )

        logger.info(

            f"Saved {path}"

        )

        return path

    # ==================================================
    # QUALITY REPORT
    # ==================================================

    def print_report(self):

        print("\n")

        print("=" * 60)

        print("StrideIQ Dataset Report")

        print("=" * 60)

        for k, v in self.report.items():

            print(f"{k}: {v}")

        print("=" * 60)

        print()

    # ==================================================
    # COMPLETE PIPELINE
    # ==================================================

    def run_pipeline(

        self,

        assumed_node="foot_left",

        source_label=None,

        accel_unit="g",

        gyro_unit="deg",

        normalization="zscore"

    ):

        """
        Executes the complete preprocessing pipeline.
        """

        self.load()

        self.inspect()

        self.reshape(

            assumed_node,

            source_label

        )

        self.detect_sampling_rate()

        self.normalize_timestamps()

        self.convert_units(

            accel_unit,

            gyro_unit

        )

        self.remove_duplicates()

        self.sort()

        self.interpolate_missing()

        self.normalize(

            normalization

        )

        self.validate()

        output = self.export()

        self.print_report()

        return output 