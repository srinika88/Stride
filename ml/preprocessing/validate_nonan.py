"""
StrideIQ — NONAN Ground-Truth Validation
Compares StrideIQ's own computed pelvic_tilt / pelvic_rotation (from
features_waist.py, run on NONAN's raw Pelvis accel/gyro) against
NONAN's own published ground-truth columns (Pelvic Tilt LT, Pelvic
Rotation LT) for the same windows.

This is NOT part of the ingestion pipeline (glue.py) — reshape() drops
the ground-truth columns since they're not in the raw_sensor_data
schema. This script reloads the raw CSV independently so both sides
are available at once.

Why this matters specifically for pelvic_rotation: it's computed
gyro-only (no magnetometer in NONAN, or in most cheap MPU9250 setups
either), so it drifts over time — this script tells you how much,
instead of assuming.

Usage:
    python -m ml.preprocessing.validate_nonan --csv path/to/nonan_session.csv
"""

import argparse

import numpy as np
import pandas as pd

from backend import config
from processing.features_waist import (
    complementary_filter, compute_pelvic_tilt, compute_pelvic_rotation,
)


def _load_and_prep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "timestamp"})

    # Same axis remap as dataset_prep.py's "nonan" column_mapping —
    # X is vertical (confirmed by NONAN's own paper: "X-axis points
    # away from the ground"), remapped to az; Y/Z -> ax/ay.
    df["az"] = df["Pelvis Accel Sensor X (mG)"] * 9.80665 / 1000.0
    df["ax"] = df["Pelvis Accel Sensor Y (mG)"] * 9.80665 / 1000.0
    df["ay"] = df["Pelvis Accel Sensor Z (mG)"] * 9.80665 / 1000.0
    df["gz"] = np.radians(df["Noraxon MyoMotion-Segments-Pelvis-Gyroscope-x (deg/s)"])
    df["gx"] = np.radians(df["Noraxon MyoMotion-Segments-Pelvis-Gyroscope-y (deg/s)"])
    df["gy"] = np.radians(df["Noraxon MyoMotion-Segments-Pelvis-Gyroscope-z (deg/s)"])

    df["gt_tilt"] = df["Pelvic Tilt LT (deg)"]
    df["gt_rotation"] = df["Pelvic Rotation LT (deg)"]

    return df


def validate(csv_path: str, window_size: float = None, step: float = None):
    window_size = window_size or config.WINDOW_SIZE
    step = step or (config.WINDOW_SIZE - config.WINDOW_OVERLAP)

    df = _load_and_prep(csv_path)
    t_start, t_end = df["timestamp"].min(), df["timestamp"].max()

    rows = []
    t = t_start
    while t + window_size <= t_end:
        window = df[(df["timestamp"] >= t) & (df["timestamp"] < t + window_size)]
        if len(window) < 15:
            t += step
            continue

        ax, ay, az = window["ax"].values, window["ay"].values, window["az"].values
        gx, gy, gz = window["gx"].values, window["gy"].values, window["gz"].values

        pitch, roll = complementary_filter(ax, ay, az, gx, gy)
        tilt = compute_pelvic_tilt(pitch)

        # gyro-only rotation, matching what ingestion actually computes
        # for NONAN (no magnetometer available)
        nan_mag = np.full(len(gz), np.nan)
        rotation = compute_pelvic_rotation(np.degrees(gz), nan_mag, nan_mag, nan_mag, pitch, roll)

        rows.append({
            "t": t,
            "our_tilt": tilt["mean"],
            "gt_tilt": window["gt_tilt"].mean(),
            "our_rotation": rotation["mean"],
            "gt_rotation": window["gt_rotation"].mean(),
        })
        t += step

    result = pd.DataFrame(rows)
    result["tilt_error"] = result["our_tilt"] - result["gt_tilt"]
    result["rotation_error"] = result["our_rotation"] - result["gt_rotation"]

    print(f"\n{len(result)} windows compared\n")
    print("PELVIC TILT")
    print(f"  mean abs error: {result['tilt_error'].abs().mean():.2f} deg")
    print(f"  correlation with ground truth: {result['our_tilt'].corr(result['gt_tilt']):.3f}")
    print("\nPELVIC ROTATION (gyro-only, drift expected to grow with window count)")
    print(f"  mean abs error: {result['rotation_error'].abs().mean():.2f} deg")
    print(f"  error at window 1: {result['rotation_error'].iloc[0]:.2f} deg")
    print(f"  error at window {len(result)}: {result['rotation_error'].iloc[-1]:.2f} deg")
    print(f"  correlation with ground truth: {result['our_rotation'].corr(result['gt_rotation']):.3f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    validate(args.csv) 