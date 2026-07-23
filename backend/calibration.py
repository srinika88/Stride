"""
StrideIQ — Waist IMU Calibration

The waist ESP32 + MPU9250 just came online (fresh solder joints), and
nothing in the codebase computed sensor bias before this. Every waist
feature that depends on integrating gyro/accel over time — pelvic tilt,
pelvic rotation, vertical oscillation — accumulates raw sensor bias
directly into its output otherwise. A gyro that reads 1.5 deg/s at rest
looks identical to real rotation to processing/features_waist.py's
complementary filter; it has no way to know that's just hardware offset.

This module has two jobs:
1. Collect a short "stand still" phase (gyro bias, accel bias) and a
   short "rotate through every orientation" phase (magnetometer hard-iron
   offset), over the SAME BLE connection backend/ble/waist_node.py
   already uses.
2. Persist the result to data/waist_calibration.json, and offer
   apply_calibration() so raw packets can be corrected before they ever
   reach features_waist.py.

This is intentionally NOT wired into waist_node.py automatically —
that's a one-line integration you can add once you've confirmed a
calibration run looks sane, rather than something silently applied to
data you haven't inspected yet. See "Wiring it in" below.

Scope note: this does hard-iron (offset) magnetometer correction only,
not soft-iron (scale/shear) correction. Soft-iron calibration needs an
ellipsoid fit across many orientations and is overkill for a project at
this scale — hard-iron alone removes most of the heading error a
tilt-compensated compass sees indoors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np

from backend import config
from backend.ble.waist_node import WaistNode, WaistPacket

logger = logging.getLogger("strideiq.calibration")

DEFAULT_CALIBRATION_PATH = os.path.join(config.DATA_DIR, "waist_calibration.json")

STATIONARY_SECONDS = 5      # "hold still" phase — gyro + accel bias
ROTATION_SECONDS = 15       # "slowly rotate through every axis" phase — magnetometer

# A stationary MPU9250 at 50Hz should give ~250 samples in 5s; below
# this we don't trust the bias estimate enough to save it.
MIN_STATIONARY_SAMPLES = 100
MIN_ROTATION_SAMPLES = 200


async def _collect_packets(seconds: float, prompt: str) -> list[WaistPacket]:
    """
    Connects to the waist node, collects packets for `seconds`, then
    disconnects. Reuses WaistNode as-is via its callback hook — no
    changes needed to backend/ble/waist_node.py.
    """
    packets: list[WaistPacket] = []
    node = WaistNode(parent=None, callback=packets.append)

    address = await node.find_device()
    if address is None:
        raise RuntimeError(
            "Could not find the waist node over BLE. Check it's powered "
            "on and advertising, and that config.BLE_DEVICE_NAMES['waist'] "
            "matches the name set in firmware/waist/config.h."
        )

    print(f"\n{prompt}")
    print(f"Collecting for {seconds:.0f}s...")

    connect_task = asyncio.create_task(node.run())
    await asyncio.sleep(seconds)
    await node.disconnect()

    try:
        await asyncio.wait_for(connect_task, timeout=3)
    except asyncio.TimeoutError:
        connect_task.cancel()

    return packets


def _compute_gyro_accel_bias(packets: list[WaistPacket]) -> dict:
    if len(packets) < MIN_STATIONARY_SAMPLES:
        raise RuntimeError(
            f"Only got {len(packets)} stationary samples, need at least "
            f"{MIN_STATIONARY_SAMPLES}. Check the BLE connection held for "
            f"the full {STATIONARY_SECONDS}s window and try again."
        )

    gx = np.array([p.gx for p in packets])
    gy = np.array([p.gy for p in packets])
    gz = np.array([p.gz for p in packets])

    ax = np.array([p.ax for p in packets])
    ay = np.array([p.ay for p in packets])
    az = np.array([p.az for p in packets])

    # Gyro at rest should read ~0 on every axis — whatever it actually
    # reads IS the bias to subtract.
    gyro_bias = [float(gx.mean()), float(gy.mean()), float(gz.mean())]

    # Accel bias assumes the waist band sits with its Z axis roughly
    # vertical at rest (typical belt-mounted orientation) — X/Y should
    # read ~0, Z should read ~1 unit of gravity in whatever scale the
    # firmware reports. We only correct X/Y here; Z is left alone
    # deliberately, since "what should Z read at rest" depends on the
    # exact mounting angle and guessing wrong would inject a worse error
    # than leaving it uncorrected. If your waist band sits at a
    # noticeable angle, re-check this assumption before trusting it.
    accel_bias_xy = [float(ax.mean()), float(ay.mean())]
    accel_z_mean = float(az.mean())

    logger.info(
        f"Gyro bias (deg/s or rad/s, whatever firmware units are): {gyro_bias}"
    )
    logger.info(
        f"Accel X/Y bias: {accel_bias_xy}, mean Z at rest: {accel_z_mean:.3f} "
        f"(compare against ~9.81 if firmware outputs m/s^2, or ~1.0 if it "
        f"outputs g's — this tells you which unit convention you're actually "
        f"in, worth checking against features_waist.py's assumption)"
    )

    return {
        "gyro_bias": gyro_bias,
        "accel_bias_xy": accel_bias_xy,
        "accel_z_at_rest": accel_z_mean,
        "n_samples": len(packets),
    }


def _compute_mag_offset(packets: list[WaistPacket]) -> dict:
    if len(packets) < MIN_ROTATION_SAMPLES:
        raise RuntimeError(
            f"Only got {len(packets)} rotation samples, need at least "
            f"{MIN_ROTATION_SAMPLES}. Rotate the sensor more slowly/for "
            f"longer through as many orientations as you can and try again."
        )

    mx = np.array([p.mx for p in packets])
    my = np.array([p.my for p in packets])
    mz = np.array([p.mz for p in packets])

    # Hard-iron offset: the magnetometer reading is a sphere centered on
    # (offset_x, offset_y, offset_z) rather than the origin, distorted by
    # nearby ferrous material/magnets on the PCB. Center of the observed
    # min/max range on each axis approximates that offset.
    mag_offset = [
        float((mx.max() + mx.min()) / 2),
        float((my.max() + my.min()) / 2),
        float((mz.max() + mz.min()) / 2),
    ]

    spread = [float(mx.max() - mx.min()), float(my.max() - my.min()), float(mz.max() - mz.min())]
    if min(spread) < 5:
        logger.warning(
            f"Magnetometer axis spread during rotation was small ({spread}) — "
            "this usually means the sensor wasn't rotated through enough "
            "orientations. The hard-iron offset below may be unreliable; "
            "consider re-running with a fuller rotation."
        )

    logger.info(f"Magnetometer hard-iron offset: {mag_offset}")

    return {
        "mag_offset": mag_offset,
        "mag_axis_spread": spread,
        "n_samples": len(packets),
    }


async def run_calibration() -> dict:
    """
    Full calibration: stand still (always), then rotate — SKIPPED if this
    waist board has no magnetometer (confirmed via WHO_AM_I = 0x70, an
    MPU6500 rather than a genuine MPU9250). firmware/waist/main.cpp sends
    literal "nan" for mx/my/mz on this hardware, so there's nothing to
    calibrate in that phase — asking you to rotate the sensor for 15s
    would just be collecting NaN. processing/features_waist.py already
    falls back to gyro-only pelvic rotation whenever mx/my/mz are NaN, so
    this isn't a missing feature, just a hardware-driven accuracy tradeoff
    (gyro-only yaw drifts uncorrected over longer sessions).
    """
    stationary_packets = await _collect_packets(
        STATIONARY_SECONDS,
        prompt="Phase 1/2 — set the waist node down flat and DO NOT MOVE IT.",
    )
    bias = _compute_gyro_accel_bias(stationary_packets)

    has_magnetometer = not all(
        p.mx != p.mx for p in stationary_packets  # NaN != NaN is True
    )

    if not has_magnetometer:
        logger.info(
            "No magnetometer data detected (all NaN) — skipping the rotation "
            "phase. Pelvic rotation will run gyro-only on this hardware; see "
            "features_waist.py's has_mag fallback."
        )
        return {
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_magnetometer": False,
            **bias,
        }

    rotation_packets = await _collect_packets(
        ROTATION_SECONDS,
        prompt=(
            "Phase 2/2 — pick the waist node up and SLOWLY rotate it through "
            "as many orientations as you can (tumble it in all 3 axes, don't "
            "just spin it flat on a table)."
        ),
    )
    mag = _compute_mag_offset(rotation_packets)

    return {
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_magnetometer": True,
        **bias,
        **mag,
    }


def save_calibration(calib: dict, path: str = None) -> str:
    path = path or DEFAULT_CALIBRATION_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    logger.info(f"Saved waist calibration to {path}")
    return path


def load_calibration(path: str = None) -> dict | None:
    path = path or DEFAULT_CALIBRATION_PATH
    if not os.path.exists(path):
        logger.warning(f"No calibration file at {path} — run backend/calibration.py first")
        return None
    with open(path) as f:
        return json.load(f)


def apply_calibration(ax, ay, az, gx, gy, gz, mx, my, mz, calib: dict):
    """
    Applies a saved calibration dict to one raw 9-axis reading. Call this
    on each WaistPacket's fields before handing them to
    processing/features_waist.py — e.g. inside whatever consumes the
    packet callback in backend/mqtt or backend/glue.py.

    Returns the same 9 values, bias/offset-corrected. Does not touch
    accel Z (see _compute_gyro_accel_bias's note on why) or apply any
    magnetometer scale correction (hard-iron offset only, see module
    docstring).
    """
    gx = gx - calib["gyro_bias"][0]
    gy = gy - calib["gyro_bias"][1]
    gz = gz - calib["gyro_bias"][2]

    ax = ax - calib["accel_bias_xy"][0]
    ay = ay - calib["accel_bias_xy"][1]
    # az intentionally left uncorrected

    if calib.get("has_magnetometer", True) and "mag_offset" in calib:
        mx = mx - calib["mag_offset"][0]
        my = my - calib["mag_offset"][1]
        mz = mz - calib["mag_offset"][2]
    # else: no magnetometer on this hardware — mx/my/mz are already NaN
    # coming from firmware, leave them as-is rather than subtracting
    # into a NaN offset that doesn't exist.

    return ax, ay, az, gx, gy, gz, mx, my, mz


# ============================================================
# Wiring it in (do this manually once you trust a calibration run):
#
# In whatever module owns the WaistNode callback (backend/glue.py or
# backend/mqtt/feedback_loop.py), near the top:
#
#     from backend.calibration import load_calibration, apply_calibration
#     _waist_calib = load_calibration()
#
# Then in the callback that receives each WaistPacket, before it's
# passed onward:
#
#     if _waist_calib:
#         packet.ax, packet.ay, packet.az, packet.gx, packet.gy, packet.gz, \
#             packet.mx, packet.my, packet.mz = apply_calibration(
#                 packet.ax, packet.ay, packet.az,
#                 packet.gx, packet.gy, packet.gz,
#                 packet.mx, packet.my, packet.mz,
#                 _waist_calib,
#             )
#
# Left as a manual step rather than done automatically here, since it
# touches the live data path and is worth reviewing once rather than
# silently changing behavior on a deadline day.
# ============================================================


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    print("StrideIQ waist calibration")
    print("=" * 40)
    try:
        result = asyncio.run(run_calibration())
    except RuntimeError as e:
        print(f"\nCalibration failed: {e}")
    else:
        path = save_calibration(result)
        print(f"\nDone. Calibration saved to {path}")
        print(json.dumps(result, indent=2)) 