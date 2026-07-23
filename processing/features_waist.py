"""
StrideIQ — Waist Feature Extraction (Part 1, corrected)

Computes biomechanics features from the waist IMU.

Fixes applied vs. the original draft:
- SAMPLING_RATE now imported from config.py (was hardcoded to 100,
  actual project runs at 50Hz — every filter cutoff was silently wrong)
- Accelerometer units converted g -> m/s^2 before any physical
  integration (firmware/MPU9250 outputs g's; oscillation math needs
  SI units or displacement comes out wrong by a factor of ~9.81)
- Gravity removal and magnetometer heading now use the complementary
  filter's pitch/roll estimate for tilt compensation, instead of
  ignoring it
- Short-window guard added before any filtfilt call, to avoid a hard
  crash on partial/short windows (BLE dropouts, trailing windows)
- Pelvic tilt/rotation now return both mean (average posture) and
  range (variability) — range is often the more clinically relevant
  signal, and returning only one was a silent design choice before
"""

from __future__ import annotations

import numpy as np
import logging

from scipy.signal import butter, filtfilt

from backend import config 

logger = logging.getLogger("strideiq.features_waist")


# =====================================================
# Configuration — pulled from config.py, not redefined here
# =====================================================

SAMPLING_RATE = config.DEFAULT_SAMPLING_RATE  # was hardcoded to 100, now matches the real 50Hz
DT = 1 / SAMPLING_RATE
GRAVITY = 9.81
ALPHA = 0.98

# Minimum samples needed for a stable 4th-order filtfilt call.
# Below this, filtering is skipped and the raw signal is used as-is,
# rather than crashing.
MIN_FILTER_SAMPLES = 15


# =====================================================
# Butterworth Filters — with a short-window guard
# =====================================================

def _safe_filtfilt(b, a, signal):
    if len(signal) < MIN_FILTER_SAMPLES:
        logger.warning(
            f"Signal too short for filtering ({len(signal)} samples, "
            f"need >= {MIN_FILTER_SAMPLES}) — returning unfiltered"
        )
        return np.asarray(signal)
    return filtfilt(b, a, signal)


def lowpass(signal, cutoff=5):
    nyquist = SAMPLING_RATE / 2
    b, a = butter(4, cutoff / nyquist, btype="low")
    return _safe_filtfilt(b, a, signal)


def highpass(signal, cutoff=0.25):
    nyquist = SAMPLING_RATE / 2
    b, a = butter(4, cutoff / nyquist, btype="high")
    return _safe_filtfilt(b, a, signal)


# =====================================================
# Accelerometer Pitch / Roll
# =====================================================

def accelerometer_pitch(ax, ay, az):
    return np.degrees(np.arctan2(ax, np.sqrt(ay**2 + az**2)))


def accelerometer_roll(ax, ay, az):
    return np.degrees(np.arctan2(ay, az))


# =====================================================
# Complementary Filter — unchanged, this part was correct
# =====================================================

def complementary_filter(ax, ay, az, gx, gy):
    pitch = np.zeros(len(ax))
    roll = np.zeros(len(ax))

    pitch[0] = accelerometer_pitch(ax[0], ay[0], az[0])
    roll[0] = accelerometer_roll(ax[0], ay[0], az[0])

    for i in range(1, len(ax)):
        gyro_pitch = pitch[i - 1] + gy[i] * DT
        gyro_roll = roll[i - 1] + gx[i] * DT

        acc_pitch = accelerometer_pitch(ax[i], ay[i], az[i])
        acc_roll = accelerometer_roll(ax[i], ay[i], az[i])

        pitch[i] = ALPHA * gyro_pitch + (1 - ALPHA) * acc_pitch
        roll[i] = ALPHA * gyro_roll + (1 - ALPHA) * acc_roll

    return pitch, roll


# =====================================================
# Gravity Compensation — now tilt-compensated using pitch/roll
# =====================================================

def remove_gravity_tilt_compensated(az_ms2, pitch_deg, roll_deg):
    """
    Estimates the gravity component actually projected onto the
    device's z-axis given its current tilt, using the pitch/roll
    the complementary filter already computed — rather than assuming
    gravity is always ~9.81 straight down the z-axis regardless of
    orientation (which breaks down exactly when the waist tilts,
    i.e. the thing pelvic tilt is supposed to measure).
    """
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)
    gravity_component = GRAVITY * np.cos(pitch_rad) * np.cos(roll_rad)
    return az_ms2 - gravity_component


# =====================================================
# Vertical Acceleration / Oscillation
# =====================================================

def vertical_acceleration(az_g, pitch_deg, roll_deg):
    """az_g: raw accelerometer z-axis reading in g's (firmware units)."""
    az_ms2 = np.asarray(az_g) * GRAVITY  # g -> m/s^2, was missing before
    vertical = remove_gravity_tilt_compensated(az_ms2, pitch_deg, roll_deg)
    return highpass(vertical, cutoff=0.25)


def compute_vertical_oscillation(az_g, pitch_deg, roll_deg):
    vertical = vertical_acceleration(az_g, pitch_deg, roll_deg)

    velocity = np.cumsum(vertical) * DT
    velocity = highpass(velocity, cutoff=0.2)

    displacement = np.cumsum(velocity) * DT
    displacement = highpass(displacement, cutoff=0.2)

    oscillation_m = np.max(displacement) - np.min(displacement)
    return float(oscillation_m * 100)  # m -> cm, now correctly scaled


# =====================================================
# Pelvic Tilt — mean AND range
# =====================================================

def compute_pelvic_tilt(pitch):
    return {
        "mean": float(np.mean(pitch)),
        "range": float(np.max(pitch) - np.min(pitch)),
    }


# =====================================================
# Magnetometer Heading — tilt-compensated
# =====================================================

def tilt_compensated_heading(mx, my, mz, pitch_deg, roll_deg):
    """
    Raw atan2(my, mx) only gives a correct heading when the sensor is
    level. Since the waist tilts during running, that assumption breaks
    exactly when it matters — this applies the standard tilt-compensated
    compass formula using the pitch/roll already computed.
    """
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    mx_comp = (mx * np.cos(pitch_rad)) + (mz * np.sin(pitch_rad))
    my_comp = (
        mx * np.sin(roll_rad) * np.sin(pitch_rad)
        + my * np.cos(roll_rad)
        - mz * np.sin(roll_rad) * np.cos(pitch_rad)
    )
    return np.degrees(np.arctan2(my_comp, mx_comp))


# =====================================================
# Pelvic Rotation — mean AND range, tilt-compensated
# =====================================================

def compute_pelvic_rotation(gz, mx, my, mz, pitch_deg, roll_deg):
    has_mag = not (np.isnan(mx).all() or np.isnan(my).all() or np.isnan(mz).all())

    yaw = np.zeros(len(gz))

    if has_mag:
        heading_vals = tilt_compensated_heading(mx, my, mz, pitch_deg, roll_deg)
        yaw[0] = heading_vals[0]
        for i in range(1, len(gz)):
            gyro = yaw[i - 1] + gz[i] * DT
            mag = heading_vals[i]
            yaw[i] = ALPHA * gyro + (1 - ALPHA) * mag
    else:
        # No magnetometer (no hardware, or a dataset like NONAN with no
        # mag columns) -- pure gyro dead-reckoning. Drifts over longer
        # windows with no compass correction, but degrades gracefully
        # instead of crashing; still usable for relative rotation range
        # within a single short window.
        logger.warning("No magnetometer data available -- pelvic rotation is gyro-only (uncorrected, may drift)")
        yaw = np.cumsum(gz) * DT

    return {
        "mean": float(np.mean(yaw)),
        "range": float(np.max(yaw) - np.min(yaw)),
    }


# =====================================================
# Prepare IMU Arrays — unchanged, this part was correct
# =====================================================

def prepare_arrays(readings):
    ax = np.asarray([r["ax"] for r in readings], dtype=float)
    ay = np.asarray([r["ay"] for r in readings], dtype=float)
    az = np.asarray([r["az"] for r in readings], dtype=float)
    gx = np.asarray([r["gx"] for r in readings], dtype=float)
    gy = np.asarray([r["gy"] for r in readings], dtype=float)
    gz = np.asarray([r["gz"] for r in readings], dtype=float)
    # Magnetometer may be entirely absent (no hardware, or a dataset like
    # NONAN with no mag columns) -- readings hold None in that case.
    # float(None) would raise, so route missing values through np.nan.
    mx = np.asarray([np.nan if r["mx"] is None else r["mx"] for r in readings], dtype=float)
    my = np.asarray([np.nan if r["my"] is None else r["my"] for r in readings], dtype=float)
    mz = np.asarray([np.nan if r["mz"] is None else r["mz"] for r in readings], dtype=float)
    return ax, ay, az, gx, gy, gz, mx, my, mz


# =====================================================
# PART 2 — Smoothness, jerk, and stability metrics
# =====================================================

def compute_cadence(ax, ay, az) -> float:
    """
    Cadence from the waist signal, via FFT dominant frequency on the
    vertical-ish acceleration magnitude. This is a SECONDARY cadence
    estimate — foot-derived cadence (from actual footstrike counts,
    in features_foot.py) is more accurate and should be preferred
    when both are available. This exists mainly as a cross-check, and
    as a fallback for windows where foot data is missing/dropped.
    """
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    magnitude = magnitude - np.mean(magnitude)  # remove DC offset before FFT

    if len(magnitude) < MIN_FILTER_SAMPLES:
        return None

    freqs = np.fft.rfftfreq(len(magnitude), d=DT)
    spectrum = np.abs(np.fft.rfft(magnitude))

    # Ignore near-0Hz (DC/drift) — running cadence is realistically 1.3-3.5 Hz
    valid = (freqs > 1.0) & (freqs < 4.0)
    if not np.any(valid):
        return None

    dominant_freq = freqs[valid][np.argmax(spectrum[valid])]
    return float(dominant_freq * 60)  # Hz -> steps/min


def compute_sparc(velocity: np.ndarray, sample_rate: float = SAMPLING_RATE) -> float:
    """
    SPARC (Spectral Arc Length) — a smoothness metric widely used in
    movement science. More negative = smoother movement, values closer
    to 0 = jerkier/less smooth. Operates on a velocity profile, so
    this expects the already-integrated vertical velocity signal.

    Standard implementation (Balasubramanian et al.) — this is a
    reasonably faithful port, but if you want to cite it directly,
    worth cross-checking against a reference implementation before
    trusting exact values in your evaluation writeup.
    """
    if len(velocity) < MIN_FILTER_SAMPLES:
        return None

    velocity = velocity - np.mean(velocity)
    n = len(velocity)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(velocity))

    if np.max(spectrum) == 0:
        return None

    spectrum_norm = spectrum / np.max(spectrum)

    # Restrict to a reasonable frequency band (0-10 Hz) and an amplitude
    # threshold, per the standard SPARC procedure
    cutoff_freq = 10.0
    amp_threshold = 0.05

    valid_idx = np.where((freqs <= cutoff_freq) & (spectrum_norm >= amp_threshold))[0]
    if len(valid_idx) < 2:
        return None

    f_sel = freqs[valid_idx]
    s_sel = spectrum_norm[valid_idx]

    arc_length = -np.sum(
        np.sqrt(np.diff(f_sel) ** 2 + np.diff(s_sel) ** 2)
    )
    return float(arc_length)


def compute_rms_jerk(vertical_accel_ms2: np.ndarray) -> float:
    """
    RMS jerk — rate of change of acceleration. Higher = more abrupt,
    less efficient movement transitions. Same "smoothness" family of
    metric as SPARC, computed differently (time-domain vs frequency-
    domain), useful as a cross-check against each other.
    """
    if len(vertical_accel_ms2) < 3:
        return None
    jerk = np.diff(vertical_accel_ms2) / DT
    return float(np.sqrt(np.mean(jerk ** 2)))


def compute_harmonic_ratio(vertical_accel_ms2: np.ndarray, cadence_hz: float) -> float:
    """
    Harmonic Ratio — compares even vs. odd harmonics of the stride
    frequency in the acceleration spectrum. Higher = more rhythmic,
    controlled movement; used in gait research as a stability/smoothness
    indicator. Requires a real cadence estimate — returns None if
    cadence couldn't be computed for this window (can't identify
    the fundamental stride frequency without it).
    """
    if cadence_hz is None or len(vertical_accel_ms2) < MIN_FILTER_SAMPLES:
        return None

    n = len(vertical_accel_ms2)
    freqs = np.fft.rfftfreq(n, d=DT)
    spectrum = np.abs(np.fft.rfft(vertical_accel_ms2 - np.mean(vertical_accel_ms2)))

    stride_freq = cadence_hz / 60  # steps/min -> Hz... but cadence here is
    # already in steps/min from compute_cadence, and a full gait cycle is
    # 2 steps, so the fundamental stride frequency is roughly half that
    stride_freq = stride_freq / 2

    def _amplitude_near(target_freq, tolerance=0.3):
        idx = np.argmin(np.abs(freqs - target_freq))
        if abs(freqs[idx] - target_freq) > tolerance:
            return 0.0
        return spectrum[idx]

    even_sum = sum(_amplitude_near(stride_freq * h) for h in [2, 4, 6, 8, 10])
    odd_sum = sum(_amplitude_near(stride_freq * h) for h in [1, 3, 5, 7, 9])

    if odd_sum == 0:
        return None

    return float(even_sum / odd_sum)


def compute_step_regularity(ax, ay, az) -> float:
    """
    Step regularity via autocorrelation — how well the signal
    repeats itself one stride later. Higher (closer to 1.0) = more
    regular/consistent stride pattern.
    """
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    magnitude = magnitude - np.mean(magnitude)

    if len(magnitude) < MIN_FILTER_SAMPLES:
        return None

    autocorr = np.correlate(magnitude, magnitude, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # keep non-negative lags only
    autocorr = autocorr / autocorr[0]  # normalize so lag-0 = 1.0

    # Look for the first strong secondary peak after lag 0 — that's the
    # stride-repetition peak. Search a plausible lag range for running
    # cadence (roughly 0.15s-0.6s between steps at 50Hz sampling).
    min_lag = int(0.15 * SAMPLING_RATE)
    max_lag = min(int(0.6 * SAMPLING_RATE), len(autocorr) - 1)

    if max_lag <= min_lag:
        return None

    peak_val = np.max(autocorr[min_lag:max_lag])
    return float(peak_val)


def compute_trunk_stability_index(pitch, roll) -> float:
    """
    A simple composite: lower combined variance in pitch/roll over
    the window = a more stable trunk. Not a standard named metric
    like SPARC/HR — a straightforward, defensible combination of
    what the complementary filter already gives you.
    """
    return float(np.std(pitch) + np.std(roll))


def compute_cadence_variability(cadence_values: list) -> float:
    """
    Coefficient of variation of cadence across multiple windows in a
    run (not within a single window — this needs to be called at the
    run level, after cadence has been computed per-window across the
    whole session). Lower = more consistent cadence throughout the run.
    """
    valid = [c for c in cadence_values if c is not None]
    if len(valid) < 2:
        return None
    mean_c = np.mean(valid)
    if mean_c == 0:
        return None
    return float(np.std(valid) / mean_c) 


# =====================================================
# Integration — one function, one window's worth of readings in,
# one feature dict out, matching the features table's columns
# =====================================================

def compute_tilt_baseline(readings: list[dict]) -> float:
    """
    readings: raw_sensor_data rows for the waist node, covering ONLY
    the calibration period (e.g. the first few seconds of a run, where
    the subject is standing still) -- not the whole run.

    Returns the mean pitch during that period, to be subtracted from
    every subsequent window's pelvic_tilt for this run.

    Why this exists: accelerometer_pitch() gives an absolute inclination
    relative to gravity, not a clinically-defined pelvic tilt angle --
    it has no idea what "neutral" looks like for this subject's mounting
    position. Two runs (or two subjects) with identical real pelvic
    tilt but slightly different sensor mounting angles would otherwise
    read as different tilt values. Comparing this pipeline's raw output
    against NONAN's own (subject-calibrated) Pelvic Tilt LT column
    showed essentially no correlation (-0.07) for exactly this reason --
    subtracting a per-run baseline is the fix.
    """
    if len(readings) < MIN_FILTER_SAMPLES:
        raise ValueError(
            f"Only {len(readings)} samples in the calibration window -- "
            f"need at least {MIN_FILTER_SAMPLES}. Make sure the subject "
            f"holds still for the full calibration duration."
        )

    ax, ay, az, gx, gy, gz, mx, my, mz = prepare_arrays(readings)
    pitch, _roll = complementary_filter(ax, ay, az, gx, gy)
    return float(np.mean(pitch))


def extract_waist_features(readings: list[dict], tilt_offset: float = 0.0) -> dict:
    """
    readings: raw_sensor_data rows for the waist node, ONE window.
    tilt_offset: baseline pitch (degrees) from compute_tilt_baseline(),
    subtracted from this window's tilt so pelvic_tilt reflects deviation
    from this run's calibrated neutral pose, not raw sensor inclination.
    Defaults to 0.0 (no calibration applied) so existing callers that
    don't pass it still work, just without the correction.
    Returns a dict with keys matching features table columns for
    every waist-derived metric. cadence_variability is NOT included
    here — it's a run-level metric, computed separately in the
    pipeline after all windows are processed (see features_pipeline.py).
    """
    if len(readings) < MIN_FILTER_SAMPLES:
        logger.warning(f"Window too short ({len(readings)} samples) for waist feature extraction")
        return _empty_waist_features()

    ax, ay, az, gx, gy, gz, mx, my, mz = prepare_arrays(readings)

    pitch, roll = complementary_filter(ax, ay, az, gx, gy)

    vertical_osc = compute_vertical_oscillation(az, pitch, roll)
    tilt = compute_pelvic_tilt(pitch)
    tilt["mean"] -= tilt_offset  # apply per-run calibration
    rotation = compute_pelvic_rotation(gz, mx, my, mz, pitch, roll)
    cadence = compute_cadence(ax, ay, az)

    vertical_accel = vertical_acceleration(az, pitch, roll)
    velocity = highpass(np.cumsum(vertical_accel) * DT, cutoff=0.2)

    sparc = compute_sparc(velocity)
    rms_jerk = compute_rms_jerk(vertical_accel)
    harmonic_ratio = compute_harmonic_ratio(vertical_accel, cadence)
    step_regularity = compute_step_regularity(ax, ay, az)
    trunk_stability = compute_trunk_stability_index(pitch, roll)

    return {
        "cadence": cadence,
        "vertical_oscillation": vertical_osc,
        "running_smoothness": sparc,  # SPARC maps onto the schema's running_smoothness column
        "pelvic_tilt": tilt["mean"],
        "pelvic_tilt_range": tilt["range"],
        "pelvic_rotation": rotation["mean"],
        "pelvic_rotation_range": rotation["range"],
        "sparc": sparc,
        "rms_jerk": rms_jerk,
        "harmonic_ratio": harmonic_ratio,
        "step_regularity": step_regularity,
        "trunk_stability_index": trunk_stability,
    }


def _empty_waist_features() -> dict:
    keys = ["cadence", "vertical_oscillation", "running_smoothness", "pelvic_tilt",
            "pelvic_tilt_range", "pelvic_rotation", "pelvic_rotation_range",
            "sparc", "rms_jerk", "harmonic_ratio", "step_regularity", "trunk_stability_index"]
    return {k: None for k in keys} 