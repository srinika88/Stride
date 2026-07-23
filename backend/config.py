"""
StrideIQ Configuration
Single source of truth for everything locked in on Day 0.
"""

import os

# ============================================================
# Environment
# ============================================================
ENV = os.getenv("STRIDEIQ_ENV", "development")  # 'development' or 'production'
DEBUG = ENV == "development"  # never hardcode True — real risk during a live demo

# ============================================================
# Database
# ============================================================
DATABASE_NAME = "strideiq.db"
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", DATABASE_NAME)


# ============================================================
# Flask / REST API
# ============================================================
HOST = "0.0.0.0"
PORT = 5000

API_ENDPOINTS = {
    "run_start": "/run/start",
    "run_stop": "/run/stop",
    "metrics_live": "/metrics/live",
    "runs": "/runs",
    "recommendations": "/recommendations",
}

# ============================================================
# MQTT
# ============================================================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
MQTT_QOS = 1  # at-least-once delivery — matters given you're already tracking packet loss

MQTT_TOPICS = {
    "raw_left": "strideiq/raw/left",
    "raw_right": "strideiq/raw/right",
    "raw_waist": "strideiq/raw/waist",
    "features": "strideiq/features",
    "predictions": "strideiq/predictions",
    "feedback": "strideiq/feedback",
}

# ============================================================
# BLE — device identifiers, locked on Day 0, referenced everywhere
# ============================================================
BLE_DEVICE_NAMES = {
    "waist": "StrideIQ-Waist",
    # MakeCode's Bluetooth extension has no way to set a custom BLE
    # advertised name (confirmed — no setName() in the JS/blocks API);
    # every micro:bit advertises as "BBC micro:bit [xxxxx]" with a fixed
    # 5-char code baked into that specific board. receiver.py's scan
    # does a substring match, so matching on just the code works fine —
    # confirmed via nRF Connect + reading each board's live UART data
    # (packets prefixed "L," / "R,") to identify which code is which.
    "foot_left": "vopev",
    "foot_right": "perev",
}
BLE_CONNECTION_TIMEOUT = 10.0    # seconds before a connection attempt is considered failed
BLE_RECONNECT_ATTEMPTS = 3
BLE_RECONNECT_BACKOFF_SEC = 2.0  # wait between retries — avoids hammering a flaky connection

# ============================================================
# Sensor / windowing
# ============================================================
WINDOW_SIZE = 2.0           # seconds
WINDOW_OVERLAP = 0.5        # seconds
DEFAULT_SAMPLING_RATE = 50  # Hz

# Nodes without gyro/magnetometer — drives which raw_sensor_data columns stay NULL
NODES_WITHOUT_ORIENTATION = ["foot_left", "foot_right"]

# Footstrike peak-detection threshold — start conservative, tune against real Day 3-4 test data
FOOTSTRIKE_PEAK_THRESHOLD_G = 2.0
MIN_PEAK_DISTANCE = 3

# ============================================================
# ML
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MODEL_VERSION = "v1"        # bump every retrain — predictions.model_version depends on this

CANDIDATE_MODELS = ["random_forest", "svm", "mlp"]

FORM_LABELS = ["Normal", "High Impact", "Asymmetry", "Excessive Pelvic Tilt"]
LABEL_SOURCES = ["mediapipe", "induced_manual", "heuristic"]

# Heuristic labeler thresholds — used only for label_source='heuristic',
# to bootstrap training labels on public IMU datasets with no video and
# no induced-fault protocol. Start conservative; retune once you have
# your own MediaPipe-labeled sessions to check these against.
HEURISTIC_HIGH_IMPACT_G = 3.5      # footstrike impact_intensity (m/s^2 equivalent) above this -> "High Impact"
HEURISTIC_ASYMMETRY_SCORE = 0.20   # symmetry_score (0=symmetric) above this -> "Asymmetry"
HEURISTIC_PELVIC_TILT_DEG = 8.0    # abs(pelvic_tilt), degrees, above this -> "Excessive Pelvic Tilt"
                                    # (normal running anterior pelvic tilt is roughly 4-7 deg;
                                    # 8 deg is a starting cutoff, not a validated clinical threshold —
                                    # retune once you have MediaPipe-labeled sessions to check against)
CALIBRATION_DURATION_SEC = 2.0     # seconds of still-standing at run start used to zero pelvic_tilt

N_CLUSTERS = 3               # K-Means, for drift-point detection
RETRAIN_MIN_NEW_SESSIONS = 3  # don't retrain on every single session — batch it

# Actuation trigger — when does the wearable actually buzz
ACTUATION_CONFIDENCE_THRESHOLD = 0.7
ACTUATION_LABELS = ["High Impact", "Asymmetry", "Excessive Pelvic Tilt"]  # "Normal" never triggers feedback

# ============================================================
# LangChain / LLM
# ============================================================
LLM_MODEL = os.getenv("STRIDEIQ_LLM_MODEL", "")  # set via env var, never hardcode a key or model name here
LLM_MAX_TOKENS = 400  # coaching messages should stay short 
LLM_TEMPERATURE = 0.4  # low — you want consistent, factual phrasing, not creative variation
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 25))  # margin above the ~5-7s seen on a Mac; Pi untested
# ============================================================
# Data paths
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")
LABEL_PATH = os.path.join(DATA_DIR, "labels")

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "strideiq.log") 