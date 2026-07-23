-- StrideIQ Database Schema
-- SQLite

PRAGMA foreign_keys = ON;

-- ============================================================
-- RUNS: parent table, one row per running session
-- ============================================================
CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    duration_sec INTEGER,
    run_type TEXT CHECK(run_type IN ('natural', 'induced_fault')) NOT NULL,
    notes TEXT,
    is_filmed BOOLEAN DEFAULT 0,
    video_filepath TEXT,
    sync_marker_timestamp REAL
);

-- ============================================================
-- NODES: lookup table for the 3 wearable devices
-- ============================================================
CREATE TABLE nodes (
    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_name TEXT UNIQUE NOT NULL,       -- 'waist', 'foot_left', 'foot_right'
    device_type TEXT NOT NULL,            -- 'ESP32', 'microbit'
    ble_identifier TEXT NOT NULL,
    has_gyro BOOLEAN NOT NULL,
    has_magnetometer BOOLEAN NOT NULL
);

-- Seed the 3 known nodes
INSERT INTO nodes (node_name, device_type, ble_identifier, has_gyro, has_magnetometer) VALUES
    ('waist', 'ESP32', 'StrideIQ-Waist', 1, 1),
    ('foot_left', 'microbit', 'StrideIQ-FootL', 0, 0),
    ('foot_right', 'microbit', 'StrideIQ-FootR', 0, 0);

-- ============================================================
-- RAW_SENSOR_DATA: every packet received, highest-volume table
-- ============================================================
CREATE TABLE raw_sensor_data (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,              -- timestamped on arrival at the Pi
    packet_seq INTEGER,                   -- device-side counter, for packet-loss detection
    ax REAL, ay REAL, az REAL,            -- accelerometer, all nodes
    gx REAL, gy REAL, gz REAL,            -- gyro, NULL for feet
    mx REAL, my REAL, mz REAL,            -- magnetometer, NULL for feet
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);

CREATE INDEX idx_raw_run_node_time ON raw_sensor_data (run_id, node_id, timestamp);

-- ============================================================
-- CONNECTION_LOG: BLE connection events, for reliability evaluation
-- ============================================================
CREATE TABLE connection_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    event_type TEXT CHECK(event_type IN ('connected', 'disconnected', 'reconnect_attempt')) NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);

-- ============================================================
-- FEATURES: computed per sliding window
-- ============================================================
CREATE TABLE features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,

    cadence REAL,
    vertical_oscillation REAL,
    running_smoothness REAL,
    pelvic_tilt REAL,                     -- NULL until waist node integrated; mean value
    pelvic_tilt_range REAL,               -- NULL until waist node integrated; variability within window
    pelvic_rotation REAL,                 -- NULL until waist node integrated; exploratory/low-confidence, mean
    pelvic_rotation_range REAL,           -- NULL until waist node integrated; variability within window

    sparc REAL,                           -- waist-derived smoothness (SPARC), NULL until waist integrated
    rms_jerk REAL,                        -- waist-derived, NULL until waist integrated
    harmonic_ratio REAL,                  -- waist-derived, NULL until waist integrated
    step_regularity REAL,                 -- waist-derived, NULL until waist integrated
    trunk_stability_index REAL,           -- waist-derived, NULL until waist integrated
    cadence_variability REAL,             -- waist-derived, NULL until waist integrated

    impact_left REAL,
    impact_right REAL,
    footstrike_timing_left REAL,
    footstrike_timing_right REAL,
    symmetry_score REAL,

    mean_accel REAL,
    variance_accel REAL,
    rms REAL,
    sma REAL,
    dominant_freq REAL,

    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_features_run_window ON features (run_id, window_start);

-- ============================================================
-- LABELS: ground truth, from MediaPipe or induced-fault sessions
-- Never conflate with predictions.
-- ============================================================
CREATE TABLE labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    label TEXT NOT NULL,                  -- 'Normal', 'High Impact', 'Asymmetry', 'Excessive Pelvic Tilt'
    label_source TEXT CHECK(label_source IN ('mediapipe', 'induced_manual')) NOT NULL,
    confidence REAL,
    is_primary BOOLEAN DEFAULT 1,         -- set to 0 for co-occurring secondary labels if using multi-label
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_labels_run_window ON labels (run_id, window_start);

-- ============================================================
-- PREDICTIONS: classifier output, tracked by model version
-- ============================================================
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL,
    model_version TEXT NOT NULL,          -- critical: you'll retrain 4-5 times
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_predictions_run_window ON predictions (run_id, window_start);

-- ============================================================
-- CLUSTER_RESULTS: unsupervised drift-detection output
-- ============================================================
CREATE TABLE cluster_results (
    cluster_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    window_start REAL NOT NULL,
    cluster_id INTEGER NOT NULL,
    is_drift_point BOOLEAN DEFAULT 0,     -- flags the window where dominant cluster shifts
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- ============================================================
-- RECOMMENDATIONS: structured recommendation + LLM explanation
-- ============================================================
CREATE TABLE recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    related_metric TEXT,
    recommendation_text TEXT,             -- structured, pre-LLM
    llm_explanation TEXT,                 -- LangChain-generated coaching sentence
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);