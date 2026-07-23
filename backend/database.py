# ==========================================================
# Database Connection
# ==========================================================

import os
import sqlite3
from backend import config


def get_connection(db_path: str = None):
    db_path = db_path or config.DATABASE_PATH
    # Create the data folder if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn 


# ==========================================================
# Initialize Database
# ==========================================================

def initialize_database(db_path: str = None):

    conn = get_connection(db_path) 
    cursor = conn.cursor()

    # ==========================================================
    # RUNS
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (

        run_id INTEGER PRIMARY KEY AUTOINCREMENT,

        date TEXT NOT NULL,

        start_time TEXT,

        end_time TEXT,

        duration_sec INTEGER,

        run_type TEXT
            CHECK(run_type IN ('natural','induced_fault'))
            NOT NULL,

        notes TEXT,

        is_filmed INTEGER DEFAULT 0
            CHECK(is_filmed IN (0,1)),

        video_filepath TEXT,

        sync_marker_timestamp REAL
    );
    """)

    # ==========================================================
    # NODES
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nodes (

        node_id INTEGER PRIMARY KEY AUTOINCREMENT,

        node_name TEXT UNIQUE NOT NULL,

        device_type TEXT NOT NULL,

        ble_identifier TEXT UNIQUE NOT NULL,

        has_gyro INTEGER
            CHECK(has_gyro IN (0,1))
            NOT NULL,

        has_magnetometer INTEGER
            CHECK(has_magnetometer IN (0,1))
            NOT NULL,

        sampling_rate_hz INTEGER NOT NULL
    );
    """)

    # ==========================================================
    # MODELS
    # ==========================================================

    # ==========================================================
    # Seed Wearable Nodes
    # ==========================================================

    cursor.execute("""
    INSERT OR IGNORE INTO nodes
    (
        node_name,
        device_type,
        ble_identifier,
        has_gyro,
        has_magnetometer,
        sampling_rate_hz
    )

    VALUES

    (
        'waist',
        'ESP32',
        'StrideIQ-Waist',
        1,
        1,
        100
    ),

    (
        'foot_left',
        'microbit',
        'StrideIQ-FootL',
        0,
        0,
        50
    ),

    (
        'foot_right',
        'microbit',
        'StrideIQ-FootR',
        0,
        0,
        50
    ),

    (
        'chest',
        'microbit',
        'StrideIQ-Chest',
        0,
        0,
        50
    );
    """)
        # ==========================================================
    # RAW SENSOR DATA
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_sensor_data (

        reading_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        node_id INTEGER NOT NULL,

        timestamp REAL NOT NULL,

        packet_seq INTEGER,

        ax REAL,
        ay REAL,
        az REAL,

        gx REAL,
        gy REAL,
        gz REAL,

        mx REAL,
        my REAL,
        mz REAL,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id),

        FOREIGN KEY (node_id)
            REFERENCES nodes(node_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_raw_run_node_time
    ON raw_sensor_data(run_id, node_id, timestamp);
    """)

    # ==========================================================
    # CONNECTION LOG
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS connection_log (

        log_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        node_id INTEGER NOT NULL,

        event_type TEXT
            CHECK(event_type IN
            (
                'connected',
                'disconnected',
                'reconnect_attempt'
            ))
            NOT NULL,

        timestamp REAL NOT NULL,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id),

        FOREIGN KEY (node_id)
            REFERENCES nodes(node_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_connection_log
    ON connection_log(run_id, node_id, timestamp);
    """)

    # ==========================================================
    # FEATURES
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features (

        feature_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        window_start REAL NOT NULL,

        window_end REAL NOT NULL,

        cadence REAL,

        cadence_variability REAL,

        vertical_oscillation REAL,

        running_smoothness REAL,

        pelvic_tilt REAL,

        pelvic_tilt_range REAL,

        pelvic_rotation REAL,

        pelvic_rotation_range REAL,

        sparc REAL,

        rms_jerk REAL,

        harmonic_ratio REAL,

        step_regularity REAL,

        trunk_stability_index REAL,

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

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_features_run_window
    ON features(run_id, window_start);
    """)
        # ==========================================================
    # LABELS (Ground Truth)
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS labels (

        label_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        window_start REAL NOT NULL,

        window_end REAL NOT NULL,

        label TEXT NOT NULL,

        label_source TEXT
            CHECK(label_source IN
            (
                'mediapipe',
                'induced_manual',
                'heuristic'
            ))
            NOT NULL,

        confidence REAL,

        is_primary INTEGER DEFAULT 1
            CHECK(is_primary IN (0,1)),

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_labels_run_window
    ON labels(run_id, window_start);
    """)


    # ==========================================================
    # PREDICTIONS
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        window_start REAL NOT NULL,

        window_end REAL NOT NULL,

        predicted_label TEXT NOT NULL,

        confidence REAL,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_run_window
    ON predictions(run_id, window_start);
    """)


    # ==========================================================
    # CLUSTER RESULTS
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cluster_results (

        cluster_result_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        window_start REAL NOT NULL,

        cluster_id INTEGER NOT NULL,

        is_drift_point INTEGER DEFAULT 0
            CHECK(is_drift_point IN (0,1)),

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
    );
    """)


    # ==========================================================
    # RECOMMENDATIONS
    # ==========================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (

        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,

        run_id INTEGER NOT NULL,

        timestamp REAL NOT NULL,

        related_metric TEXT,

        recommendation_text TEXT,

        llm_explanation TEXT,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
    );
    """)


    # ==========================================================
    # SAVE DATABASE
    # ==========================================================

    conn.commit()
    conn.close()

    print("StrideIQ database initialized successfully!")