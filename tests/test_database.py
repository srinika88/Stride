"""
Tests for backend.database_manager.DatabaseManager against a throwaway
SQLite file -- no hardware, no MQTT broker needed. Run with: pytest tests/test_database.py
"""
import os
import sys
import tempfile

import pytest 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database_manager import DatabaseManager


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # DatabaseManager creates schema fresh
    manager = DatabaseManager(db_path=path)
    yield manager
    if os.path.exists(path):
        os.remove(path)


def test_start_and_get_run(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    run = db.get_run(run_id)
    assert run is not None
    assert run["run_type"] == "natural"


def test_stop_run_sets_duration(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    db.stop_run(run_id, end_time="08:05:00", duration_sec=300)
    run = db.get_run(run_id)
    assert run["duration_sec"] == 300


def test_get_all_runs_returns_list(db):
    db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    db.start_run(date="2026-07-22", start_time="09:00:00", run_type="natural")
    runs = db.get_all_runs()
    assert len(runs) == 2


def test_insert_and_get_raw_readings(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    node_id = db.get_node_id("foot_left")
    base = {"run_id": run_id, "node_id": node_id, "packet_seq": None,
            "ax": 0.0, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0,
            "mx": None, "my": None, "mz": None}
    readings = [
        {**base, "timestamp": 0.0, "packet_seq": 1},
        {**base, "timestamp": 0.02, "packet_seq": 2, "ax": 0.15, "az": 9.7},
    ]
    db.insert_raw_readings(readings)
    stored = db.get_raw_readings(run_id, node_name="foot_left")
    assert len(stored) == 2


def test_insert_and_get_features(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    full_row = {
        "run_id": run_id, "window_start": 0.0, "window_end": 1.0,
        "cadence": 170.0, "vertical_oscillation": None, "running_smoothness": None,
        "pelvic_tilt": None, "pelvic_tilt_range": None,
        "pelvic_rotation": None, "pelvic_rotation_range": None,
        "impact_left": None, "impact_right": None,
        "footstrike_timing_left": None, "footstrike_timing_right": None,
        "symmetry_score": 0.05, "mean_accel": None, "variance_accel": None,
        "rms": None, "sma": None, "dominant_freq": None,
        "sparc": None, "rms_jerk": None, "harmonic_ratio": None,
        "step_regularity": None, "trunk_stability_index": None,
        "cadence_variability": None,
    }
    db.insert_features([full_row])
    features = db.get_features(run_id)
    assert len(features) == 1
    assert features[0]["cadence"] == 170.0


def test_insert_and_get_label(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    db.insert_label(run_id, window_start=0.0, window_end=1.0,
                     label="Normal", label_source="heuristic", confidence=0.8)
    labels = db.get_labels(run_id)
    assert len(labels) == 1
    assert labels[0]["label"] == "Normal"


def test_get_run_statistics_on_empty_run(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    stats = db.get_run_statistics(run_id)
    assert stats is not None


def test_delete_run_removes_it(db):
    run_id = db.start_run(date="2026-07-22", start_time="08:00:00", run_type="natural")
    db.delete_run(run_id)
    assert db.get_run(run_id) is None


def test_database_exists_flag(db):
    assert db.database_exists() is True