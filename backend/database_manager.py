"""
StrideIQ Database Manager
Single point of contact for all SQLite reads/writes.
Nothing else in the codebase should open a raw sqlite3 connection directly.
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from backend.database import initialize_database 
from backend import config

logger = logging.getLogger("strideiq.database")


class DatabaseManager:
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self._ensure_schema() 

    # ------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------
    @contextmanager
    def _connect(self):
        """
        One connection per call, closed automatically, foreign keys enforced.
        SQLite connections aren't safe to share across threads/greenlets, so
        we open/close per-operation rather than holding a long-lived connection.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Database operation failed, rolled back")
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
     """
    Creates the database and tables if they don't already exist.
    Safe to call every time because initialize_database() uses
    CREATE TABLE IF NOT EXISTS.
    """
     logger.info("Ensuring database schema exists...")
     initialize_database(self.db_path)

    # ------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------
    def start_run(self, date, start_time, run_type, notes=None, is_filmed=False) -> int:
        if run_type not in ("natural", "induced_fault"):
            raise ValueError(f"Invalid run_type: {run_type}")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs (date, start_time, run_type, notes, is_filmed)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, start_time, run_type, notes, is_filmed),
            )
            run_id = cur.lastrowid
            logger.info(f"Started run {run_id} ({run_type})")
            return run_id

    def stop_run(self, run_id: int, end_time, duration_sec: int):
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET end_time = ?, duration_sec = ? WHERE run_id = ?""",
                (end_time, duration_sec, run_id),
            )
        logger.info(f"Stopped run {run_id}, duration {duration_sec}s")

    def set_sync_marker(self, run_id: int, video_filepath: str, sync_timestamp: float):
        """Call once per filmed session, after the sync-clap moment is identified."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs
                   SET video_filepath = ?, sync_marker_timestamp = ?, is_filmed = 1
                   WHERE run_id = ?""",
                (video_filepath, sync_timestamp, run_id),
            )

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def get_all_runs(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM runs ORDER BY date DESC, start_time DESC").fetchall()

    # ------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------
    def get_node_id(self, node_name: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT node_id FROM nodes WHERE node_name = ?", (node_name,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown node_name: {node_name}")
            return row["node_id"]

    # ------------------------------------------------------------
    # Raw sensor data — highest-volume writes, kept as a batch insert
    # ------------------------------------------------------------
    def insert_raw_readings(self, readings: list[dict]):
        """
        readings: list of dicts with keys matching raw_sensor_data columns.
        Batched in one transaction — don't call this once per packet, buffer
        and flush periodically (e.g. every 0.5s or every N packets) or you'll
        bottleneck on 3 concurrent BLE streams writing individually.
        """
        if not readings:
            return
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO raw_sensor_data
                   (run_id, node_id, timestamp, packet_seq, ax, ay, az, gx, gy, gz, mx, my, mz)
                   VALUES (:run_id, :node_id, :timestamp, :packet_seq,
                           :ax, :ay, :az, :gx, :gy, :gz, :mx, :my, :mz)""",
                readings,
            )

    def get_raw_readings(self, run_id: int, node_name: str = None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if node_name:
                node_id = self.get_node_id(node_name)
                return conn.execute(
                    """SELECT * FROM raw_sensor_data
                       WHERE run_id = ? AND node_id = ? ORDER BY timestamp""",
                    (run_id, node_id),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM raw_sensor_data WHERE run_id = ? ORDER BY timestamp", (run_id,)
            ).fetchall()

    # ------------------------------------------------------------
    # Connection log — BLE reliability tracking
    # ------------------------------------------------------------
    def log_connection_event(self, run_id: int, node_name: str, event_type: str, timestamp: float):
        if event_type not in ("connected", "disconnected", "reconnect_attempt"):
            raise ValueError(f"Invalid event_type: {event_type}")
        node_id = self.get_node_id(node_name)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO connection_log (run_id, node_id, event_type, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (run_id, node_id, event_type, timestamp),
            )
        if event_type == "disconnected":
            logger.warning(f"Node {node_name} disconnected during run {run_id} at {timestamp}")

    def get_packet_loss_rate(self, run_id: int, node_name: str) -> float | None:
        """
        Uses packet_seq gaps to estimate loss rate — expects seq to increment
        by 1 per packet on the device side.
        """
        with self._connect() as conn:
            node_id = self.get_node_id(node_name)
            rows = conn.execute(
                """SELECT packet_seq FROM raw_sensor_data
                   WHERE run_id = ? AND node_id = ? ORDER BY packet_seq""",
                (run_id, node_id),
            ).fetchall()
        if len(rows) < 2:
            return None
        seqs = [r["packet_seq"] for r in rows if r["packet_seq"] is not None]
        if len(seqs) < 2:
            return None
        expected = seqs[-1] - seqs[0] + 1
        received = len(seqs)
        return round(1 - (received / expected), 4)

    # ------------------------------------------------------------
    # Features 
    # ------------------------------------------------------------
    def insert_features(self, feature_rows: list[dict]):
        if not feature_rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO features
                   (run_id, window_start, window_end, cadence, vertical_oscillation,
                    running_smoothness, pelvic_tilt, pelvic_tilt_range,
                    pelvic_rotation, pelvic_rotation_range,
                    impact_left, impact_right, footstrike_timing_left, footstrike_timing_right,
                    symmetry_score, mean_accel, variance_accel, rms, sma, dominant_freq,
                    sparc, rms_jerk, harmonic_ratio, step_regularity,
                    trunk_stability_index, cadence_variability)
                   VALUES (:run_id, :window_start, :window_end, :cadence, :vertical_oscillation,
                           :running_smoothness, :pelvic_tilt, :pelvic_tilt_range,
                           :pelvic_rotation, :pelvic_rotation_range,
                           :impact_left, :impact_right, :footstrike_timing_left, :footstrike_timing_right,
                           :symmetry_score, :mean_accel, :variance_accel, :rms, :sma, :dominant_freq,
                           :sparc, :rms_jerk, :harmonic_ratio, :step_regularity,
                           :trunk_stability_index, :cadence_variability)""",
                feature_rows,
            )

    def get_features(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM features WHERE run_id = ? ORDER BY window_start", (run_id,)
            ).fetchall()

    # ------------------------------------------------------------
    # Labels — ground truth, never mixed with predictions
    # ------------------------------------------------------------
    def insert_label(self, run_id, window_start, window_end, label, label_source, confidence=None, is_primary=True):
        if label not in config.FORM_LABELS:
            raise ValueError(f"Unknown label: {label}")
        if label_source not in config.LABEL_SOURCES:
            raise ValueError(f"Unknown label_source: {label_source}")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO labels
                   (run_id, window_start, window_end, label, label_source, confidence, is_primary)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, window_start, window_end, label, label_source, confidence, is_primary),
            )

    def get_labels(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM labels WHERE run_id = ? ORDER BY window_start", (run_id,)
            ).fetchall()

    def get_training_data(self) -> list[sqlite3.Row]:
        """Joins features to labels — the actual training set for model.fit()."""
        with self._connect() as conn:
            return conn.execute(
                """SELECT f.*, l.label
                   FROM features f
                   JOIN labels l
                     ON f.run_id = l.run_id
                    AND f.window_start = l.window_start
                   WHERE l.is_primary = 1"""
            ).fetchall()

    # ------------------------------------------------------------
    # Predictions — always tagged with model_version
    # ------------------------------------------------------------
    def insert_prediction(self, run_id, window_start, window_end, predicted_label, confidence, model_version =None):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (run_id, window_start, window_end, predicted_label, confidence, model_version)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, window_start, window_end, predicted_label, confidence, model_version or config.MODEL_VERSION),
            )

    def get_predictions(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM predictions WHERE run_id = ? ORDER BY window_start", (run_id,)
            ).fetchall()

    # ------------------------------------------------------------
    # Cluster results — drift-point detection
    # ------------------------------------------------------------
    def insert_cluster_results(self, cluster_rows: list[dict]):
        if not cluster_rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO cluster_results (run_id, window_start, cluster_id, is_drift_point)
                   VALUES (:run_id, :window_start, :cluster_id, :is_drift_point)""",
                cluster_rows,
            )

    def get_drift_points(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """SELECT * FROM cluster_results
                   WHERE run_id = ? AND is_drift_point = 1
                   ORDER BY window_start""",
                (run_id,),
            ).fetchall()

    # ------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------
    def insert_recommendation(self, run_id, timestamp, related_metric, recommendation_text, llm_explanation=None):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO recommendations
                   (run_id, timestamp, related_metric, recommendation_text, llm_explanation)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, timestamp, related_metric, recommendation_text, llm_explanation),
            )

    def get_recommendations(self, run_id: int) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM recommendations WHERE run_id = ? ORDER BY timestamp", (run_id,)
            ).fetchall()
    def get_latest_run(self) -> sqlite3.Row | None:
        """Returns the most recently created run."""
        with self._connect() as conn:
         return conn.execute(
            "SELECT * FROM runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    def delete_run(self, run_id: int):
        """
        Deletes a run and all associated records.
        """
        with self._connect() as conn:

         conn.execute(
            "DELETE FROM recommendations WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM cluster_results WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM predictions WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM labels WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM features WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM connection_log WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM raw_sensor_data WHERE run_id = ?",
            (run_id,)
        )

         conn.execute(
            "DELETE FROM runs WHERE run_id = ?",
            (run_id,)
        )

        logger.info(f"Deleted run {run_id}")
    def clear_table(self, table_name: str):
      """
      Deletes all rows from a table.

      Intended for development/testing only.
      """

      allowed_tables = {
        "runs",
        "nodes",
        "raw_sensor_data",
        "connection_log",
        "features",
        "labels",
        "predictions",
        "cluster_results",
        "recommendations"
    }

      if table_name not in allowed_tables:
        raise ValueError(f"Invalid table name: {table_name}")

      with self._connect() as conn:
        conn.execute(f"DELETE FROM {table_name}")

      logger.warning(f"Cleared table '{table_name}'")

    def database_exists(self) -> bool:
     """
     Returns True if the SQLite database file exists.
     """
     return Path(self.db_path).exists()
    
    def get_run_statistics(self, run_id: int) -> dict:
      """
      Returns useful statistics for one run.
      """

      with self._connect() as conn:

        raw_packets = conn.execute(
            """
            SELECT COUNT(*)
            FROM raw_sensor_data
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

        feature_windows = conn.execute(
            """
            SELECT COUNT(*)
            FROM features
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

        labels = conn.execute(
            """
            SELECT COUNT(*)
            FROM labels
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

        predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

        recommendations = conn.execute(
            """
            SELECT COUNT(*)
            FROM recommendations
            WHERE run_id = ?
            """,
            (run_id,)
        ).fetchone()[0]

      return {
        "run_id": run_id,
        "raw_packets": raw_packets,
        "feature_windows": feature_windows,
        "labels": labels,
        "predictions": predictions,
        "recommendations": recommendations,
    } 