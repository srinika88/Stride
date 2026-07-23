"""
StrideIQ -- Raw Data Logger

Mirrors raw sensor readings to per-run, per-node CSV files under
config.RAW_DATA_PATH as they arrive. DatabaseManager.insert_raw_readings()
writes to SQLite, which is fine for the live pipeline but awkward for
building a training dataset (no easy per-run CSV for pandas / teammates /
re-running feature extraction without touching the DB).

Pure add-on -- call alongside DatabaseManager.insert_raw_readings, doesn't
replace it and doesn't require it (useful during hardware bring-up when
you just want to sanity check a node's output before a DB/run exists).
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone

from backend import config

logger = logging.getLogger("strideiq.data_logger")


class DataLogger:
    """Appends raw sensor reading dicts to per-run, per-node CSV files."""

    def __init__(self, raw_data_path: str = config.RAW_DATA_PATH):
        self.raw_data_path = raw_data_path
        os.makedirs(self.raw_data_path, exist_ok=True)
        self._open_files: dict[tuple[int, str], tuple] = {}

    def _path_for(self, run_id: int, node_name: str) -> str:
        return os.path.join(self.raw_data_path, f"run{run_id}_{node_name}.csv")

    def _writer_for(self, run_id: int, node_name: str, fieldnames: list[str]):
        key = (run_id, node_name)
        if key not in self._open_files:
            path = self._path_for(run_id, node_name)
            is_new = not os.path.exists(path)
            fh = open(path, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            self._open_files[key] = (fh, writer)
            logger.info(f"Logging raw readings for run {run_id} / {node_name} -> {path}")
        return self._open_files[key][1]

    def log_readings(self, run_id: int, node_name: str, readings: list[dict]):
        """
        readings: list of dicts (one per BLE/MQTT packet) with whatever
        fields that node publishes (timestamp, accel/gyro axes, etc).
        Column set is inferred from the first reading, assumed stable
        for the run.
        """
        if not readings:
            return
        fieldnames = list(readings[0].keys())
        if "logged_at" not in fieldnames:
            fieldnames = fieldnames + ["logged_at"]
        writer = self._writer_for(run_id, node_name, fieldnames)
        now = datetime.now(timezone.utc).isoformat()
        for r in readings:
            row = dict(r)
            row.setdefault("logged_at", now)
            writer.writerow(row)
        self._open_files[(run_id, node_name)][0].flush()

    def close(self, run_id: int | None = None):
        """Close file handle(s). Pass run_id to close just that run's files, or
        omit to close everything (e.g. on shutdown)."""
        keys = [k for k in self._open_files if run_id is None or k[0] == run_id]
        for key in keys:
            fh, _ = self._open_files.pop(key)
            fh.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass 