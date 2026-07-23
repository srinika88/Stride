"""
StrideIQ — Dataset Ingestion
The missing link between dataset_prep.py and train.py:

  raw dataset CSV(s)
      -> DatasetPreprocessor.run_pipeline()      (dataset_prep.py, already built)
      -> DatabaseManager.insert_raw_readings()    (writes into raw_sensor_data)
      -> features_pipeline.process_run_windows()  (writes into features -- foot
                                                     AND waist, whichever exist)
      -> heuristic_labeler.label_run()            (writes into labels)

After this runs, train.py's db.get_training_data() has real rows to
pull from.

(Relocated from backend/glue.py to match this module's own usage string,
which always said `python -m ml.preprocessing.ingest_dataset` -- the file
just hadn't caught up to its own docstring. Also swapped the old
foot-only processing.features_foot.process_run_windows import for
ml.features_pipeline.process_run_windows, which is the real orchestrator:
it merges foot + waist features into one row per window and degrades
gracefully when only one side has data -- exactly the waist-only NoNaN
case below.)

Usage:
    # Foot dataset (Camargo, gaitmotion) -- paired left/right IMU streams
    python -m ml.preprocessing.ingest_dataset \\
        --dataset camargo \\
        --left path/to/left_foot.csv \\
        --right path/to/right_foot.csv \\
        --notes "Camargo EPIC Lab public dataset"

    # single-stream foot dataset (no paired left/right) -- labeled onto
    # foot_left only; symmetry-dependent labels won't apply since
    # compute_symmetry_score needs both feet present in the same window
    python -m ml.preprocessing.ingest_dataset \\
        --dataset gaitmotion \\
        --left path/to/single_stream.csv

    # NONAN -- pelvis-only mocap data, StrideIQ's waist-node analog.
    # No foot columns are mapped for NONAN, so this always goes through
    # the waist-only path regardless of --left/--right.
    python -m ml.preprocessing.ingest_dataset \\
        --dataset nonan \\
        --waist path/to/nonan_session.csv \\
        --notes "NONAN pelvis data, pre-ESP32 supplement"
"""

import argparse
import logging
import time
from datetime import date

import pandas as pd

from backend import config
from backend.database_manager import DatabaseManager
from ml.features_pipeline import process_run_windows
from ml.preprocessing.dataset_prep import DatasetPreprocessor
from ml.preprocessing.heuristic import label_run

logger = logging.getLogger("strideiq.ml.ingest_dataset")
logging.basicConfig(level=config.LOG_LEVEL)


def _to_readings(df: pd.DataFrame, run_id: int, node_id: int) -> list[dict]:
    """
    Maps the dataset_prep.py output schema (ax, ay, az, gx, gy, gz, mx,
    my, mz, timestamp, packet_seq) onto raw_sensor_data's insert shape.
    Note: dataset_prep.py's normalize() step z-scores/min-maxes the
    sensor channels by default -- fine for relative features (symmetry,
    smoothness) but distorts absolute-magnitude features like impact
    intensity's real g-force meaning. Ingestion here intentionally
    skips normalize() (see the manual pipeline calls below) so
    footstrike/impact thresholds in config.py stay physically meaningful.
    """
    records = df.to_dict("records")
    readings = []
    for r in records:
        readings.append({
            "run_id": run_id,
            "node_id": node_id,
            "timestamp": float(r["timestamp"]),
            "packet_seq": int(r["packet_seq"]),
            "ax": float(r["ax"]), "ay": float(r["ay"]), "az": float(r["az"]),
            "gx": float(r["gx"]), "gy": float(r["gy"]), "gz": float(r["gz"]),
            "mx": r["mx"] if pd.notna(r["mx"]) else None,
            "my": r["my"] if pd.notna(r["my"]) else None,
            "mz": r["mz"] if pd.notna(r["mz"]) else None,
        })
    return readings


def _run_prep(dataset: str, path: str, assumed_node: str) -> pd.DataFrame:
    prep = DatasetPreprocessor(dataset=dataset, input_path=path)
    prep.load()
    prep.reshape(assumed_node=assumed_node)
    prep.detect_sampling_rate()
    prep.normalize_timestamps()
    # NONAN's accel columns are in mG (confirmed in its own CSV header,
    # e.g. "Pelvis Accel Sensor X (mG)") -- everything else here is in g.
    accel_unit = "mg" if dataset == "nonan" else "g"
    prep.convert_units(accel_unit=accel_unit)
    prep.remove_duplicates()
    prep.sort()
    prep.interpolate_missing()
    # deliberately skip .normalize() -- see _to_readings docstring
    prep.validate()
    return prep.processed


def ingest(dataset: str, left_path: str = None, right_path: str = None,
           waist_path: str = None, notes: str = None) -> int:
    db = DatabaseManager()

    run_id = db.start_run(
        date=date.today().isoformat(),
        start_time=time.strftime("%H:%M:%S"),
        run_type="natural",
        notes=notes or f"Ingested public dataset: {dataset}",
        is_filmed=False,
    )
    logger.info(f"Created run {run_id} for dataset '{dataset}'")

    # ---- Waist-only path (NONAN pelvis data, or any dataset where you
    # only pass --waist) ----
    if waist_path:
        waist_df = _run_prep(dataset, waist_path, assumed_node="waist")
        node_id_waist = db.get_node_id("waist")
        db.insert_raw_readings(_to_readings(waist_df, run_id, node_id_waist))
        logger.info(f"Inserted {len(waist_df)} raw readings for waist")

    # ---- Foot path (Camargo, gaitmotion, or any dataset with foot IMU
    # columns) ----
    if left_path:
        left_df = _run_prep(dataset, left_path, assumed_node="foot_left")
        node_id_left = db.get_node_id("foot_left")
        db.insert_raw_readings(_to_readings(left_df, run_id, node_id_left))
        logger.info(f"Inserted {len(left_df)} raw readings for foot_left")

        if right_path:
            right_df = _run_prep(dataset, right_path, assumed_node="foot_right")
            node_id_right = db.get_node_id("foot_right")
            db.insert_raw_readings(_to_readings(right_df, run_id, node_id_right))
            logger.info(f"Inserted {len(right_df)} raw readings for foot_right")
        else:
            logger.warning(
                "No --right path given. Mirroring the left stream onto "
                "foot_right as a placeholder -- impact/timing features on "
                "the mirrored side won't be real and symmetry_score will "
                "always read as ~0 (perfectly symmetric) for this run. "
                "Use paired left/right data for anything symmetry-related."
            )
            node_id_right = db.get_node_id("foot_right")
            db.insert_raw_readings(_to_readings(left_df, run_id, node_id_right))

    if not left_path and not waist_path:
        raise ValueError("Must supply --left (foot data) and/or --waist (waist/pelvis data)")

    # features_pipeline.process_run_windows merges foot + waist features
    # per window and handles either side being absent gracefully -- no
    # separate waist-only code path needed here.
    n_windows = process_run_windows(run_id, db)
    logger.info(f"Computed features for {n_windows} windows")

    label_summary = label_run(run_id, db)
    logger.info(f"Labeling complete for run {run_id}: {label_summary}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a public IMU dataset into StrideIQ's pipeline")
    parser.add_argument("--dataset", required=True, choices=DatasetPreprocessor.SUPPORTED_DATASETS)
    parser.add_argument("--left", default=None, help="Path to left-foot (or single-stream) CSV")
    parser.add_argument("--right", default=None, help="Path to right-foot CSV, if available")
    parser.add_argument("--waist", default=None, help="Path to waist/pelvis CSV (e.g. NONAN)")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    run_id = ingest(args.dataset, args.left, args.right, args.waist, args.notes)
    print(f"\nIngested as run_id={run_id}. Run 'python -m ml.train' next to train on it.") 