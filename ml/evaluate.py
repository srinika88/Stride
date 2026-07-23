"""
StrideIQ — Model Evaluation
Produces exactly what your proposal's Evaluation section promises:
accuracy, precision, recall, F1-score, confusion matrix, and comparison
against offline MediaPipe pose labels. Runs against a held-out test
set, not the training data itself (train.py's cross-validation already
covers in-training performance — this is a separate, honest check).
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from backend import config 
from backend.database_manager import DatabaseManager
from ml.predict import load_model, predict_window, ModelNotTrainedError
from ml.train import HOLDOUT_SPLIT_PATH

logger = logging.getLogger("strideiq.ml.evaluate")


def load_holdout_set(db: DatabaseManager, test_size: float = 0.2, random_state: int = 42, force_new: bool = False):
    """
    Returns the held-out test rows, creating and PERSISTING the split to
    data/holdout_split.json the first time this runs.

    Previously this picked a fresh random split every call with no
    record of which rows it picked — meaning train.py (which always
    trains on 100% of the data, with no way to know what evaluate.py
    considered "held out") had no way to actually respect it. The
    numbers evaluate.py reported were therefore not a true held-out
    evaluation; the "test" rows had almost certainly already been
    trained on.

    Now: the split is computed once and saved by (run_id, window_start)
    key. ml.train.load_training_data() reads this same file and
    excludes these exact rows by default. Re-running evaluate.py
    reuses the same split (so scores are comparable run-to-run) unless
    force_new=True — pass that only if you deliberately want a fresh
    split (e.g. dataset composition changed enough that the old split
    no longer makes sense).
    """
    rows = db.get_training_data()
    if not rows:
        raise ValueError("No labeled data available to evaluate against.")

    rows = [dict(r) for r in rows]

    if not force_new and HOLDOUT_SPLIT_PATH and Path(HOLDOUT_SPLIT_PATH).exists():
        with open(HOLDOUT_SPLIT_PATH) as f:
            saved = json.load(f)
        test_keys = {(k[0], k[1]) for k in saved["test_keys"]}
        test_rows = [r for r in rows if (r["run_id"], r["window_start"]) in test_keys]
        if test_rows:
            logger.info(f"Reusing existing holdout split ({len(test_rows)} rows) from {HOLDOUT_SPLIT_PATH}")
            return test_rows
        logger.warning("Saved holdout split matched no current rows (data may have changed) — creating a new one.")

    y = np.array([r["label"] for r in rows])

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 distinct labels present to evaluate meaningfully.")

    try:
        train_rows, test_rows = train_test_split(
            rows, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Stratification fails if any class has too few samples for a split —
        # fall back to a non-stratified split rather than crashing, but flag it
        logger.warning(
            "Stratified split failed (a class likely has too few samples) — "
            "falling back to a non-stratified split. Evaluation numbers for "
            "underrepresented classes will be unreliable until more data exists."
        )
        train_rows, test_rows = train_test_split(rows, test_size=test_size, random_state=random_state)

    test_keys = [[r["run_id"], r["window_start"]] for r in test_rows]
    Path(HOLDOUT_SPLIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDOUT_SPLIT_PATH, "w") as f:
        json.dump({"test_keys": test_keys, "test_size": test_size, "random_state": random_state}, f, indent=2)
    logger.info(
        f"Created new holdout split: {len(test_rows)} test rows saved to {HOLDOUT_SPLIT_PATH}. "
        f"Run (or re-run) train.py now — it will automatically exclude these rows."
    )

    return test_rows


def evaluate_model(db: DatabaseManager = None, model_version: str = None) -> dict:
    db = db or DatabaseManager()

    try:
        bundle = load_model(model_version)
    except ModelNotTrainedError as e:
        raise ValueError(str(e))

    test_rows = load_holdout_set(db)

    y_true = [r["label"] for r in test_rows]
    y_pred = []
    confidences = []

    for row in test_rows:
        label, confidence = predict_window(row, bundle)
        y_pred.append(label)
        confidences.append(confidence)

    labels_sorted = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    report = classification_report(y_true, y_pred, labels=labels_sorted, zero_division=0)

    results = {
        "model_version": bundle.get("model_name", "unknown"),
        "test_set_size": len(test_rows),
        "accuracy": float(accuracy),
        "mean_confidence": float(np.mean(confidences)),
        "per_class": {
            label: {
                "precision": float(p), "recall": float(r),
                "f1": float(f), "support": int(s),
            }
            for label, p, r, f, s in zip(labels_sorted, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
        "classification_report_text": report,
    }

    return results


def compare_against_mediapipe(db: DatabaseManager, run_id: int) -> dict:
    """
    Per the proposal's evaluation section: "comparison against offline
    MediaPipe pose labels." Compares the classifier's predictions
    against the MediaPipe-sourced ground truth labels for the SAME
    windows in a filmed run — measures how well the camera-free model
    agrees with what the camera actually saw.
    """
    labels = db.get_labels(run_id)
    mediapipe_labels = [dict(l) for l in labels if l["label_source"] == "mediapipe"]

    if not mediapipe_labels:
        logger.warning(f"Run {run_id} has no mediapipe-sourced labels to compare against")
        return {"run_id": run_id, "comparable_windows": 0}

    predictions = db.get_predictions(run_id)
    pred_by_window = {(p["window_start"], p["window_end"]): p["predicted_label"] for p in predictions}

    matched = 0
    total = 0
    for label_row in mediapipe_labels:
        key = (label_row["window_start"], label_row["window_end"])
        if key in pred_by_window:
            total += 1
            if pred_by_window[key] == label_row["label"]:
                matched += 1

    agreement = matched / total if total > 0 else None

    return {
        "run_id": run_id,
        "comparable_windows": total,
        "agreement_rate": agreement,
    }


def save_evaluation_report(results: dict, output_path: str = None):
    output_path = output_path or str(Path(config.MODEL_DIR) / "evaluation_report.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    db = DatabaseManager()
    try:
        results = evaluate_model(db)
    except ValueError as e:
        print(f"Cannot evaluate yet: {e}")
    else:
        print(f"Model: {results['model_version']}")
        print(f"Test set size: {results['test_set_size']}")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Mean confidence: {results['mean_confidence']:.3f}")
        print("\nPer-class metrics:")
        for label, metrics in results["per_class"].items():
            print(f"  {label}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                  f"F1={metrics['f1']:.2f} (n={metrics['support']})")
        print("\n" + results["classification_report_text"])
        print("\nConfusion matrix:")
        print(results["confusion_matrix"]["labels"])
        for row in results["confusion_matrix"]["matrix"]:
            print(row)

        save_evaluation_report(results) 