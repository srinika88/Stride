"""
StrideIQ — Prediction
Loads the model bundle saved by train.py (model + imputer + scaler +
feature names together — a model alone isn't usable without knowing
how its inputs were preprocessed) and produces real predictions from
feature rows, writing them via database_manager.insert_prediction().

This is the piece recommend.py's classify_window() was always meant to
be swapped for — same (label, confidence) output shape, so nothing
downstream needs to change.
"""

import json
import logging
from pathlib import Path

import numpy as np
import joblib

from backend import config 
from backend.database_manager import DatabaseManager
from backend.mqtt.mqtt_client import StrideMQTTClient

logger = logging.getLogger("strideiq.ml.predict")


class ModelNotTrainedError(Exception):
    pass


def load_model(version: str = None):
    """
    Loads a specific version if given, otherwise the "current" model
    saved at config.MODEL_PATH. Raises a clear, specific error rather
    than a confusing pickle/joblib stack trace if nothing's been trained
    yet — this WILL happen before you have real data, and the error
    should say that plainly.
    """
    path = Path(config.MODEL_DIR) / f"model_{version}.joblib" if version else Path(config.MODEL_PATH)

    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {path}. Run train.py first — "
            f"this needs real labeled data before there's anything to load."
        )

    bundle = joblib.load(path)
    logger.info(f"Loaded model: {bundle['model_name']} from {path}")
    return bundle


def predict_window(feature_row: dict, bundle: dict) -> tuple[str, float]:
    """
    feature_row: one row from the `features` table (as a dict).
    Returns (predicted_label, confidence) — same shape as
    recommend.py's classify_window(), so it's a drop-in replacement.
    """
    feature_names = bundle["feature_names"]
    x = np.array([[feature_row.get(col) for col in feature_names]], dtype=float)

    x_imputed = bundle["imputer"].transform(x)
    x_scaled = bundle["scaler"].transform(x_imputed)

    model = bundle["model"]
    label = model.predict(x_scaled)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_scaled)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 1.0  # model doesn't expose probabilities — treat as fully confident

    return str(label), confidence


# Which feature(s) to surface as "related_metric"/"details" per label, for
# the LLM coaching prompt and for a human skimming the feedback loop's log.
# Not a rigorous per-instance feature-attribution method (that would need
# something like SHAP) — just a reasonable, explainable pointer to the
# metric(s) most associated with each label, matching what
# ml.recommendation_engine's rule thresholds already look at.
_LABEL_RELATED_COLUMNS = {
    "High Impact": ("impact_left", "impact_right"),
    "Asymmetry": ("symmetry_score",),
    "Excessive Pelvic Tilt": ("pelvic_tilt_range",),
    "Normal": (),
}


def _describe_prediction(label: str, feature_row: dict) -> tuple[str, str]:
    cols = _LABEL_RELATED_COLUMNS.get(label, ())
    if not cols:
        return label, "No single metric stood out this window."

    details = ", ".join(f"{col}={feature_row.get(col)}" for col in cols if feature_row.get(col) is not None)
    return cols[0], details or "Relevant metric(s) missing for this window."


def predict_run(
    run_id: int,
    db: DatabaseManager,
    model_version: str = None,
    mqtt_client: StrideMQTTClient = None,
) -> list[dict]:
    """
    Runs prediction over every feature window in a run, writes results
    to the predictions table (tagged with the real model version, per
    the retrain-tracking design), and returns the results for anything
    that wants to act on them immediately (e.g. the feedback loop).
    """
    bundle = load_model(model_version)
    features = db.get_features(run_id)

    if not features:
        logger.warning(f"No feature rows for run {run_id} — nothing to predict on")
        return []

    # predict.py previously only wrote to the DB — nothing ever published to
    # the MQTT "predictions" topic that backend/mqtt/feedback_loop.py
    # actually listens on, so the real-time coaching loop never ran against
    # real classifier output. Connect a client here (or accept one passed
    # in, e.g. from a longer-running process that already has one open)
    # so each prediction reaches the feedback loop as it's produced.
    owns_client = mqtt_client is None
    if owns_client:
        mqtt_client = StrideMQTTClient()
        try:
            mqtt_client.connect()
            mqtt_client.client.loop_start()
        except Exception as e:
            logger.warning(f"Could not connect to MQTT broker ({e}) — predictions will still be written to the DB, but the live feedback loop won't see them.")
            mqtt_client = None

    results = []
    for row in features:
        row = dict(row)
        label, confidence = predict_window(row, bundle)

        db.insert_prediction(
            run_id=run_id,
            window_start=row["window_start"],
            window_end=row["window_end"],
            predicted_label=label,
            confidence=confidence,
            model_version=bundle.get("model_name", config.MODEL_VERSION),
        )

        if mqtt_client is not None:
            related_metric, details = _describe_prediction(label, row)
            payload = {
                "run_id": run_id,
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "predicted_label": label,
                "confidence": confidence,
                "related_metric": related_metric,
                "details": details,
            }
            try:
                mqtt_client.publish_prediction(json.dumps(payload))
            except Exception as e:
                logger.warning(f"Failed to publish prediction over MQTT (DB write still succeeded): {e}")

        results.append({
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "predicted_label": label,
            "confidence": confidence,
        })

    if owns_client and mqtt_client is not None:
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()

    logger.info(f"Run {run_id}: {len(results)} predictions written")
    return results


if __name__ == "__main__":
    db = DatabaseManager()
    runs = db.get_all_runs()
    if not runs:
        print("No runs found.")
    else:
        latest_run_id = runs[0]["run_id"]
        try:
            results = predict_run(latest_run_id, db)
            for r in results:
                print(r)
        except ModelNotTrainedError as e:
            print(f"Cannot predict yet: {e}") 