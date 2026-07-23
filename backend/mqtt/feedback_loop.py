"""
StrideIQ — MQTT Feedback Loop
Listens on the predictions topic, decides whether to trigger actuation,
and publishes to the feedback topic. Also wires the LLM coaching call
so a recommendation gets stored whenever a prediction comes through.

Testable right now by publishing fake prediction messages to your own
local Mosquitto broker — no hardware, no trained model needed. Once
predict.py exists and publishes real classifier output to the predictions
topic, this file doesn't change at all.
"""

import json
import logging
import time

from backend import config
from backend.database_manager import DatabaseManager
from llm.langchain_pipeline import generate_coaching
from backend.mqtt.mqtt_client import StrideMQTTClient
from ml.recommendation_engine import recommend as rule_based_recommend

logger = logging.getLogger("strideiq.feedback_loop")


class FeedbackLoop:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.mqtt = StrideMQTTClient(config.MQTT_BROKER, config.MQTT_PORT)

    def start(self):
        self.mqtt.connect()
        self.mqtt.client.subscribe(config.MQTT_TOPICS["predictions"])
        self.mqtt.client.on_message = self._on_prediction
        logger.info("Feedback loop listening on predictions topic")

    def _on_prediction(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(f"Bad prediction payload: {msg.payload}")
            return

        self._process_prediction(payload)

    def _process_prediction(self, payload: dict):
        """
        payload expected shape:
        {
          "run_id": int, "window_start": float, "window_end": float,
          "predicted_label": str, "confidence": float,
          "related_metric": str, "details": str
        }
        """
        run_id = payload["run_id"]
        label = payload["predicted_label"]
        confidence = payload["confidence"]

        # Decide actuation
        actuation_code = None
        if label in config.ACTUATION_LABELS and confidence >= config.ACTUATION_CONFIDENCE_THRESHOLD:
            actuation_code = "HIGH_IMPACT" if label == "High Impact" else "ASYMMETRY"

        if actuation_code:
            self._trigger_actuation(actuation_code)

        # Generate coaching + store recommendation. ml.recommendation_engine
        # existed as a documented "guaranteed fallback if the LLM call
        # fails" but nothing actually called it — a raised exception here
        # (LLM down, Ollama not running, network hiccup) meant NO
        # recommendation got stored at all, silently. Now it does.
        runner_history = self._build_runner_history(run_id)
        try:
            coaching_text = generate_coaching(
                related_metric=payload.get("related_metric"),
                predicted_label=label,
                confidence=confidence,
                details=payload.get("details", ""),
                runner_history=runner_history,
            )
        except Exception as e:
            logger.warning(
                f"LLM coaching failed ({e}) — falling back to rule-based "
                f"recommendation text so this window still gets a recommendation."
            )
            coaching_text = rule_based_recommend(label)

        self.db.insert_recommendation(
            run_id=run_id,
            timestamp=payload.get("window_start", time.time()),
            related_metric=payload.get("related_metric"),
            recommendation_text=f"{label} (confidence {confidence})",
            llm_explanation=coaching_text,
        )

        logger.info(f"Run {run_id}: {label} -> actuation={actuation_code}, coaching stored")

    def _build_runner_history(self, run_id: int, limit: int = 3) -> str:
        """
        Short summary of this run's last few recommendations, so the LLM
        can notice patterns (e.g. "still high impact after 3 windows")
        instead of treating every window as a first impression.
        """
        try:
            recent = self.db.get_recommendations(run_id)[-limit:]
        except Exception:
            return "No prior context available."
        if not recent:
            return "No prior context available."
        lines = [f"- {r['recommendation_text']}" for r in recent]
        return "Recent windows this run:\n" + "\n".join(lines)

    def _trigger_actuation(self, code: str):
        message = json.dumps({"code": code})
        self.mqtt.client.publish(config.MQTT_TOPICS["feedback"], message)
        logger.info(f"Published feedback: {code}")

    def stop(self):
        self.mqtt.disconnect()


if __name__ == "__main__":
    # Standalone test: publishes a few fake predictions to your local
    # broker and confirms the loop reacts correctly — no hardware,
    # no trained model, just mock messages proving the wiring works.
    import paho.mqtt.client as mqtt_lib

    db = DatabaseManager()
    loop = FeedbackLoop(db)
    loop.start()

    time.sleep(1)  # let the subscription settle

    test_publisher = mqtt_lib.Client()
    test_publisher.connect(config.MQTT_BROKER, config.MQTT_PORT)

    fake_predictions = [
        {"run_id": 1, "window_start": time.time(), "window_end": time.time() + 2,
         "predicted_label": "High Impact", "confidence": 0.85,
         "related_metric": "impact", "details": "left foot impact 4.5g"},
        {"run_id": 1, "window_start": time.time(), "window_end": time.time() + 2,
         "predicted_label": "Normal", "confidence": 0.9,
         "related_metric": None, "details": "all normal"},
    ]

    for pred in fake_predictions:
        test_publisher.publish(config.MQTT_TOPICS["predictions"], json.dumps(pred))
        time.sleep(1)

    time.sleep(2)
    loop.stop()
    print("Test complete — check recommendations table for stored coaching.") 