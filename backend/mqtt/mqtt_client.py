"""
StrideIQ MQTT Client

Handles:
- BLE sensor packet publishing
- Feature publishing
- ML prediction publishing
- Feedback subscription

Architecture:

BLE
 |
 v
MQTT
 |
 +--> SQLite
 +--> ML pipeline
 +--> Dashboard
 +--> Wearable feedback
"""


import json
import logging

from dataclasses import asdict, is_dataclass
from typing import Callable, Optional
from backend import config 
import paho.mqtt.client as mqtt



logger = logging.getLogger("StrideIQ-MQTT")



# ======================================================
# MQTT Topics
# ======================================================


TOPIC_SENSOR_WAIST = (
    "strideiq/sensor/waist/raw"
)


TOPIC_SENSOR_FOOT_L = (
    "strideiq/sensor/footL/raw"
)


TOPIC_SENSOR_FOOT_R = (
    "strideiq/sensor/footR/raw"
)


TOPIC_FEATURES = (
    "strideiq/features"
)


TOPIC_PREDICTIONS = (
    "strideiq/predictions"
)


TOPIC_FEEDBACK_L = (
    "strideiq/feedback/footL"
)


TOPIC_FEEDBACK_R = (
    "strideiq/feedback/footR"
)



FOOT_TOPIC_MAP = {

    "L": TOPIC_SENSOR_FOOT_L,

    "R": TOPIC_SENSOR_FOOT_R 

}



DEFAULT_PORT = 1883 



# ======================================================
# Serialization
# ======================================================


def to_json(payload):

    if is_dataclass(payload):

        return json.dumps(
            asdict(payload)
        )


    if isinstance(payload, dict): 

        return json.dumps(payload)


    raise TypeError(
        f"Unsupported MQTT payload {type(payload)}"
    )



# ======================================================
# MQTT Client
# ======================================================


class StrideMQTTClient:


    def __init__(
        self,
        host=config.MQTT_BROKER,
        port=config.MQTT_PORT
    ):


        self.host = host

        self.port = port


        self.client = mqtt.Client(
            client_id="strideiq-edge"
        )


        self.connected = False


        self.feedback_callback = None



        self.client.on_connect = (
            self.on_connect
        )


        self.client.on_disconnect = (
            self.on_disconnect
        )


        self.client.on_message = (
            self.on_message
        )



    # --------------------------------------------------

    def connect(self):

        logger.info(
            f"Connecting MQTT {self.host}:{self.port}"
        )


        self.client.connect(
            self.host,
            self.port,
            keepalive=30
        )


        self.client.loop_start()



    # --------------------------------------------------

    def disconnect(self):

        self.client.loop_stop()

        self.client.disconnect()



    # --------------------------------------------------

    def on_connect(
        self,
        client,
        userdata,
        flags,
        rc
    ):


        if rc == 0:

            self.connected = True


            logger.info(
                "MQTT connected"
            )


            client.subscribe(
                TOPIC_FEEDBACK_L
            )


            client.subscribe(
                TOPIC_FEEDBACK_R
            )


        else:

            logger.error(
                f"MQTT failed rc={rc}"
            )



    # --------------------------------------------------

    def on_disconnect(
        self,
        client,
        userdata,
        rc
    ):


        self.connected = False


        logger.warning(
            "MQTT disconnected"
        )



    # --------------------------------------------------

    def on_message(
        self,
        client,
        userdata,
        msg
    ):


        if self.feedback_callback is None:

            return



        try:

            payload = json.loads(
                msg.payload.decode()
            )


            self.feedback_callback(
                msg.topic,
                payload
            )


        except Exception as e:


            logger.error(
                f"Feedback error {e}"
            )



    # --------------------------------------------------

    def set_feedback_callback(
        self,
        callback: Callable
    ):

        self.feedback_callback = callback



    # ==================================================
    # Publishing
    # ==================================================


    def publish_foot_packet(
        self,
        packet
    ):


        topic = FOOT_TOPIC_MAP.get(
            packet.node
        )


        if topic:

            self.publish(
                topic,
                packet,
                qos=0
            )



    def publish_waist_packet(
        self,
        packet
    ):

        self.publish(
            TOPIC_SENSOR_WAIST,
            packet,
            qos=0
        )



    def publish_features(
        self,
        features
    ):

        self.publish(
            TOPIC_FEATURES,
            features,
            qos=1
        )



    def publish_prediction(
        self,
        prediction
    ):

        self.publish(
            TOPIC_PREDICTIONS,
            prediction,
            qos=1
        )



    def publish_feedback(
        self,
        foot,
        feedback
    ):


        topic = (
            TOPIC_FEEDBACK_L
            if foot=="L"
            else TOPIC_FEEDBACK_R
        )


        self.publish(
            topic,
            feedback,
            qos=1
        )



    # --------------------------------------------------

    def publish(
        self,
        topic,
        payload,
        qos=0
    ):


        try:

            body = to_json(
                payload
            )


            result = self.client.publish(

                topic,

                body,

                qos=qos,

                retain=False

            )


            if result.rc != mqtt.MQTT_ERR_SUCCESS:

                logger.warning(
                    f"Publish failed {topic}"
                )


        except Exception as e:

            logger.error(
                f"MQTT publish error {e}"
            )



# ======================================================
# Test
# ======================================================

if __name__ == "__main__":


    logging.basicConfig(
        level=logging.INFO
    )


    mqtt_client = StrideMQTTClient()


    mqtt_client.connect()


    try:

        while True:

            pass


    except KeyboardInterrupt:

        mqtt_client.disconnect()