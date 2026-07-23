"""
Tests for backend.mqtt.mqtt_client.StrideMQTTClient's construction and
topic config -- no live Mosquitto broker needed (instantiating a paho
Client doesn't connect anywhere).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import config
from backend.mqtt.mqtt_client import StrideMQTTClient


def test_client_constructs_with_defaults():
    client = StrideMQTTClient()
    assert client.host == config.MQTT_BROKER
    assert client.port == config.MQTT_PORT
    assert client.connected is False


def test_client_constructs_with_overrides():
    client = StrideMQTTClient(host="test.broker.local", port=1884)
    assert client.host == "test.broker.local"
    assert client.port == 1884


def test_expected_topics_present():
    for key in ("raw_left", "raw_right", "raw_waist", "features", "predictions", "feedback"):
        assert key in config.MQTT_TOPICS
        assert config.MQTT_TOPICS[key].startswith("strideiq/")


def test_callbacks_wired_on_construction():
    client = StrideMQTTClient()
    assert client.client.on_connect == client.on_connect
    assert client.client.on_disconnect == client.on_disconnect
