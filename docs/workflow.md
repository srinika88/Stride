# StrideIQ Workflow

Runner starts run

↓

Wearables collect IMU data

- Left Foot (micro:bit)
- Right Foot (micro:bit)
- Chest (micro:bit)
- Waist (ESP32 + 9-axis IMU)

↓

BLE Transmission

↓

Raspberry Pi receives packets

↓

MQTT publishes sensor data

↓

SQLite stores raw sensor data

↓

Feature Extraction

↓

Machine Learning Classifier

↓

Recommendation Engine

↓

LangChain

↓

Ollama

↓

Flask REST API

↓

Streamlit Dashboard

↓

Runner views live metrics and receives AI coaching