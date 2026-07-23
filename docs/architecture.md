# StrideIQ Architecture

## Overview

StrideIQ is an AIoT wearable running coach that analyzes a runner's biomechanics in real time using four wearable sensor nodes and an edge-computing Raspberry Pi.

The system detects running-form deviations, predicts biomechanical issues using machine learning, recommends corrective actions, and explains them in natural language using a locally hosted LLM.

---

## Hardware

### Wearable Nodes

- Chest: micro:bit V2 (accelerometer)
- Left Foot: micro:bit V2 (accelerometer)
- Right Foot: micro:bit V2 (accelerometer)
- Waist: ESP32 + 9-axis IMU

### Edge Device

- Raspberry Pi 5

Responsibilities:

- BLE Central
- MQTT Broker/Client
- SQLite Database
- Machine Learning
- LangChain
- Ollama
- Flask Backend
- Streamlit Dashboard

---

## Software Stack

Firmware
↓
BLE Communication
↓
MQTT
↓
SQLite
↓
Feature Extraction
↓
Machine Learning
↓
Recommendation Engine
↓
LangChain
↓
Ollama
↓
Flask API
↓
Streamlit Dashboard

---

## Data Flow

Wearable Sensors

↓

BLE

↓

Raspberry Pi

↓

MQTT

↓

SQLite

↓

Feature Extraction

↓

Machine Learning

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

---

## Machine Learning Pipeline

Raw IMU Data

↓

Sliding Window Segmentation

↓

Feature Extraction

↓

Feature Scaling

↓

Model Training

↓

Hyperparameter Tuning (GridSearchCV)

↓

Best Model Selection

↓

Prediction

↓

Recommendation Generation

---

## AI Components

### Machine Learning

Responsible for:

- Detecting running-form deviations
- Predicting injury-related biomechanical issues

### Recommendation Engine

Responsible for:

- Selecting corrective exercises and coaching actions

### LangChain + Ollama

Responsible for:

- Converting structured recommendations into natural-language coaching

The LLM does not perform prediction or recommendation; it only generates human-readable explanations.