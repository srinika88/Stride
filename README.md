# Stride - AI Running Coach

Stride is an AI-assisted running analysis system designed to help runners understand their movement patterns and receive actionable feedback.

The project explores how wearable sensors, computer vision, signal processing and machine learning can work together to turn movement data into understandable running insights.

## The Problem

Runners often receive limited, fragmented feedback about their running form. Understanding patterns such as cadence, foot-strike behaviour, symmetry and stability typically requires specialised equipment or expert analysis.

Stride explores a more accessible, data-driven approach to running feedback.

## What It Does

The system brings together multiple technical components:

- **Wearable sensing:** Collects movement data from connected sensor devices.
- **Signal processing:** Processes sensor streams and extracts movement-related features.
- **Computer vision:** Uses pose-analysis techniques to explore running movement.
- **Machine learning:** Experiments with movement classification and pattern analysis.
- **AI-assisted coaching:** Explores converting analysis outputs into understandable feedback.
- **Dashboard:** Provides an interface for viewing running insights.


## System Architecture

```text
Wearable Sensors / Camera
          |
          v
   Data Collection
          |
          v
 Signal Processing
          |
          v
 Feature Extraction
          |
          v
 Movement Analysis
          |
          v
 AI-Assisted Feedback
          |
          v
     User Dashboard
```

## Technology

The project explores technologies across:

- Python
- React and Vite
- Computer vision and pose estimation
- Machine learning
- Bluetooth Low Energy (BLE)
- Signal processing
- Local LLM integration

See individual project directories for implementation-specific dependencies.

## Repository Structure

| Directory | Purpose |
|---|---|
| `backend/` | Backend services and APIs |
| `cv/` | Computer vision and pose analysis |
| `ml/` | Machine learning experiments |
| `processing/` | Sensor data processing |
| `llm/` | AI-assisted feedback |
| `docs/` | Project documentation |
| `strideiq-frontend/` | Frontend application |


## Current Status

Stride is a research and development prototype.

The repository contains implementation work across sensing, movement analysis and AI-assisted feedback. Further work is needed to establish reproducible hardware validation, participant-level model evaluation, end-to-end demo evidence and production security.

## Running the Project

Setup instructions and dependencies are documented within the individual components. Refer to their respective README files where available.

## Future Improvements

- Reproducible end-to-end demonstration
- Expanded hardware validation and calibration
- Participant/session-level model evaluation
- Improved API security and deployment readiness
- User testing and feedback-driven iteration

## Project Context

Developed as part of a running-coach project at the National University of Singapore (NUS), May–July 2026.

## Disclaimer

Stride is an experimental running-analysis project, not a medical diagnostic or injury-prevention system.
