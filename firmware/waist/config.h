// StrideIQ — Waist Node Config
// Mirrors backend/config.py naming so the two stay easy to cross-reference.

#ifndef STRIDEIQ_CONFIG_H
#define STRIDEIQ_CONFIG_H

// Must match config.BLE_DEVICE_NAMES["waist"] exactly
#define DEVICE_NAME "StrideIQ-Waist"
#define NODE_ID "WAIST"
// Matches config.DEFAULT_SAMPLING_RATE (50 Hz -> 20ms)
#define SAMPLE_INTERVAL_MS 20

// Nordic UART Service UUIDs — same ones the foot micro:bits use,
// so the Pi-side receiver pattern (FootNode-style class) is reusable
// for the waist node with minimal changes.
#define NUS_SERVICE_UUID   "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_TX_CHAR_UUID   "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  // waist -> Pi (notify)
#define NUS_RX_CHAR_UUID   "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  // Pi -> waist (write, feedback)

// I2C pins for MPU9250 — adjust if your ESP32 dev board wiring differs
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define MPU9250_ADDRESS 0x68  // default; becomes 0x69 if the AD0 pin is pulled high


// Gyroscope calibration samples
#define GYRO_CALIBRATION_SAMPLES 500


// BLE notification size
#define BLE_PACKET_SIZE 128

#endif 