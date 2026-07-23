# Sensor Packet Format

Every wearable sends data to the Raspberry Pi in JSON format.

## Packet Structure

```json
{
    "timestamp": 1721056890,
    "node": "left",
    "ax": 0.32,
    "ay": -0.17,
    "az": 0.98,
    "gx": null,
    "gy": null,
    "gz": null,
    "mx": null,
    "my": null,
    "mz": null
}
```

## Node Types

- left
- right
- chest
- waist

## Sensor Availability

| Node | Accelerometer | Gyroscope | Magnetometer |
|------|---------------|-----------|--------------|
| Left Foot | ✓ | ✗ | ✗ |
| Right Foot | ✓ | ✗ | ✗ |
| Chest | ✓ | ✗ | ✗ |
| Waist | ✓ | ✓ | ✓ |

## Notes

- Every packet is timestamped by the Raspberry Pi when received.
- Feet and chest send only accelerometer data.
- Waist sends full 9-axis IMU data.