// StrideIQ — Waist Node IMU Wrapper
// NOTE: the physical chip on this board turned out to be an MPU6500
// (confirmed via WHO_AM_I = 0x70), NOT a genuine MPU9250 — same 0x68
// I2C address and pinout, but no onboard AK8963 magnetometer. That's
// why mpu.setup() was originally failing (the library's init checks
// for the AK8963 and doesn't find one). We're keeping the same
// MPU9250 library class since it drives the 6-axis part fine; mag
// fields are just never populated from real hardware — see read()
// below and the note in getMagX()/Y()/Z() usage there.
//
// Fusion into pelvic tilt/rotation still happens entirely on the Pi.
// processing/features_waist.py's compute_pelvic_rotation() already
// has a graceful no-magnetometer fallback (gyro-only dead-reckoning) —
// it just needs mx/my/mz to genuinely be NaN, not 0.0, to trigger it.

#ifndef STRIDEIQ_IMU_H
#define STRIDEIQ_IMU_H

#include "MPU9250.h"

struct ImuReading {
    float ax, ay, az;  // g
    float gx, gy, gz;  // deg/s
    float mx, my, mz;  // uT — always NaN on this board (no magnetometer present)
    bool valid;
};

class WaistImu {
public:
    bool begin();
    ImuReading read();

private:
    MPU9250 mpu;
};

#endif 