#include "imu.h"
#include "config.h"
#include <Wire.h>
#include <math.h>  // NAN

bool WaistImu::begin() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

    // mpu.setup() returns false on this board because the library's
    // init routine also tries to bring up the AK8963 magnetometer,
    // which doesn't exist on this MPU6500 chip (WHO_AM_I = 0x70,
    // confirmed via a raw I2C register read — the chip DOES ACK on
    // the bus at 0x68, this isn't a wiring/address problem). That's a
    // known, understood failure mode now, not an unknown fault — so
    // we log it and continue rather than halting like a genuine
    // "sensor not detected" case would warrant.
    if (!mpu.setup(MPU9250_ADDRESS)) {
        Serial.println(
            "mpu.setup() reported failure — expected on this board "
            "(MPU6500, no magnetometer; see imu.h). Continuing with "
            "accel/gyro only."
        );
    }

    // One-time calibration on first boot — accel/gyro calibration needs
    // the board held still and flat. Left as default (uncalibrated) here;
    // backend/calibration.py handles gyro/accel bias correction on the
    // Pi side instead, working from raw streamed values.

    return true;
}

ImuReading WaistImu::read() {
    ImuReading r;

    if (mpu.update()) {
        r.ax = mpu.getAccX(); r.ay = mpu.getAccY(); r.az = mpu.getAccZ();
        r.gx = mpu.getGyroX(); r.gy = mpu.getGyroY(); r.gz = mpu.getGyroZ();

        // No AK8963 on this chip — do NOT call mpu.getMagX()/Y()/Z().
        // On this hardware those calls return stale/zero readings, not
        // NaN, which would trick processing/features_waist.py's
        // has_mag check into thinking real magnetometer data exists
        // and feeding zeros into the tilt-compensated heading math —
        // silently WRONG pelvic rotation, worse than no data at all.
        // Sending genuine NaN here is what makes that code's existing
        // gyro-only fallback actually trigger correctly.
        r.mx = NAN; r.my = NAN; r.mz = NAN;

        r.valid = true;
    } else {
        r.valid = false;
    }

    return r;
} 