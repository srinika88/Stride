// StrideIQ — Waist Node Main
// Same protocol as the foot micro:bits, one line per sample:
//   WAIST,<seq>,<ts_ms>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<mx>,<my>,<mz>
// Maps directly onto raw_sensor_data's columns — no new schema needed.
//
// Hardware note: this board's chip is an MPU6500 (confirmed via
// WHO_AM_I), not a genuine MPU9250 — no onboard magnetometer. mx/my/mz
// are always sent as the literal string "nan" (not via imu.h's NAN
// value serialized through String(float,3), which isn't guaranteed to
// print as text a Python float() parser recognizes on every platform).
// backend/ble/waist_node.py does float("nan") on each field, and
// processing/features_waist.py already has a real-NaN-triggered
// gyro-only fallback for pelvic rotation — this is what makes that
// fallback actually engage instead of silently getting fed zeros.

#include "config.h"
#include "imu.h"
#include "ble.h"

WaistImu imu;
WaistBle ble;

uint32_t packetSeq = 0;

void onFeedback(String msg) {
    // Expected: "FEEDBACK,<code>"  e.g. "FEEDBACK,EXCESSIVE_PELVIC_TILT"
    int commaIndex = msg.indexOf(',');
    if (commaIndex == -1) return;

    String code = msg.substring(commaIndex + 1);
    // TODO: trigger buzzer/LED here once the waist enclosure has an actuator wired —
    // same pattern as the micro:bit's triggerFeedback(), just no built-in
    // speaker/LED matrix on a bare ESP32, so this needs real hardware first.
    Serial.print("Feedback received: ");
    Serial.println(code);
}

void setup() {
    Serial.begin(115200);

    imu.begin();  // logs its own status; MPU6500 has no mag, see imu.cpp

    ble.begin(onFeedback);
    Serial.println("StrideIQ waist node ready, advertising as " DEVICE_NAME);
}

void loop() {
    if (ble.isConnected()) {
        ImuReading r = imu.read();

        if (r.valid) {
            uint32_t ts = millis();

            String line = "WAIST," + String(packetSeq) + "," + String(ts) + ","
                + String(r.ax, 3) + "," + String(r.ay, 3) + "," + String(r.az, 3) + ","
                + String(r.gx, 3) + "," + String(r.gy, 3) + "," + String(r.gz, 3) + ","
                + "nan,nan,nan";  // no magnetometer on this chip — see header note

            ble.send(line);
            packetSeq++;
        }
    }

    delay(SAMPLE_INTERVAL_MS);
} 
