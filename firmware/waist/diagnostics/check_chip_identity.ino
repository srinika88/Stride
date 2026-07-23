// StrideIQ — chip identity check
// Flash this BY ITSELF (not the main waist firmware) to confirm
// whether this board is a real MPU9250 or an MPU6500 sold under a
// fake MPU9250 label. Open Serial Monitor at 115200 baud after flashing.

#include <Wire.h>

#define MPU_ADDR   0x68   // matches your ADO=GND wiring
#define MPU_WHOAMI 0x75
#define AK_ADDR    0x0C   // AK8963 magnetometer sub-chip address
#define AK_WHOAMI  0x00

uint8_t readRegister(uint8_t devAddr, uint8_t regAddr) {
    Wire.beginTransmission(devAddr);
    Wire.write(regAddr);
    Wire.endTransmission(false);
    Wire.requestFrom(devAddr, (uint8_t)1);
    if (Wire.available()) return Wire.read();
    return 0xFF; // sentinel: no response
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Wire.begin(21, 22); // SDA, SCL -- matches your firmware's I2C_SDA_PIN/I2C_SCL_PIN

    uint8_t mpuId = readRegister(MPU_ADDR, MPU_WHOAMI);
    Serial.print("MPU WHO_AM_I = 0x");
    Serial.println(mpuId, HEX);
    Serial.println("  (0x71 = MPU9250 identity, 0x70/0x73 = MPU6500 family -- but some fakes spoof this, so check below too)");

    // MPU9250 must first enable I2C bypass (INT_PIN_CFG, reg 0x37, bit 1)
    // before the AK8963 becomes visible on the main bus.
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x37);
    Wire.write(0x02); // BYPASS_EN
    Wire.endTransmission();
    delay(10);

    uint8_t akId = readRegister(AK_ADDR, AK_WHOAMI);
    Serial.print("AK8963 (magnetometer) WHO_AM_I = 0x");
    Serial.println(akId, HEX);

    if (akId == 0x48) {
        Serial.println("RESULT: Real MPU9250 -- magnetometer confirmed present.");
    } else if (akId == 0xFF) {
        Serial.println("RESULT: No response from magnetometer address -- this is an MPU6500, no magnetometer.");
    } else {
        Serial.println("RESULT: Unexpected value -- inconclusive, re-check wiring/power and retry.");
    }
}

void loop() {}
