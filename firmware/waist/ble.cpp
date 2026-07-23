#include "ble.h"
#include "config.h"

void WaistBle::begin(std::function<void(String)> onFeedbackReceived) {
    NimBLEDevice::init(DEVICE_NAME);  // sets the advertised name explicitly —
                                        // this is the piece the micro:bit firmware
                                        // is missing; check the Pi sees "StrideIQ-Waist"
                                        // in a scan before trusting name-based pairing here too

    server = NimBLEDevice::createServer();
    static ServerCallbacks serverCb(this);
    server->setCallbacks(&serverCb); 

    NimBLEService* service = server->createService(NUS_SERVICE_UUID);

    txChar = service->createCharacteristic(
        NUS_TX_CHAR_UUID,
        NIMBLE_PROPERTY::NOTIFY
    );

    rxChar = service->createCharacteristic(
        NUS_RX_CHAR_UUID,
        NIMBLE_PROPERTY::WRITE 
    );
    rxCallbacks.callback = onFeedbackReceived;
    rxChar->setCallbacks(&rxCallbacks);

    service->start();

    NimBLEAdvertising* advertising = NimBLEDevice::getAdvertising();
    advertising->addServiceUUID(NUS_SERVICE_UUID);
    advertising->start();
}

void WaistBle::send(const String& line) {
    if (connected && txChar) {
        txChar->setValue(line.c_str());
        txChar->notify();
    }
}

bool WaistBle::isConnected() {
    return connected;
} 