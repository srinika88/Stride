// StrideIQ — Waist Node BLE Wrapper
// Same Nordic UART Service pattern as the foot micro:bits: streams
// CSV lines out over the TX characteristic, listens for "FEEDBACK,<code>"
// on the RX characteristic to trigger actuation. Uses NimBLE-Arduino
// (lighter weight than the stock ESP32 BLE library).

#ifndef STRIDEIQ_BLE_H
#define STRIDEIQ_BLE_H

#include <NimBLEDevice.h>
#include <functional>

class WaistBle {
public:
    void begin(std::function<void(String)> onFeedbackReceived);
    void send(const String& line);
    bool isConnected();

private:
    NimBLEServer* server = nullptr;
    NimBLECharacteristic* txChar = nullptr;
    NimBLECharacteristic* rxChar = nullptr;
    bool connected = false;

    class ServerCallbacks : public NimBLEServerCallbacks {
    public:
        WaistBle* parent;
        ServerCallbacks(WaistBle* p) : parent(p) {}
        void onConnect(NimBLEServer* server, NimBLEConnInfo& connInfo) override { 
        parent->connected = true; 
}
        void onDisconnect(NimBLEServer* server, NimBLEConnInfo& connInfo, int reason) override { 
        parent->connected = false;
       // Correct way to restart advertising via the server instance
        server->getAdvertising()->start(); 
}
    };

    class RxCallbacks : public NimBLECharacteristicCallbacks {
    public:
        std::function<void(String)> callback;
        void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& connInfo) override { 
            if (callback) {
                callback(String(c->getValue().c_str()));
            }
        }
    };

    RxCallbacks rxCallbacks;
};

#endif