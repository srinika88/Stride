/**
 * StrideIQ — Left Foot Node
 * 
 * micro:bit V2, MakeCode (JavaScript), requires the "Bluetooth" extension
 * 
 * Streams raw accelerometer readings over BLE UART to the Raspberry Pi.
 * 
 * Kept deliberately dumb per config.py's design principle: no computation
 * 
 * on-device, just sample + send. Peak detection, filtering, and feature
 * 
 * extraction all happen on the Pi.
 */
bluetooth.onBluetoothConnected(function () {
    bleConnected = true
    basic.showIcon(IconNames.Yes)
    basic.clearScreen()
})
bluetooth.onBluetoothDisconnected(function () {
    bleConnected = false
    basic.showIcon(IconNames.No)
})
// ------------------------------------------------------------
// Visual connection status while idle — helps debug mounting/BLE
// on a real run without needing the Pi's logs open
// ------------------------------------------------------------
input.onButtonPressed(Button.A, function () {
    if (bleConnected) {
        basic.showIcon(IconNames.Yes)
    } else {
        basic.showIcon(IconNames.No)
    }
    basic.pause(500)
    basic.clearScreen()
})
function triggerFeedback (code: string) {
    // Different pattern per issue type — lets the runner distinguish
    // feedback types by feel/sound without looking at the device
    if (code == "HIGH_IMPACT") {
        basic.showIcon(IconNames.Square)
        music.playTone(880, music.beat(BeatFraction.Half))
    } else if (code == "ASYMMETRY") {
        basic.showIcon(IconNames.Triangle)
        music.playTone(660, music.beat(BeatFraction.Half))
    } else {
        basic.showIcon(IconNames.Diamond)
        music.playTone(440, music.beat(BeatFraction.Half))
    }
    basic.pause(600)
    basic.clearScreen()
}
// ------------------------------------------------------------
// Listen for feedback commands from the Pi
// Expected RX format: "FEEDBACK,<code>"  e.g. "FEEDBACK,HIGH_IMPACT"
// ------------------------------------------------------------
bluetooth.onUartDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    msg = bluetooth.uartReadUntil(serial.delimiters(Delimiters.NewLine))
    parts = msg.split(",")
    if (parts[0] == "FEEDBACK") {
        triggerFeedback(parts[1])
    }
})
let packetSeq = 0
let payload = ""
let ts = 0
let az = 0
let ay = 0
let ax = 0
let parts: string[] = []
let msg = ""
let bleConnected = false
// ------------------------------------------------------------
// Config — must match backend/config.py exactly
// ------------------------------------------------------------
let DEVICE_NAME = "StrideIQ-FootL"
// 50 Hz, matches DEFAULT_SAMPLING_RATE
let SAMPLE_INTERVAL_MS = 20
// prefix so the Pi can tell L and R apart if needed
let NODE_TAG = "L"
// ------------------------------------------------------------
// Bluetooth setup
// ------------------------------------------------------------
bluetooth.setTransmitPower(7)
bluetooth.startUartService()
// ------------------------------------------------------------
// Startup indicator
// ------------------------------------------------------------
basic.showString("L")
basic.pause(500)
basic.clearScreen()
// ------------------------------------------------------------
// Main sampling loop
// ------------------------------------------------------------
basic.forever(function () {
    if (bleConnected) {
        ax = input.acceleration(Dimension.X)
        ay = input.acceleration(Dimension.Y)
        az = input.acceleration(Dimension.Z)
        ts = input.runningTime()
        // CSV format: tag,seq,timestamp_ms,ax,ay,az
        // e.g. "L,4821,193882,-102,34,981"
        payload = "" + NODE_TAG + "," + packetSeq + "," + ts + "," + ax + "," + ay + "," + az
        bluetooth.uartWriteString("" + payload + "\n")
        packetSeq += 1
    }
    basic.pause(SAMPLE_INTERVAL_MS)
})