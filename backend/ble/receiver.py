"""
StrideIQ BLE Receiver (combined)

Receives raw sensor packets from all three wearable nodes:
    - Left foot micro:bit   (6-field packet: NODE,SEQ,TS,AX,AY,AZ)
    - Right foot micro:bit  (6-field packet: NODE,SEQ,TS,AX,AY,AZ)
    - Waist ESP32            (12-field packet: WAIST,SEQ,TS,AX,AY,AZ,GX,GY,GZ,MX,MY,MZ)

Responsibilities:
    - BLE Central connection (3 concurrent peripherals)
    - Parse UART packets per node's format
    - Identify node (L / R / WAIST)
    - Detect packet loss / duplicates / out-of-order via sequence number
    - Timestamp on Raspberry Pi arrival
    - Buffer incoming packets in memory
    - Batch-write to SQLite via DatabaseManager
    - Publish each packet to MQTT 
"""

import asyncio
import logging
import time

from dataclasses import dataclass, asdict
from collections import deque
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner

from backend import config
from backend.calibration import load_calibration, apply_calibration
from backend.database_manager import DatabaseManager
from backend.mqtt.mqtt_client import StrideMQTTClient


# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("StrideIQ-BLE")


# =====================================================
# Nordic UART Service UUIDs (shared by all three node types —
# MakeCode micro:bits and the ESP32, if it exposes the same
# Nordic UART profile)
# =====================================================

UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"


# =====================================================
# Device configuration
#
# Names come from config.BLE_DEVICE_NAMES. All three boards must be
# flashed to advertise under these specific names (bluetooth.setName(...)
# in firmware) or the scan may match the wrong board.
#
# Leave address=None initially. Once you've confirmed a scan finds the
# correct name, hardcode the MAC address here to skip scanning on
# future connects (faster + more reliable reconnects).
# =====================================================

FOOT_DEVICES = {
    "L": {
        "name": config.BLE_DEVICE_NAMES["foot_left"],
        "address": None
    },
    "R": {
        "name": config.BLE_DEVICE_NAMES["foot_right"],
        # Hardcoded — this board's name doesn't reliably appear in scan
        # results (BLE splits advertising data across an ADV packet and
        # a separate scan-response packet; a scan sometimes only catches
        # one of the two, especially in a room with lots of other BLE
        # traffic). Confirmed via a longer 20s scan showing the same
        # MAC consistently advertising the Nordic UART Service UUID
        # right next to the Left board (RSSI=-36, closest match in the
        # room) — this is genuinely the right foot micro:bit, name
        # resolution just isn't reliable for it. address takes priority
        # over name in find_device(), so this skips scanning entirely.
        "address": "E0:5A:1B:5F:64:E6"
    }
}

WAIST_DEVICE = {
    "name": config.BLE_DEVICE_NAMES["waist"],
    "address": None
}

BUFFER_SIZE = 5000
RECONNECT_DELAY = 2
STALE_TIMEOUT = 5
DB_FLUSH_SIZE = 50


# =====================================================
# Packet structures
# =====================================================

@dataclass
class FootPacket:
    node: str
    sequence: int
    device_timestamp_ms: int
    acc_x: int
    acc_y: int
    acc_z: int
    pi_timestamp: float
    dropped_packets: int = 0


@dataclass
class WaistPacket:
    node: str
    sequence: int
    device_timestamp_ms: int
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    pi_timestamp: float
    dropped_packets: int = 0


# =====================================================
# Shared sequence-gap logic (used by both node types so the
# loss/duplicate/rollover rules stay identical everywhere)
# =====================================================

def check_sequence(node_tag: str, last_sequence: Optional[int], seq: int) -> Optional[int]:
    """
    Returns dropped-count (0 if none), or None if this packet should be
    discarded (duplicate / out-of-order, not a real rollover).
    """
    if last_sequence is None:
        return 0

    diff = seq - last_sequence

    if diff > 1:
        dropped = diff - 1
        logger.warning(f"{node_tag}: lost {dropped} packets")
        return dropped

    if diff == 0:
        logger.warning(f"{node_tag}: duplicate packet")
        return None

    if diff < 0:
        if abs(diff) > 30000:
            logger.info(f"{node_tag}: sequence rollover")
            return 0
        else:
            logger.warning(f"{node_tag}: out-of-order packet")
            return None

    return 0


# =====================================================
# Foot Node (Left / Right)
# =====================================================

class FootNode:

    def __init__(self, parent, node_tag: str, callback: Callable[[FootPacket], None]):
        self.parent = parent
        self.node_tag = node_tag
        self.callback = callback
        self.client = None
        self.rx_buffer = ""
        self.last_sequence = None
        self.last_packet_time = time.time()

    async def find_device(self):
        device_conf = FOOT_DEVICES[self.node_tag]

        if device_conf["address"]:
            return device_conf["address"]

        logger.info(f"{self.node_tag}: scanning BLE devices...")
        devices = await BleakScanner.discover(timeout=5)

        for device in devices:
            if device.name and device_conf["name"].lower() in device.name.lower():
                logger.info(f"{self.node_tag}: found {device.name} {device.address}")
                return device.address

        return None

    def notification_handler(self, sender, data):
        self.last_packet_time = time.time()
        message = data.decode("utf-8", errors="ignore")
        self.rx_buffer += message

        while "\n" in self.rx_buffer:
            line, self.rx_buffer = self.rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.parse_packet(line)

    def parse_packet(self, line):
        values = line.split(",")

        if len(values) != 6:
            logger.warning(f"{self.node_tag}: bad packet {line}")
            return

        try:
            node, seq, ts, ax, ay, az = values
            seq = int(seq)
            ts = int(ts)
            ax = int(ax)
            ay = int(ay)
            az = int(az)
        except ValueError:
            logger.warning(f"{self.node_tag}: invalid packet format")
            return

        if node != self.node_tag:
            logger.warning(
                f"{self.node_tag}: wrong node identifier (got '{node}') "
                f"— possible device mispairing"
            )
            return

        dropped = check_sequence(self.node_tag, self.last_sequence, seq)
        if dropped is None:
            return  # duplicate / out-of-order, discard

        self.last_sequence = seq

        packet = FootPacket(
            node=node,
            sequence=seq,
            device_timestamp_ms=ts,
            acc_x=ax,
            acc_y=ay,
            acc_z=az,
            pi_timestamp=time.time(),
            dropped_packets=dropped
        )

        self.callback(packet)

    async def run(self):
        while self.parent.running:
            address = await self.find_device()

            if not address:
                logger.warning(f"{self.node_tag}: device not found")
                await asyncio.sleep(RECONNECT_DELAY)
                continue

            try:
                async with BleakClient(address) as client:
                    self.client = client
                    logger.info(f"{self.node_tag}: connected")

                    await client.start_notify(UART_TX_UUID, self.notification_handler)

                    while client.is_connected:
                        await asyncio.sleep(1)

                        if time.time() - self.last_packet_time > STALE_TIMEOUT:
                            logger.warning(f"{self.node_tag}: connection stale")
                            break

            except Exception as e:
                logger.error(f"{self.node_tag}: {e}")

            await asyncio.sleep(RECONNECT_DELAY) 


# =====================================================
# Waist Node (ESP32, 9-axis)
# =====================================================

class WaistNode:
    """
    Same structure/lifecycle as FootNode — connection loop, notification
    handling, sequence-gap detection, stale-connection recovery. Only
    parse_packet() differs, since the waist sends 9 axes instead of 3.
    """

    def __init__(self, parent, callback: Callable[[WaistPacket], None]):
        self.parent = parent
        self.node_tag = "WAIST"
        self.callback = callback
        self.client = None
        self.rx_buffer = ""
        self.last_sequence = None
        self.last_packet_time = time.time()

    async def find_device(self):
        if WAIST_DEVICE["address"]:
            return WAIST_DEVICE["address"]

        logger.info("WAIST: scanning BLE devices...")
        devices = await BleakScanner.discover(timeout=5)
        for device in devices:
            if device.name and WAIST_DEVICE["name"].lower() in device.name.lower():
                logger.info(f"WAIST: found {device.name} {device.address}")
                return device.address
        return None

    def notification_handler(self, sender, data):
        self.last_packet_time = time.time()
        message = data.decode("utf-8", errors="ignore")
        self.rx_buffer += message

        while "\n" in self.rx_buffer:
            line, self.rx_buffer = self.rx_buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.parse_packet(line)

    def parse_packet(self, line):
        values = line.split(",")

        if len(values) != 12:  # WAIST,seq,ts,ax,ay,az,gx,gy,gz,mx,my,mz
            logger.warning(f"WAIST: bad packet (expected 12 fields, got {len(values)}): {line}")
            return

        try:
            node = values[0]
            seq = int(values[1])
            ts = int(values[2])
            ax, ay, az, gx, gy, gz, mx, my, mz = [float(v) for v in values[3:12]]
        except ValueError:
            logger.warning("WAIST: invalid packet format")
            return

        if node != "WAIST":
            logger.warning(f"WAIST: wrong node identifier (got '{node}') — possible device mispairing")
            return

        dropped = check_sequence("WAIST", self.last_sequence, seq)
        if dropped is None:
            return  # duplicate / out-of-order, discard

        self.last_sequence = seq

        packet = WaistPacket(
            node=node, sequence=seq, device_timestamp_ms=ts,
            acc_x=ax, acc_y=ay, acc_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            mag_x=mx, mag_y=my, mag_z=mz,
            pi_timestamp=time.time(), dropped_packets=dropped,
        )
        self.callback(packet)

    async def run(self):
        while self.parent.running:
            address = await self.find_device()

            if not address:
                logger.warning("WAIST: device not found")
                await asyncio.sleep(RECONNECT_DELAY)
                continue

            try:
                async with BleakClient(address) as client:
                    self.client = client
                    logger.info("WAIST: connected")

                    await client.start_notify(UART_TX_UUID, self.notification_handler)

                    while client.is_connected:
                        await asyncio.sleep(1)
                        if time.time() - self.last_packet_time > STALE_TIMEOUT:
                            logger.warning("WAIST: connection stale")
                            break

            except Exception as e:
                logger.error(f"WAIST: {e}")

            await asyncio.sleep(RECONNECT_DELAY)


# =====================================================
# BLE Receiver Manager (all three nodes)
# =====================================================

class BLEReceiver:

    def __init__(self, run_id: int, on_packet: Optional[Callable] = None):
        self.on_packet = on_packet
        self.running = True
        self.run_id = run_id

        self.db = DatabaseManager()
        self.db_buffer = []

        self.mqtt = StrideMQTTClient(config.MQTT_BROKER, config.MQTT_PORT)
        self.mqtt.connect()

        self.buffers = {
            "L": deque(maxlen=BUFFER_SIZE),
            "R": deque(maxlen=BUFFER_SIZE),
            "WAIST": deque(maxlen=BUFFER_SIZE),
        }

        self.nodes = [
            FootNode(self, "L", self.handle_packet),
            FootNode(self, "R", self.handle_packet),
            WaistNode(self, self.handle_waist_packet),
        ]

        # Loaded once at startup, not per-packet — apply_calibration() is
        # cheap, but there's no reason to re-read the JSON file 50x/sec.
        # None if backend/calibration.py hasn't been run yet; packets pass
        # through uncorrected in that case (with a one-time warning below).
        self.waist_calibration = load_calibration()
        if self.waist_calibration is None:
            logger.warning(
                "No waist calibration found (data/waist_calibration.json missing) — "
                "waist readings will be used raw, uncorrected for gyro/accel bias. "
                "Run backend/calibration.py first for accurate pelvic tilt/rotation."
            )

    # -------------------------------------------------
    # Foot packet handling
    # -------------------------------------------------
    def packet_to_db_row(self, packet):
        """
        Handles both FootPacket and WaistPacket. Foot packets don't have
        gyro/mag fields, so getattr() falls back to None for those —
        one function instead of two near-identical ones.
        """
        if packet.node == "L":
            node_name = "foot_left"
        elif packet.node == "R":
            node_name = "foot_right"
        else:
            node_name = "waist"

        node_id = self.db.get_node_id(node_name)

        return {
            "run_id": self.run_id,
            "node_id": node_id,
            "timestamp": packet.pi_timestamp,
            "packet_seq": packet.sequence,
            "ax": getattr(packet, "acc_x", getattr(packet, "ax", None)),
            "ay": getattr(packet, "acc_y", getattr(packet, "ay", None)),
            "az": getattr(packet, "acc_z", getattr(packet, "az", None)), 
            "gx": getattr(packet, "gyro_x", None),
            "gy": getattr(packet, "gyro_y", None),
            "gz": getattr(packet, "gyro_z", None),
            "mx": getattr(packet, "mag_x", None),
            "my": getattr(packet, "mag_y", None),
            "mz": getattr(packet, "mag_z", None),
        }

    def handle_packet(self, packet: FootPacket):
        self.buffers[packet.node].append(packet)

        if packet.sequence % 100 == 0:
            logger.debug(asdict(packet))

        row = self.packet_to_db_row(packet)
        self._buffer_and_flush(row)

        self.mqtt.publish_foot_packet(packet) 

        if self.on_packet:
            self.on_packet(packet)

    # -------------------------------------------------
    # Waist packet handling
    # -------------------------------------------------
    def handle_waist_packet(self, packet: WaistPacket):
        # Correct gyro/accel bias (and mag hard-iron offset, if this board
        # has a magnetometer) BEFORE this packet reaches the DB, MQTT, or
        # any callback — every downstream consumer (features_waist.py,
        # the dashboard, MQTT subscribers) sees corrected values, rather
        # than each needing to know to apply calibration itself.
        if self.waist_calibration is not None:
            (
                packet.acc_x, packet.acc_y, packet.acc_z,
                packet.gyro_x, packet.gyro_y, packet.gyro_z,
                packet.mag_x, packet.mag_y, packet.mag_z,
            ) = apply_calibration(
                packet.acc_x, packet.acc_y, packet.acc_z,
                packet.gyro_x, packet.gyro_y, packet.gyro_z,
                packet.mag_x, packet.mag_y, packet.mag_z,
                self.waist_calibration,
            )

        self.buffers["WAIST"].append(packet)

        if packet.sequence % 100 == 0:
            logger.debug(asdict(packet))

        row = self.packet_to_db_row(packet)
        self._buffer_and_flush(row)

        self.mqtt.publish_waist_packet(packet)

        if self.on_packet:
            self.on_packet(packet)

    # -------------------------------------------------
    # Shared DB buffering
    # -------------------------------------------------
    def _buffer_and_flush(self, row: dict): 
        self.db_buffer.append(row)
        if len(self.db_buffer) >= DB_FLUSH_SIZE:
            self.db.insert_raw_readings(self.db_buffer)
            self.db_buffer.clear()

    # -------------------------------------------------
    def stop(self):
        if self.db_buffer:
            self.db.insert_raw_readings(self.db_buffer)
            self.db_buffer.clear()
        
        self.running = False
        self.mqtt.disconnect() 

    async def start(self):
        self.running = True

        tasks = [asyncio.create_task(node.run()) for node in self.nodes]

        while self.running:
            await asyncio.sleep(0.2)

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# =====================================================
# Test Run
# =====================================================

if __name__ == "__main__":
    receiver = BLEReceiver(run_id=1)

    try:
        asyncio.run(receiver.start())
    except KeyboardInterrupt:
        logger.info("BLE receiver stopped") 