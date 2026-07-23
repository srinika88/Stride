"""
StrideIQ Foot BLE Node

Receives accelerometer packets from a micro:bit foot sensor.

Packet format

L,SEQ,TIMESTAMP,AX,AY,AZ
R,SEQ,TIMESTAMP,AX,AY,AZ

Responsibilities
---------------
• Connect to micro:bit
• Parse packets
• Detect packet loss
• Timestamp on Raspberry Pi
• Deliver FootPacket objects to receiver.py
"""

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import Callable

from bleak import BleakClient
from bleak import BleakScanner

from backend import config

logger = logging.getLogger("StrideIQ-Foot")

UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

RECONNECT_DELAY = 2
STALE_TIMEOUT = 5


# ==========================================================
# Packet
# ==========================================================

@dataclass
class FootPacket:

    node: str

    sequence: int

    device_timestamp_ms: int

    acc_x: float
    acc_y: float
    acc_z: float

    pi_timestamp: float

    dropped_packets: int = 0


# ==========================================================
# Foot Node
# ==========================================================

class FootNode:

    def __init__(
        self,
        parent,
        node: str,
        callback: Callable[[FootPacket], None]
    ):

        self.parent = parent

        self.node = node.upper()

        self.callback = callback

        self.client = None

        self.rx_buffer = ""

        self.last_sequence = None

        self.last_packet_time = time.time()

    # -----------------------------------------------------

    async def find_device(self):

        if self.node == "L":

            if getattr(config, "LEFT_DEVICE_ADDRESS", None):
                return config.LEFT_DEVICE_ADDRESS

            expected_name = config.BLE_DEVICE_NAMES["foot_left"]

        else:

            if getattr(config, "RIGHT_DEVICE_ADDRESS", None):
                return config.RIGHT_DEVICE_ADDRESS

            expected_name = config.BLE_DEVICE_NAMES["foot_right"] 

        logger.info(f"{self.node}: scanning...")

        devices = await BleakScanner.discover(timeout=5)

        for device in devices:

            if device.name is None:
                continue

            if expected_name.lower() in device.name.lower():

                logger.info(
                    f"{self.node}: found {device.address}"
                )

                return device.address

        return None

    # -----------------------------------------------------

    def notification_handler(self, sender, data):

        self.last_packet_time = time.time()

        message = data.decode(
            "utf-8",
            errors="ignore"
        )

        self.rx_buffer += message

        while "\n" in self.rx_buffer:

            line, self.rx_buffer = self.rx_buffer.split("\n", 1)

            line = line.strip()

            if line:

                self.parse_packet(line)

    # -----------------------------------------------------

    def parse_packet(self, line):

        values = line.split(",")

        if len(values) != 6:

            logger.warning(f"{self.node}: bad packet\n{line}")

            return

        try:

            node = values[0]

            seq = int(values[1])

            timestamp = int(values[2])

            ax = int(values[3])
            ay = int(values[4])
            az = int(values[5])

        except Exception as e:

            logger.warning(f"{self.node}: parse error {e}")

            return

        if node != self.node:

            logger.warning(
                f"{self.node}: wrong packet ({node})"
            )

            return

        dropped = 0

        if self.last_sequence is not None:

            diff = seq - self.last_sequence

            if diff > 1:

                dropped = diff - 1

                logger.warning(
                    f"{self.node}: lost {dropped} packets"
                )

            elif diff == 0:

                logger.warning(
                    f"{self.node}: duplicate packet"
                )

                return

            elif diff < 0:

                if abs(diff) > 30000:

                    logger.info(
                        f"{self.node}: sequence rollover"
                    )

                else:

                    logger.warning(
                        f"{self.node}: out-of-order packet"
                    )

                    return

        self.last_sequence = seq

        packet = FootPacket(

            node=node,

            sequence=seq,

            device_timestamp_ms=timestamp,

            acc_x=ax,
            acc_y=ay,
            acc_z=az,

            pi_timestamp=time.time(),

            dropped_packets=dropped

        )

        self.callback(packet)

    # -----------------------------------------------------

    async def run(self):

        while self.parent.running:

            address = await self.find_device()

            if not address:

                logger.warning(
                    f"{self.node}: device not found"
                )

                await asyncio.sleep(RECONNECT_DELAY)

                continue

            try:

                async with BleakClient(address) as client:

                    self.client = client

                    logger.info(
                        f"{self.node}: connected"
                    )

                    await client.start_notify(

                        UART_TX_UUID,

                        self.notification_handler

                    )

                    while client.is_connected:

                        await asyncio.sleep(1)

                        if (

                            time.time()

                            -

                            self.last_packet_time

                            >

                            STALE_TIMEOUT

                        ):

                            logger.warning(
                                f"{self.node}: stale connection"
                            )

                            break

            except Exception as e:

                logger.error(
                    f"{self.node}: {e}"
                )

            await asyncio.sleep(RECONNECT_DELAY)

    # -----------------------------------------------------

    async def disconnect(self):

        try:

            if self.client and self.client.is_connected:

                await self.client.stop_notify(UART_TX_UUID)

                await self.client.disconnect()

                logger.info(
                    f"{self.node}: disconnected"
                )

        except Exception as e:

            logger.warning(
                f"{self.node}: disconnect error {e}"
            )

    # -----------------------------------------------------

    def is_connected(self):

        if self.client is None:

            return False

        return self.client.is_connected