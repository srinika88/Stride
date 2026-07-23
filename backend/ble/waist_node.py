"""
StrideIQ Waist BLE Node

Receives 9-axis IMU packets from the ESP32 waist sensor.

Packet format

W,SEQ,TIMESTAMP,
AX,AY,AZ,
GX,GY,GZ,
MX,MY,MZ

Example

W,1542,198233,
0.12,-0.88,9.71,
0.41,-0.12,0.07,
21.5,-13.2,44.1

Responsibilities

• Connect to ESP32 over BLE
• Parse 9-axis IMU packets
• Detect packet loss
• Timestamp packets on Raspberry Pi
• Deliver WaistPacket objects to receiver.py
"""

import asyncio
import logging
import time

from dataclasses import dataclass
from typing import Callable

from bleak import BleakClient
from bleak import BleakScanner

from backend import config


logger = logging.getLogger("StrideIQ-Waist")


UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

RECONNECT_DELAY = 2
STALE_TIMEOUT = 5


# ===========================================================
# Packet
# ===========================================================


@dataclass
class WaistPacket:

    node: str

    sequence: int

    device_timestamp_ms: int

    ax: float
    ay: float
    az: float

    gx: float
    gy: float
    gz: float

    mx: float
    my: float
    mz: float

    pi_timestamp: float

    dropped_packets: int = 0


# ===========================================================
# Waist Node
# ===========================================================


class WaistNode:

    def __init__(
        self,
        parent,
        callback: Callable[[WaistPacket], None]
    ):

        self.parent = parent

        self.callback = callback

        self.client = None

        self.rx_buffer = ""

        self.last_sequence = None

        self.last_packet_time = time.time()


    # ------------------------------------------------------

    async def find_device(self):

        if getattr(config, "WAIST_DEVICE_ADDRESS", None):

            return config.WAIST_DEVICE_ADDRESS

        logger.info("Scanning for waist node...")

        devices = await BleakScanner.discover(timeout=5)

        for device in devices:

            if device.name is None:
                continue

            expected_name = config.BLE_DEVICE_NAMES["waist"]
            if expected_name.lower() in device.name.lower(): 

                logger.info(
                    f"Found waist node {device.address}"
                )

                return device.address

        return None


    # ------------------------------------------------------

    def notification_handler(
        self,
        sender,
        data
    ):

        self.last_packet_time = time.time()

        message = data.decode(
            "utf-8",
            errors="ignore"
        )

        self.rx_buffer += message

        while "\n" in self.rx_buffer:

            line, self.rx_buffer = (
                self.rx_buffer.split("\n", 1)
            )

            line = line.strip()

            if line:

                self.parse_packet(line)


    # ------------------------------------------------------

    def parse_packet(self, line):

        values = line.split(",")

        if len(values) != 12:

            logger.warning(
                f"Bad waist packet:\n{line}"
            )

            return

        try:

            (
                node,
                seq,
                timestamp,

                ax,
                ay,
                az,

                gx,
                gy,
                gz,

                mx,
                my,
                mz

            ) = values

            seq = int(seq)

            timestamp = int(timestamp)

            ax = float(ax)
            ay = float(ay)
            az = float(az)

            gx = float(gx)
            gy = float(gy)
            gz = float(gz)

            mx = float(mx)
            my = float(my)
            mz = float(mz)

        except Exception as e:

            logger.warning(
                f"Parse error {e}"
            )

            return

        if node != "WAIST":

            logger.warning(
                "Invalid waist node ID"
            )

            return

        dropped = 0

        if self.last_sequence is not None:

            diff = seq - self.last_sequence

            if diff > 1:

                dropped = diff - 1

                logger.warning(
                    f"Waist lost {dropped} packets"
                )

            elif diff == 0:

                logger.warning(
                    "Duplicate waist packet"
                )

                return

            elif diff < 0:

                if abs(diff) > 30000:

                    logger.info(
                        "Waist sequence rollover"
                    )

                else:

                    logger.warning(
                        "Out-of-order waist packet"
                    )

                    return

        self.last_sequence = seq
        packet = WaistPacket(

            node=node,

            sequence=seq,

            device_timestamp_ms=timestamp,

            ax=ax,
            ay=ay,
            az=az,

            gx=gx,
            gy=gy,
            gz=gz,

            mx=mx,
            my=my,
            mz=mz,

            pi_timestamp=time.time(),

            dropped_packets=dropped

        )

        self.callback(packet)

    # ------------------------------------------------------

    async def run(self):

        while self.parent.running:

            address = await self.find_device()

            if not address:

                logger.warning(
                    "Waist device not found"
                )

                await asyncio.sleep(RECONNECT_DELAY)

                continue

            try:

                async with BleakClient(address) as client:

                    self.client = client

                    logger.info(
                        "Waist connected"
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
                                "Waist connection stale"
                            )

                            break

            except Exception as e:

                logger.error(
                    f"Waist: {e}"
                )

            await asyncio.sleep(RECONNECT_DELAY)

    # ------------------------------------------------------

    async def disconnect(self):

        try:

            if self.client and self.client.is_connected:

                await self.client.stop_notify(UART_TX_UUID)

                await self.client.disconnect()

                logger.info("Waist disconnected")

        except Exception as e:

            logger.warning(
                f"Disconnect error: {e}"
            )

    # ------------------------------------------------------

    def is_connected(self):

        if self.client is None:

            return False

        return self.client.is_connected