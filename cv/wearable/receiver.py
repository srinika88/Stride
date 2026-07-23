"""
StrideIQ

receiver.py

Receives BLE packets from wearable nodes.

Responsibilities
----------------
- Discover BLE devices
- Connect automatically
- Receive UART packets
- Timestamp arrivals
- Validate packets
- Push packets into a queue

NO signal processing.
NO ML.
NO coaching.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from bleak import BleakClient, BleakScanner


# ============================================================
# Nordic UART UUIDs
# ============================================================

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"

UART_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


# ============================================================
# Expected BLE Device Names
# ============================================================

DEVICE_NAMES = {

    "L": "StrideIQ-FootL",

    "R": "StrideIQ-FootR",

    "W": "StrideIQ-Waist",

}


# ============================================================
# Raw Packet
# ============================================================

@dataclass
class RawPacket:
    """
    Raw accelerometer packet from a wearable.

    Packet format:

    NODE,SEQ,TIME,AX,AY,AZ
    """

    node: str

    sequence: int

    sensor_timestamp: int

    arrival_timestamp: float

    ax: float

    ay: float

    az: float


# ============================================================
# Receiver
# ============================================================

class WearableReceiver:

    """
    BLE Receiver.

    Manages all wearable nodes.
    """

    def __init__(self):

        # Active BLE clients
        self.clients: Dict[str, BleakClient] = {}

        # Latest packet from each node
        self.latest_packets: Dict[str, RawPacket] = {}

        # Packet queue
        self.packet_queue: asyncio.Queue[RawPacket] = asyncio.Queue()

        # Connection status
        self.connected = {

            "L": False,

            "R": False,

            "W": False,

        }

        print("[Receiver] Initialized.")
        # --------------------------------------------------------
    # Scan for Wearables
    # --------------------------------------------------------

    async def scan(self, timeout: float = 5.0):

        print("[Receiver] Scanning...")

        devices = await BleakScanner.discover(timeout=timeout)

        found = {}

        for device in devices:

            for node, expected_name in DEVICE_NAMES.items():

                if device.name == expected_name:

                    print(

                        f"[Receiver] Found "

                        f"{node}: {device.address}"

                    )

                    found[node] = device

        return found


    # --------------------------------------------------------
    # Connect to One Node
    # --------------------------------------------------------

    async def connect_node(

        self,

        node: str,

        device,

    ):

        print(

            f"[Receiver] Connecting to {node}..."

        )

        client = BleakClient(device)

        await client.connect()

        if not client.is_connected:

            raise RuntimeError(

                f"Failed to connect to {node}"

            )

        self.clients[node] = client

        self.connected[node] = True

        print(

            f"[Receiver] {node} connected."

        )

        # Subscribe to UART notifications

        await client.start_notify(

            UART_TX_UUID,

            self._notification_callback,

        )


    # --------------------------------------------------------
    # Connect to Every Available Device
    # --------------------------------------------------------

    async def connect_all(self):

        found = await self.scan()

        for node, device in found.items():

            try:

                await self.connect_node(

                    node,

                    device,

                )

            except Exception as e:

                print(

                    f"[Receiver] "

                    f"{node} failed: {e}"

                )

        print(

            "[Receiver] Connection summary:",

            self.connected,

        )
        # --------------------------------------------------------
    # BLE Notification Callback
    # --------------------------------------------------------

    def _notification_callback(

        self,

        sender: int,

        data: bytearray,

    ):

        """
        Called automatically whenever a wearable
        sends a UART packet.
        """

        try:

            message = data.decode("utf-8").strip()

            packet = self._parse_packet(message)

            if packet is None:

                return

            self.latest_packets[packet.node] = packet

            self.packet_queue.put_nowait(packet)

        except Exception as e:

            print(f"[Receiver] Packet error: {e}")


    # --------------------------------------------------------
    # Parse UART Packet
    # --------------------------------------------------------

    def _parse_packet(

        self,

        message: str,

    ) -> Optional[RawPacket]:

        """
        Expected format:

        L,12,4214,-12,45,981
        """

        parts = message.split(",")

        if len(parts) != 6:

            return None

        try:

            node = parts[0]

            if node not in DEVICE_NAMES:

                return None

            packet = RawPacket(

                node=node,

                sequence=int(parts[1]),

                sensor_timestamp=int(parts[2]),

                arrival_timestamp=time.perf_counter(),

                ax=float(parts[3]),

                ay=float(parts[4]),

                az=float(parts[5]),

            )

            return packet

        except Exception:

            return None
        # --------------------------------------------------------
    # Disconnect One Device
    # --------------------------------------------------------

    async def disconnect_node(

        self,

        node: str,

    ):

        client = self.clients.get(node)

        if client is None:

            return

        try:

            await client.disconnect()

        except Exception:

            pass

        self.connected[node] = False

        print(f"[Receiver] {node} disconnected.")


    # --------------------------------------------------------
    # Disconnect Everything
    # --------------------------------------------------------

    async def disconnect_all(self):

        for node in list(self.clients.keys()):

            await self.disconnect_node(node)


    # --------------------------------------------------------
    # Auto Reconnect
    # --------------------------------------------------------

    async def reconnect_missing(self):

        found = await self.scan()

        for node, device in found.items():

            if self.connected[node]:

                continue

            try:

                await self.connect_node(

                    node,

                    device,

                )

            except Exception:

                pass
        # --------------------------------------------------------
    # Get Next Packet
    # --------------------------------------------------------

    async def get_packet(self) -> RawPacket:

        """
        Wait until a packet arrives.
        """

        packet = await self.packet_queue.get()

        return packet


    # --------------------------------------------------------
    # Non-blocking Packet Read
    # --------------------------------------------------------

    def get_latest_packet(

        self,

        node: str,

    ) -> Optional[RawPacket]:

        return self.latest_packets.get(node)


    # --------------------------------------------------------
    # Queue Size
    # --------------------------------------------------------

    def queue_size(self):

        return self.packet_queue.qsize()


    # --------------------------------------------------------
    # Connection Status
    # --------------------------------------------------------

    def status(self):

        return dict(self.connected) 
