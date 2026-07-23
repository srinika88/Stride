"""
main.py — StrideIQ Edge Gateway

Main orchestration layer.

Connects:

    micro:bit Foot L
            |
            | BLE UART
            ↓
    receiver.py
            |
            +------------+
            |            |
            ↓            ↓
        MQTT          GaitAnalyzer
        broker             |
                            ↓
                    live features
                    feedback


Responsibilities:
    - Start BLE connections
    - Start MQTT client
    - Route incoming packets
    - Run real-time gait analysis loop
    - Publish features and feedback

Run:
    python -m backend.main
"""


import asyncio
import logging


from backend.ble.receiver import ( 
    BLEReceiver,
    FootPacket
)

from backend.mqtt.mqtt_client import (
    StrideMQTTClient
)

from backend.processing.gait_analyzer import (
    GaitAnalyzer,
    ANALYSIS_INTERVAL_SEC
)



logger = logging.getLogger(
    "strideiq.main"
)



# ============================================================
# Build System Components
# ============================================================

def build_system():

    """
    Creates and wires all StrideIQ components.

    BLE packet flow:

        BLE
         |
         ↓
    handle_packet()
         |
         +--> MQTT
         |
         +--> GaitAnalyzer
    """


    mqtt_client = StrideMQTTClient()



    analyzer = GaitAnalyzer(

        on_features=
            lambda features:
                mqtt_client.publish_features(
                    features
                ),


        on_feedback=
            lambda feedback:
                mqtt_client.publish_feedback(
                    feedback
                )
    )



    def handle_packet(packet: FootPacket):

        """
        Called whenever a foot sensor packet
        arrives from BLE.
        """


        logger.debug(
            f"Packet received: {packet}"
        )


        # --------------------------------
        # 1. Store/publish raw sensor data
        # --------------------------------

        mqtt_client.publish_foot_packet(
            packet
        )


        # --------------------------------
        # 2. Feed real-time analyzer
        # --------------------------------

        analyzer.ingest_packet(
            packet
        )



    ble_receiver = BLEReceiver(
        run_id =1,
        on_packet = handle_packet
    )


    return (
        mqtt_client,
        ble_receiver,
        analyzer
    )



# ============================================================
# Real-time Analysis Loop
# ============================================================

async def analysis_loop(
    analyzer: GaitAnalyzer
):

    """
    Runs gait analysis periodically.

    Does NOT run at BLE frequency.

    BLE:
        50Hz

    Analysis:
        every 1 second
    """


    while True:


        await asyncio.sleep(
            ANALYSIS_INTERVAL_SEC
        )


        try:

            features = analyzer.analyze()


            logger.debug(
                f"Features generated: {features}"
            )


        except Exception:

            logger.exception(
                "Gait analysis failed"
            )



# ============================================================
# Main Runtime
# ============================================================

async def run():


    mqtt_client, ble_receiver, analyzer = (
        build_system()
    )


    logger.info(
        "Starting MQTT client..."
    )


    mqtt_client.connect()



    logger.info(
        "Starting BLE receiver..."
    )



    await asyncio.gather(

        ble_receiver.run(),

        analysis_loop(
            analyzer
        )

    )



# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":


    logging.basicConfig(

        level=logging.INFO,

        format=
        "%(asctime)s | %(levelname)s | %(message)s"

    )



    try:

        asyncio.run(
            run()
        )


    except KeyboardInterrupt:


        logger.info(
            "StrideIQ shutdown"
        )