"""
StrideIQ Session Manager

Coordinates an entire running session.

Responsibilities:
- Create run
- Start BLE collection
- Stop BLE collection
- Extract features
- Close run
"""

import asyncio
import threading
import time

from backend.ble.receiver import BLEReceiver
from  ml.features_pipeline import process_run_windows 


class SessionManager:

    def __init__(self, db):

        self.db = db

        self.receiver = None

        self.thread = None

        self.running = False

        self.run_id = None

    def start_session(
        self,
        run_type="natural",
        duration_sec=30,
        notes=None,
        is_filmed=False
    ):

        if self.running:
            raise RuntimeError("Session already running")

        self.run_id = self.db.start_run(

            date=time.strftime("%Y-%m-%d"),

            start_time=time.strftime("%H:%M:%S"),

            run_type=run_type,

            notes=notes,

            is_filmed=is_filmed

        )

        self.receiver = BLEReceiver(self.run_id)

        self.thread = threading.Thread(

            target=lambda: asyncio.run(
                self.receiver.start()
            ),

            daemon=True 

        )

        self.thread.start()

        self.running = True
        self.start_timestamp = time.time()
        return self.run_id

    def stop_session(self, run_id=None):

        if not self.running:
            raise RuntimeError("No active session")

        if run_id is None:
            run_id = self.run_id

        if self.receiver:
            self.receiver.stop()

        process_run_windows(

            run_id,

            self.db

        )
        
        duration = int(time.time() - self.start_timestamp)

        self.db.stop_run(

            run_id,

            time.strftime("%H:%M:%S"),

            duration

        )

        self.running = False

        self.receiver = None

        self.thread = None

        self.run_id = None

        return run_id 