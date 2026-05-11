import threading
import queue
import time
import math
import numpy as np
from config import *

if SD_OK:
    import sounddevice as sd

class AudioWatek:
    def __init__(self, gui_queue, klasyfikator=None):
        self.gui_q        = gui_queue
        self.klasyfikator = klasyfikator
        self.running      = False
        self.thread       = None
        self._peak        = DB_MIN
        self._peak_time   = 0.0

    def start(self, device_idx, channels):
        self.stop()
        self.running    = True
        self._peak      = DB_MIN
        self._peak_time = time.time()
        self.thread = threading.Thread(
            target=self._petla,
            args=(device_idx, channels),
            daemon=True
        )
        self.thread.start()

    def _petla(self, device_idx, channels):
        def callback(indata, frames, time_info, status):
            ch_dbs = []
            for c in range(indata.shape[1]):
                rms = float(np.sqrt(np.mean(indata[:, c] ** 2)))
                ch_dbs.append(20.0 * math.log10(max(rms, 1e-10)))

            avg_db = sum(ch_dbs) / len(ch_dbs)
            avg_db = max(DB_MIN, min(DB_MAX, avg_db))

            now = time.time()
            if avg_db >= self._peak:
                self._peak      = avg_db
                self._peak_time = now
            elif now - self._peak_time > 2.0:
                self._peak = max(avg_db, self._peak - 1.5)

            try:
                self.gui_q.put_nowait({
                    "db":       avg_db,
                    "peak":     self._peak,
                    "channels": ch_dbs,
                })
            except queue.Full:
                pass

            if self.klasyfikator is not None:
                mono = indata.mean(axis=1).copy()
                self.klasyfikator.dodaj_audio(mono)

        try:
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                blocksize  = BLOCK_SIZE,
                device     = device_idx,
                channels   = channels,
                dtype      = "float32",
                callback   = callback,
            ):
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            try:
                self.gui_q.put_nowait({"error": str(e)})
            except queue.Full:
                pass
        self.running = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
