from __future__ import annotations
import threading
import queue
import wave
import tempfile
import os
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None  # type: ignore


class AudioRecorder:
    """Simple WAV recorder using sounddevice InputStream + writer thread.

    start(): opens a temp WAV file and begins capturing.
    stop(): stops stream and writer, closes file, returns path to WAV.
    """

    def __init__(self, device: Optional[int] = None, samplerate: int = 16000, channels: int = 1):
        if sd is None:
            raise RuntimeError("sounddevice is not available; install it to use recording")
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream: Optional[sd.InputStream] = None  # type: ignore
        self._thread: Optional[threading.Thread] = None
        self._wav: Optional[wave.Wave_write] = None
        self._path: Optional[str] = None
        self._stop_evt = threading.Event()

    @staticmethod
    def list_input_devices() -> list[dict]:
        if sd is None:
            return []
        devs = sd.query_devices()
        out = []
        for idx, d in enumerate(devs):
            try:
                if int(d.get('max_input_channels', 0)) > 0:
                    out.append({"index": idx, "name": d.get('name', f'Device {idx}'), "hostapi": d.get('hostapi')})
            except Exception:
                continue
        return out

    def _callback(self, indata, frames, time, status):  # type: ignore
        if status:
            # Drop status; keep robust
            pass
        # Convert to int16
        try:
            data = np.array(indata)
            if data.dtype != np.int16:
                # Assume float32 [-1,1)
                data = np.clip(data, -1.0, 1.0)
                data = (data * 32767.0).astype(np.int16)
            self._q.put_nowait(data.copy())
        except Exception:
            pass

    def start(self) -> str:
        if sd is None:
            raise RuntimeError("sounddevice not available")
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        self._path = path
        self._wav = wave.open(path, 'wb')
        self._wav.setnchannels(self.channels)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(self.samplerate)

        def writer():
            try:
                while not self._stop_evt.is_set():
                    try:
                        chunk = self._q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        if self._wav is not None:
                            self._wav.writeframes(chunk.tobytes())
                    except Exception:
                        continue
            finally:
                try:
                    if self._wav is not None:
                        self._wav.close()
                except Exception:
                    pass

        self._stop_evt.clear()
        self._thread = threading.Thread(target=writer, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()
        return path

    def stop(self) -> Optional[str]:
        # Stop stream
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        # Stop writer
        self._stop_evt.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._thread = None
        out = self._path
        self._path = None
        return out

