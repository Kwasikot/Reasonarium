from __future__ import annotations
from typing import Optional, Callable, List
import time
import threading
import queue
import numpy as np
try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore
import os


def transcribe_whisper(file_path: str, model: Optional[str] = None, language: Optional[str] = None) -> str:
    """Transcribe audio locally using OpenAI Whisper python package.
    Requires `whisper` pip package and ffmpeg in PATH.
    model: tiny/base/small/medium/large-v2 … or path to local model
    language: ISO code like 'en', 'ru' (optional)
    """
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Whisper package not available: {e}")

    use_model = model or os.getenv("WHISPER_MODEL", "base")
    try:
        w = whisper.load_model(use_model)
    except Exception as e:
        raise RuntimeError(f"Cannot load whisper model '{use_model}': {e}")
    try:
        opts = {}
        if language:
            opts["language"] = language
        result = w.transcribe(file_path, **opts)
        txt = (result.get("text") or "").strip()
        return txt
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


class WhisperStream:
    """Lightweight mic listener with simple VAD-like gating and Whisper transcription.

    - Uses sounddevice for realtime capture
    - Segments by RMS threshold (optionally with webrtcvad if installed)
    - On each detected phrase, runs whisper on CPU and calls on_text callback with recognized text
    """

    def __init__(self, on_text: Callable[[str], None], model: str = "base", samplerate: int = 16000, device: Optional[int] = None, language: Optional[str] = None):
        if sd is None:
            raise RuntimeError("sounddevice not available")
        try:
            import whisper  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Whisper package not available: {e}")
        self._whisper = whisper.load_model(model)
        self._on_text = on_text
        self._sr = int(samplerate)
        self._dev = device
        self._lang = language
        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None  # type: ignore

        # simple gating params
        self._gate = 800.0  # RMS threshold on int16 scale; adapt online
        self._hang_ms = 700
        self._min_ms = 450

    def start(self):
        def _cb(indata, frames, time, status):  # type: ignore
            if status:
                pass
            mono = indata[:, 0]
            pcm16 = (mono * 32767.0).astype(np.int16).tobytes()
            try:
                self._q.put_nowait(pcm16)
            except Exception:
                pass

        self._stop.clear()
        self._stream = sd.InputStream(device=self._dev, channels=1, samplerate=self._sr, dtype='float32', callback=_cb)
        self._stream.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._stream:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            if self._thread:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._thread = None

    def _loop(self):
        block_ms = 20
        bpr = int(self._sr * 2 / 1000)  # bytes per ms (int16)
        ring = bytearray()
        speaking = False
        last_voice_ts = 0.0
        accum: List[bytes] = []
        # quick auto-calibration
        calib = []
        calib_ms = 1000
        collected = 0
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            if not chunk and not speaking:
                continue
            if chunk:
                ring.extend(chunk)
            # consume frames
            while len(ring) >= bpr * block_ms:
                frame = bytes(ring[: bpr * block_ms]); del ring[: bpr * block_ms]
                # calibration
                if collected < calib_ms:
                    calib.append(self._rms(frame))
                    collected += block_ms
                    if collected >= calib_ms and calib:
                        mu = float(np.mean(calib))
                        sigma = float(np.std(calib))
                        self._gate = max(600.0, mu + 3.0 * max(sigma, 1.0))
                    continue
                # gate-based speech detect
                val = self._rms(frame)
                now = time.time()
                if val > self._gate:
                    speaking = True
                    last_voice_ts = now
                    accum.append(frame)
                else:
                    if speaking and (now - last_voice_ts) * 1000 > self._hang_ms:
                        dur_ms = len(accum) * block_ms
                        speaking = False
                        if dur_ms >= self._min_ms:
                            audio = b"".join(accum)
                            accum.clear()
                            self._transcribe(audio)
                        else:
                            accum.clear()

    def _transcribe(self, audio_bytes: bytes):
        try:
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            kw = dict(fp16=False)
            if self._lang:
                kw["language"] = self._lang
            res = self._whisper.transcribe(pcm, **kw)
            txt = (res.get("text") or "").strip()
            if txt:
                try:
                    self._on_text(txt)
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def _rms(frame: bytes) -> float:
        if not frame:
            return 0.0
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(x * x)) + 1e-6)
