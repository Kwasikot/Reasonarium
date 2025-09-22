from __future__ import annotations
from typing import Optional
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

