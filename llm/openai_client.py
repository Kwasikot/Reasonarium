# llm/openai_client.py
import os, logging
import httpx
from typing import Iterator, List, Dict, Any, Optional
from openai import OpenAI

log = logging.getLogger("usefulclicker.llm")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] LLM: %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.info("OPENAI_API_KEY is not set -> using MOCK.")
            raise RuntimeError("OPENAI_API_KEY missing")

        # Явно собираем httpx.Client, чтобы OpenAI SDK НЕ создавал свой
        # и не подсовывал 'proxies' из своего кода.
        proxy = os.getenv("USEFULCLICKER_OPENAI_PROXY")  # опционально
        trust_env = str(os.getenv("USEFULCLICKER_OPENAI_TRUST_ENV", "0")).lower() in {"1","true","yes","on"}
        timeout = int(os.getenv("USEFULCLICKER_OPENAI_TIMEOUT", "60"))
        if proxy:
            http_client = httpx.Client(proxies=proxy, timeout=timeout, trust_env=trust_env)
            proxy_state = f"explicit (trust_env={trust_env})"
        else:
            # без прокси; при необходимости можно включить автоподхват системных переменных
            http_client = httpx.Client(timeout=timeout, trust_env=trust_env)
            proxy_state = f"off (trust_env={trust_env})"

        self.client = OpenAI(api_key=api_key, http_client=http_client)
        # Default model; override via env or UI
        self.model = os.getenv("USEFULCLICKER_OPENAI_MODEL", "gpt-4o-mini")
        log.info(f"OpenAI client ready (default_model={self.model}, proxy={proxy_state}).")

    def generate_text(self, prompt: str, model: str | None = None, temperature: float | None = None) -> str:
        log.info("generate_text()")
        use_model = model or self.model
        log.info(f"OpenAI generate_text: using model={use_model} temperature={temperature}")
        use_temp = 1 if temperature is None else float(temperature)
        resp = self.client.chat.completions.create(
            model=use_model,
            messages=[{"role":"user","content":prompt}],
            temperature=use_temp,
        )
        txt = (resp.choices[0].message.content or "").strip()
        log.info(f"OK ({len(txt)} chars).")
        return txt

    def generate_list(self, prompt: str, separator: str = "\n", model: str | None = None, temperature: float | None = None):
        txt = self.generate_text(prompt, model=model, temperature=temperature)
        items = [s.strip() for s in txt.split(separator) if s.strip()]
        log.info(f"split -> {len(items)} items")
        return items

    # --- Added: Chat + Streaming interfaces ---
    def generate_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Synchronous non-streaming chat completion. Returns full assistant text."""
        use_model = model or self.model
        use_temp = 1 if temperature is None else float(temperature)
        resp = self.client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=use_temp,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> Iterator[str]:
        """Yield assistant tokens as they arrive (Chat Completions stream)."""
        use_model = model or self.model
        use_temp = 1 if temperature is None else float(temperature)
        try:
            stream = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=use_temp,
                stream=True,
            )
        except Exception as e:
            log.info(f"OpenAI stream init failed: {e}")
            # Fallback: non-streaming
            yield self.generate_chat(messages, model=use_model, temperature=use_temp)
            return

        # Iterate through streamed chunks, surface only content deltas
        for chunk in stream:  # type: ignore
            try:
                if not chunk:
                    continue
                # New SDK exposes choices[...].delta.content incrementally
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    piece = getattr(delta, "content", None)
                    if piece:
                        yield piece
                        continue
                # Some variants might carry full message content directly
                msg = getattr(choice, "message", None)
                if msg is not None:
                    piece = getattr(msg, "content", None)
                    if piece:
                        yield piece
            except Exception:
                # Be tolerant to schema drift
                continue

    # --- Audio APIs ---
    def transcribe_file(self, file_path: str, model: Optional[str] = None) -> str:
        """Transcribe an audio file using OpenAI transcription model.
        Prefers gpt-4o-mini-transcribe if available, falls back to whisper-1.
        """
        use_model = model or os.getenv("USEFULCLICKER_OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
        try:
            with open(file_path, 'rb') as f:
                resp = self.client.audio.transcriptions.create(model=use_model, file=f)
            # SDK typically returns an object with .text
            text = getattr(resp, 'text', None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            # Fallback to whisper-1
            try:
                with open(file_path, 'rb') as f:
                    resp = self.client.audio.transcriptions.create(model='whisper-1', file=f)
                text = getattr(resp, 'text', None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {e}")
        return ""

    def tts_synthesize(self, text: str, out_path: str, voice: Optional[str] = None, model: Optional[str] = None, format: str = 'mp3') -> str:
        """Synthesize speech to out_path. Returns the file path."""
        use_model = model or os.getenv("USEFULCLICKER_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        use_voice = voice or os.getenv("USEFULCLICKER_OPENAI_TTS_VOICE", "alloy")
        try:
            # Try streaming-to-file API
            with self.client.audio.speech.with_streaming_response.create(
                model=use_model,
                voice=use_voice,
                input=text,
                format=format,
            ) as resp:
                resp.stream_to_file(out_path)
            return out_path
        except Exception:
            # Fallback non-streaming
            try:
                resp = self.client.audio.speech.create(model=use_model, voice=use_voice, input=text, format=format)
                # Some SDKs use .content
                data = getattr(resp, 'content', None)
                if data is None and hasattr(resp, 'read'):
                    data = resp.read()
                if data is None:
                    # Last resort: write str(resp)
                    data = bytes(str(resp), 'utf-8')
                with open(out_path, 'wb') as f:
                    f.write(data)
                return out_path
            except Exception as e:
                raise RuntimeError(f"TTS failed: {e}")
