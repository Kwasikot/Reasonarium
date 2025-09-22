"""
llm/ollama_client.py

HTTP client for Ollama with streaming and non-streaming chat/text calls.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Iterator, List, Dict, Optional

try:
    import httpx
except ImportError:  # Allow module import, fail only on use
    httpx = None  # type: ignore

log = logging.getLogger("usefulclicker.llm")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] LLM: %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


class OllamaClient:
    def __init__(self):
        self.model = os.getenv("USEFULCLICKER_OLLAMA_MODEL", "llama3.2:latest")
        self.base_url = os.getenv("USEFULCLICKER_OLLAMA_BASE", "http://localhost:11434")
        log.info(f"Ollama client ready (model={self.model}).")

    # --- Non-streaming text ---
    def generate_text(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama HTTP client")
        use_model = model or self.model
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, object] = {"model": use_model, "prompt": prompt, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}
        resp = httpx.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Newer API returns 'response', older may use 'completion'
        text = (
            data.get("response")
            or data.get("completion")
            or data.get("text")
            or ""
        )
        return str(text).strip()

    def generate_list(self, prompt: str, separator: str = "\n", model: Optional[str] = None, temperature: Optional[float] = None):
        txt = self.generate_text(prompt, model=model, temperature=temperature)
        return [s.strip() for s in txt.split(separator) if s.strip()]

    # --- Non-streaming chat ---
    def generate_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama HTTP client")
        use_model = model or self.model
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, object] = {"model": use_model, "messages": messages, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}
        resp = httpx.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # According to docs, the assistant message resides in message.content
        try:
            msg = data.get("message") or {}
            return str(msg.get("content", "")).strip()
        except Exception:
            # Fallback: some variants return 'response'
            return str(data.get("response", "")).strip()

    # --- Streaming chat ---
    def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None) -> Iterator[str]:
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama HTTP client")
        use_model = model or self.model
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, object] = {"model": use_model, "messages": messages, "stream": True}
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}

        with httpx.stream("POST", url, json=payload, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                # Two common formats: {response: "token", done: false} or {message:{content:"accum"}}
                piece = None
                if isinstance(data, dict):
                    if "response" in data:
                        piece = data.get("response")
                    elif "message" in data and isinstance(data["message"], dict):
                        piece = data["message"].get("content")
                if piece:
                    yield str(piece)
                # Stop condition
                if bool(data.get("done")):
                    break

