from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


DEFAULT_PATH = os.path.join("settings", "reasonarium_settings.xml")
UI_TEXTS_PATH = os.path.join("settings", "ui_texts_translations.xml")


class Settings:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None
        if os.path.exists(path):
            try:
                self._tree = ET.parse(path)
                self._root = self._tree.getroot()
            except Exception:
                self._tree = None
                self._root = None

    def ok(self) -> bool:
        return self._root is not None

    # Languages
    def languages(self) -> List[Tuple[str, str]]:
        root = self._root
        if root is None:
            return [("en", "English"), ("ru", "Русский")]
        out: List[Tuple[str, str]] = []
        langs = root.find("languages")
        if langs is not None:
            for el in langs.findall("lang"):
                code = el.get("code") or "en"
                name = el.get("name") or code
                out.append((code, name))
        if not out:
            out = [("en", "English"), ("ru", "Русский")]
        return out

    def default_language(self) -> str:
        root = self._root
        if root is None:
            return "en"
        langs = root.find("languages")
        if langs is not None and langs.get("default"):
            return langs.get("default") or "en"
        return "en"

    # UI texts
    def ui_texts(self, lang: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        # Prefer external translations file
        if os.path.exists(UI_TEXTS_PATH):
            try:
                tree = ET.parse(UI_TEXTS_PATH)
                ui = tree.getroot()
                # find best match
                best = None
                for lnode in ui.findall("lang"):
                    if (lnode.get("code") or "").lower() == (lang or "").lower():
                        best = lnode
                        break
                if best is None:
                    # fallback to English, else first
                    for lnode in ui.findall("lang"):
                        if (lnode.get("code") or "").lower() == "en":
                            best = lnode
                            break
                    if best is None:
                        best = ui.find("lang")
                if best is not None:
                    for t in best.findall("text"):
                        k = t.get("key") or ""
                        v = (t.text or "").strip()
                        if k:
                            out[k] = v
                    return out
            except Exception:
                pass
        # Fallback: legacy inline ui_texts in reasonarium_settings.xml
        root = self._root
        if root is None:
            return out
        ui = root.find("ui_texts")
        if ui is None:
            return out
        best = None
        for lnode in ui.findall("lang"):
            if (lnode.get("code") or "").lower() == (lang or "").lower():
                best = lnode
                break
        if best is None:
            best = ui.find("lang")
        if best is not None:
            for t in best.findall("text"):
                k = t.get("key") or ""
                v = (t.text or "").strip()
                if k:
                    out[k] = v
        return out

    # Prompts: returns list of (id, title, filepath)
    def prompts(self, lang: str) -> List[Tuple[str, str, str]]:
        root = self._root
        out: List[Tuple[str, str, str]] = []
        if root is None:
            return out
        pnode = root.find("prompts")
        if pnode is None:
            return out
        for pr in pnode.findall("prompt"):
            pid = pr.get("id") or ""
            title = None
            file_path = None
            # title/lang and file/lang
            for t in pr.findall("title"):
                if (t.get("lang") or "").lower() == lang.lower():
                    title = (t.text or "").strip()
            for f in pr.findall("file"):
                if (f.get("lang") or "").lower() == lang.lower():
                    file_path = (f.text or "").strip()
            # fallback to first available
            if not title:
                t0 = pr.find("title")
                title = (t0.text if t0 is not None else pid).strip()
            if not file_path:
                f0 = pr.find("file")
                file_path = (f0.text if f0 is not None else "").strip()
            if pid and file_path:
                out.append((pid, title or pid, file_path))
        return out

    def disciplines(self, lang: str) -> List[str]:
        root = self._root
        if root is None:
            return []
        droot = root.find("disciplines")
        if droot is None:
            return []
        best = None
        for lnode in droot.findall("lang"):
            if (lnode.get("code") or "").lower() == lang.lower():
                best = lnode
                break
        if best is None:
            best = droot.find("lang")
        items: List[str] = []
        if best is not None:
            for it in best.findall("item"):
                val = (it.text or "").strip()
                if val:
                    items.append(val)
        return items

    def openai_models(self) -> List[str]:
        root = self._root
        out: List[str] = []
        if root is None:
            return out
        mroot = root.find("openai_models")
        if mroot is None:
            return out
        for it in mroot.findall("model"):
            val = (it.text or "").strip()
            if val:
                out.append(val)
        return out

    def default_openai_model(self) -> Optional[str]:
        root = self._root
        if root is None:
            return None
        mroot = root.find("openai_models")
        if mroot is None:
            return None
        return mroot.get("default")

    def colors(self) -> dict:
        root = self._root
        out = {}
        if root is None:
            return out
        cnode = root.find("colors")
        if cnode is None:
            return out
        def txt(tag: str, default: str = "") -> str:
            el = cnode.find(tag)
            return (el.text or default).strip() if el is not None else default
        out["counterargument"] = txt("counterargument", "#fbc02d")
        out["question"] = txt("question", "#7CFC00")
        return out

    # --- Ollama settings ---
    def ollama_endpoint(self) -> Optional[str]:
        root = self._root
        if root is None:
            return None
        node = root.find("ollama")
        if node is None:
            return None
        return node.get("endpoint") or "http://localhost:11434"

    def ollama_models(self) -> List[str]:
        """Return list of Ollama models. Try HTTP /api/tags if declared, else fallback list."""
        root = self._root
        out: List[str] = []
        if root is None:
            return out
        onode = root.find("ollama")
        if onode is None:
            return out
        # Try HTTP source
        src = onode.find("models")
        if src is not None and (src.get("source") or "") == "http":
            url = src.get("url") or ""
            if url:
                try:
                    import httpx
                    r = httpx.get(url, timeout=5.0)
                    r.raise_for_status()
                    data = r.json()
                    # Expecting {models:[{name:..}, ...]} or list
                    if isinstance(data, dict) and "models" in data:
                        items = data.get("models") or []
                    else:
                        items = data
                    names = []
                    for it in items or []:
                        name = None
                        if isinstance(it, dict):
                            name = it.get("name") or it.get("model")
                        elif isinstance(it, str):
                            name = it
                        if name and name not in names:
                            names.append(name)
                    if names:
                        out = names
                except Exception:
                    # Ignore fetch errors; fall back below
                    pass
        # Fallback models
        if not out:
            fnode = onode.find("fallback_models")
            if fnode is not None:
                for it in fnode.findall("model"):
                    val = (it.text or "").strip()
                    if val:
                        out.append(val)
        return out

    # Whisper models
    def whisper_models(self) -> List[str]:
        root = self._root
        out: List[str] = []
        if root is None:
            return out
        node = root.find("whisper_models")
        if node is None:
            return out
        for it in node.findall("model"):
            val = (it.text or "").strip()
            if val:
                out.append(val)
        return out

    def default_whisper_model(self) -> Optional[str]:
        root = self._root
        if root is None:
            return None
        node = root.find("whisper_models")
        if node is None:
            return None
        return node.get("default")
