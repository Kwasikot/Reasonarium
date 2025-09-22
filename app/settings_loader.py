from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


DEFAULT_PATH = os.path.join("settings", "reasonarium_settings.xml")


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
        root = self._root
        out: Dict[str, str] = {}
        if root is None:
            return out
        ui = root.find("ui_texts")
        if ui is None:
            return out
        # find best match
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

