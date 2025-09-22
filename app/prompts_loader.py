from __future__ import annotations
import os
from typing import List, Tuple


def list_prompt_files(base_dir: str = "UsefulPrompts") -> List[Tuple[str, str]]:
    """Return list of (name, path) for prompt files in UsefulPrompts directory."""
    if not os.path.isdir(base_dir):
        return []
    out: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(base_dir)):
        if not fname.lower().endswith((".md", ".markdown")):
            continue
        path = os.path.join(base_dir, fname)
        out.append((fname, path))
    return out


def read_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

