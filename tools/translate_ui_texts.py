#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import re
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from app.settings_loader import Settings  # reuse paths and parsing


def load_languages(settings_path: str) -> List[Tuple[str, str]]:
    st = Settings(settings_path)
    if not st.ok():
        raise SystemExit(f"Cannot read settings: {settings_path}")
    return st.languages()


def load_ui_texts_map(ui_path: str) -> ET.ElementTree:
    if not os.path.exists(ui_path):
        raise SystemExit(f"Translations file not found: {ui_path}")
    return ET.parse(ui_path)


def lang_node_for(tree: ET.ElementTree, code: str) -> ET.Element | None:
    root = tree.getroot()
    for ln in root.findall('lang'):
        if (ln.get('code') or '').lower() == code.lower():
            return ln
    return None


def english_source(tree: ET.ElementTree) -> Dict[str, str]:
    ln = lang_node_for(tree, 'en')
    if ln is None:
        raise SystemExit("English (en) must be present as a source for translation.")
    out: Dict[str, str] = {}
    for t in ln.findall('text'):
        k = t.get('key') or ''
        v = (t.text or '').strip()
        if k:
            out[k] = v
    return out


def build_prompt(lang_name: str, base: Dict[str, str]) -> str:
    lines = []
    for k, v in base.items():
        lines.append(f"{k} = {v}")
    joined = "\n".join(lines)
    return (
        f"Respond strictly in {lang_name}.\n\n"
        "Translate the UI labels on the right side of '=' into the target language.\n"
        "Keep keys (left side) unchanged. Return plain text, one per line, in the format: key = translation.\n"
        "Do not add quotes or numbering. Do not explain.\n\n"
        f"{joined}\n"
    )


def choose_llm(args):
    engine = args.engine.lower()
    if engine == 'openai':
        from llm.openai_client import LLMClient
        return LLMClient()
    elif engine == 'ollama':
        from llm.ollama_client import OllamaClient
        return OllamaClient()
    else:
        raise SystemExit("engine must be 'openai' or 'ollama'")


def parse_pairs(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip('-').strip()
        if not line:
            continue
        # split at first '='
        m = re.match(r"([^=]+)=(.+)$", line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k and v:
            out[k] = v
    return out


def ensure_lang(tree: ET.ElementTree, code: str, pairs: Dict[str, str]):
    root = tree.getroot()
    ln = ET.Element('lang', {'code': code})
    for k, v in pairs.items():
        t = ET.Element('text', {'key': k})
        t.text = v
        ln.append(t)
    root.append(ln)


def main():
    ap = argparse.ArgumentParser(description="Translate ui_texts_translations.xml to all supported languages")
    ap.add_argument('--settings', default=os.path.join('settings', 'reasonarium_settings.xml'))
    ap.add_argument('--ui', default=os.path.join('settings', 'ui_texts_translations.xml'))
    ap.add_argument('--engine', default='openai', choices=['openai','ollama'])
    ap.add_argument('--model', default=None)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--limit', type=int, default=0, help='Translate only first N missing languages (for testing)')
    args = ap.parse_args()

    langs = load_languages(args.settings)
    ui_tree = load_ui_texts_map(args.ui)
    base = english_source(ui_tree)
    client = choose_llm(args)

    missing: List[Tuple[str, str]] = []
    for code, name in langs:
        if code.lower() in {'en'}:
            continue
        if lang_node_for(ui_tree, code) is None:
            missing.append((code, name))

    if args.limit > 0:
        missing = missing[:args.limit]

    if not missing:
        print("No missing languages. Nothing to do.")
        return

    for code, name in missing:
        prompt = build_prompt(name or code, base)
        print(f"Translating {code} → {name}...")
        try:
            txt = client.generate_text(prompt, model=args.model, temperature=args.temperature)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        pairs = parse_pairs(txt)
        # Ensure we got at least half of keys
        if len(pairs) < max(1, int(0.5 * len(base))):
            print(f"  WARN: low coverage ({len(pairs)}/{len(base)}). Skipping.")
            continue
        # fill missing keys with English to keep structure
        for k, v in base.items():
            if k not in pairs:
                pairs[k] = v
        ensure_lang(ui_tree, code, pairs)

    # Write back
    ui_tree.write(args.ui, encoding='utf-8', xml_declaration=True)
    print(f"Updated {args.ui}")


if __name__ == '__main__':
    main()

