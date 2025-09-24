#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import xml.etree.ElementTree as ET


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-format XML in-place with newlines and two-space indents.
    - Places each child element on its own line
    - Keeps leaf <text>...</text> elements on a single line
    - Ensures container elements like <lang code="..."> start a new line for children
    """
    i = "\n" + "  " * level
    # If element has children, we place them on their own lines
    if len(elem):
        if not (elem.text or '').strip():
            elem.text = i + "  "
        for idx, child in enumerate(list(elem)):
            indent(child, level + 1)
            # tail after each child
            if not (child.tail or '').strip():
                child.tail = i + "  " if idx < len(elem) - 1 else i
    else:
        # Leaf node: keep text content inline; normalize tail for closing tag placement
        if elem.text is not None:
            elem.text = elem.text.strip()
        if not (elem.tail or '').strip():
            elem.tail = "\n" + "  " * level if level else "\n"


def reformat(path: str, inplace: bool = True, output: str | None = None) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    tree = ET.parse(path)
    root = tree.getroot()
    indent(root, 0)
    target = path if inplace and not output else (output or path)
    tree.write(target, encoding='utf-8', xml_declaration=True)
    return target


def main():
    ap = argparse.ArgumentParser(description='Reformat ui_texts_translations.xml with one <text> per line and proper <lang> line breaks')
    ap.add_argument('--path', default=os.path.join('settings', 'ui_texts_translations.xml'), help='Path to ui_texts_translations.xml')
    ap.add_argument('--output', default=None, help='Write to this file instead of in-place')
    args = ap.parse_args()
    out = reformat(args.path, inplace=(args.output is None), output=args.output)
    print(f"Reformatted: {out}")


if __name__ == '__main__':
    main()

