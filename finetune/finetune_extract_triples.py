#!/usr/bin/env python

"""Extract triples + surface forms from WebNLG 2023 XML.
Usage example (English):
    python finetune_extract_triples.py --lang en --input en_train.xml --output en_train.jsonl
"""

import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--lang", default="en", help="Target language prefix, e.g. en or ga")
    pa.add_argument("--input", required=True, help="WebNLG *_train.xml file")
    pa.add_argument("--output", required=True, help="Destination JSONL")
    args = pa.parse_args()

    root = ET.parse(args.input).getroot()
    out: List[Dict] = []
    want_en = args.lang.lower().startswith("en")

    for entry in root.iter("entry"):
        tripleset = entry.find("modifiedtripleset") or entry.find("originaltripleset")
        if tripleset is None:
            continue
        triples = [t.text.strip() for t in tripleset]
        for lex in entry.findall("lex"):
            tag = (lex.get("lang") or "en").lower()  # blank ⇒ English
            if want_en and not tag.startswith("en"):
                continue
            if not want_en and not tag.startswith("ga"):
                continue
            text = " ".join((lex.text or "").split())
            out.append({"triples": triples, "text": text})

    Path(args.output).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), "utf-8")
    print(f"[extract] Saved {len(out)} {args.lang.upper()} examples → {args.output}")

if __name__ == "__main__":
    main()