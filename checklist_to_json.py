#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a text/markdown checklist into the canonical JSON structure.

Supported patterns:
- Headings:  '# Title', '## Title', '### Title'
- Headings:  'Title:' or '**Title:**'
- Bullets :  '- question', '* question', '1. question', '1) question'
- Optional "Objective:" at the top (first block before any heading) or via --objective flag.

Usage:
  python tools/checklist_to_json.py INPUT.md -o documents/compliance_checklist.json
  python tools/checklist_to_json.py INPUT.txt --objective "Verification of regulatory coverage" -o out.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# --------- Heuristics / Regex ---------

RE_MD_HEADING = re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*$')
RE_COLON_HEADING = re.compile(r'^\s{0,3}(?:\*\*)?(.+?)(?:\*\*)?\s*:\s*$')   # 'Title:' or '**Title:**'
RE_BULLET = re.compile(r'^\s{0,3}([-*])\s+(.*\S)\s*$')
RE_NUM_BULLET = re.compile(r'^\s{0,3}(\d+[\.\)])\s+(.*\S)\s*$')
RE_EMPTY = re.compile(r'^\s*$')

def normalize_text(s: str) -> str:
    s = s.replace('\t', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def collapse_md_bold_italics(s: str) -> str:
    # remove simple **bold** or *italics* markers
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'\*(.+?)\*', r'\1', s)
    return s.strip()

# --------- Parser ---------

def parse_checklist_lines(lines: List[str]) -> Dict:
    """
    Parse lines into:
      {"objective": str (optional),
       "items": [{"id":"C1","title":"...","questions":[...]}]}
    """
    objective_chunks: List[str] = []
    sections: List[Dict] = []

    current_title: Optional[str] = None
    current_questions: List[str] = []
    seen_first_heading = False

    def flush_section():
        nonlocal current_title, current_questions
        if current_title is not None:
            title = normalize_text(current_title)
            questions = [normalize_text(q) for q in current_questions if normalize_text(q)]
            sections.append({"title": title, "questions": questions})
        current_title, current_questions = None, []

    for raw in lines:
        line = raw.rstrip('\n')

        # Skip pure empties and collect objective text before first heading
        if RE_EMPTY.match(line):
            continue

        # Heading patterns
        m = RE_MD_HEADING.match(line)
        if m:
            # New section
            if not seen_first_heading and objective_chunks:
                seen_first_heading = True
            flush_section()
            current_title = collapse_md_bold_italics(m.group(2))
            seen_first_heading = True
            continue

        m = RE_COLON_HEADING.match(line)
        if m:
            flush_section()
            current_title = collapse_md_bold_italics(m.group(1))
            seen_first_heading = True
            continue

        # Bullets under a section
        m = RE_BULLET.match(line)
        if m and current_title is not None:
            q = collapse_md_bold_italics(m.group(2))
            current_questions.append(q)
            continue

        m = RE_NUM_BULLET.match(line)
        if m and current_title is not None:
            q = collapse_md_bold_italics(m.group(2))
            current_questions.append(q)
            continue

        # If no heading encountered yet → treat as objective prose
        if not seen_first_heading:
            objective_chunks.append(line)
        else:
            # Non-bullet, non-heading line *within a section*:
            # treat as a question if it's sentence-like and not too long
            if current_title is not None:
                txt = collapse_md_bold_italics(line).strip()
                if txt:
                    current_questions.append(txt)

    # Final flush
    flush_section()

    # Canonicalize item IDs: C1..Ck
    items = []
    for i, sec in enumerate(sections, start=1):
        items.append({
            "id": f"C{i}",
            "title": sec["title"],
            "questions": sec["questions"] or []
        })

    # Objective
    objective = None
    if objective_chunks:
        obj = normalize_text(" ".join(objective_chunks))
        if obj:
            # Remove leading "Objective:" if present
            obj = re.sub(r'^\s*Objective\s*:\s*', '', obj, flags=re.IGNORECASE)
            objective = obj

    out = {"items": items}
    if objective:
        out["objective"] = objective
    return out

# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Convert a text/markdown checklist into canonical JSON.")
    ap.add_argument("input", help="Path to .txt/.md checklist")
    ap.add_argument("-o", "--output", default="documents/compliance_checklist.json", help="Output JSON path")
    ap.add_argument("--objective", help="Override/force objective text (optional)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    text = src.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data = parse_checklist_lines(lines)

    # Allow explicit objective override
    if args.objective:
        data["objective"] = args.objective.strip()

    # Minimal validation: at least one item with title
    if not data.get("items"):
        raise SystemExit("No sections detected. Check heading formatting (e.g., '## Title' or 'Title:').")

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} with {len(data['items'])} items.")

if __name__ == "__main__":
    main()
