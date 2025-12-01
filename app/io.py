# app/io.py
# -*- coding: utf-8 -*-
"""
Ingestion module:
- Parse PDFs (Docling preferred; PyMuPDF fallback; OCR optional)
- Clean & normalize text
- Detect sections (Docling structure or regex heuristic)
- Chunk sections into ~1.6k-char windows with 15% overlap
- Write parsed sections & chunks into the active run directory

Config keys expected (see config.yaml example):
paths:
  documents: documents/PDFs
  data_dir: data
ingest:
  engine: docling | pymupdf | pdfminer
  include_tables: true/false
  include_captions: true/false
  enable_ocr: true/false
  max_chars_per_chunk: 1600
  overlap_ratio: 0.15

"""

from __future__ import annotations

import os
import re
import json
import time
import glob
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple

# Optional dependencies are imported lazily
# Docling (preferred)
try:
    from docling.document_converter import DocumentConverter  # type: ignore
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False

# PyMuPDF (fallback)
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

# OCR (optional)
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# Token estimation (optional but recommended)
try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
    _ENC = None  # lazy init
except Exception:
    _HAS_TIKTOKEN = False
    _ENC = None


# --------------------------- Data Models ---------------------------
@dataclass
class Section:
    section_id: str
    title: str
    page_start: int
    page_end: int
    text: str
    # NEW: absolute positions in the full concatenated text
    char_start: int = 0
    char_end: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    text: str
    token_estimate: int


# --------------------------- Utilities ---------------------------

def _stable_id(*parts: str, maxlen: int = 32) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:maxlen]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Rough token estimate using tiktoken if available, else 1 token ≈ 4 chars."""
    if _HAS_TIKTOKEN:
        global _ENC
        if _ENC is None:
            # Fall back to cl100k_base if model not known
            try:
                _ENC = tiktoken.encoding_for_model(model)
            except Exception:
                _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    # Heuristic fallback
    return max(1, int(len(text) / 4))

_DOTTED = re.compile(r"^\d+(\.\d+)*$")

def _id_from_token_or_counter(token: str, counter: int) -> str:
    token = (token or "").strip().rstrip(".)")
    return token if token and _DOTTED.match(token) else str(counter)

_HDR_FOOTER_CACHE: Dict[str, str] = {}


def normalize_text(text: str) -> str:
    """Normalize extracted text: fix hyphenation, collapse whitespace, keep bullets/numbering."""
    # Remove obvious headers/footers patterns across pages if present (lightweight heuristic)
    # (Real header/footer stripping should be per-page with more context; here we do minimal cleaning.)
    text = text.replace("\r", "\n")
    # Fix hyphenation at line breaks: word-\nword -> wordword
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Join broken lines where appropriate
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# add this small helper near the top with other utils (optional)
def _has_page_blocks(doc) -> bool:
    """
    Returns True if doc.pages exists and first item has a 'blocks' attribute.
    Some Docling versions expose pages as page numbers (ints), not objects.
    """
    try:
        pages = getattr(doc, "pages", None)
        if not pages:
            return False
        # Pages could be a list-like; check first non-empty
        first = None
        for p in pages:
            first = p
            break
        return hasattr(first, "blocks")
    except Exception:
        return False

def _pymupdf_page_texts(pdf_path: str) -> List[str]:
    """Return list of per-page plain texts using PyMuPDF, no OCR here."""
    if not _HAS_PYMUPDF:
        return []
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        txt = page.get_text("text") or ""
        texts.append(normalize_text(txt))
    return texts

import bisect

def _charpos_to_page(page_map: List[Tuple[int, int]], char_pos: int) -> int:
    """
    Map a character offset in full_text to a 1-based page number using page_map.
    page_map is [(page_num, start_offset)], sorted by start_offset ascending.
    """
    if not page_map:
        return 1
    # build starts array
    starts = [s for (_, s) in page_map]
    idx = bisect.bisect_right(starts, max(0, char_pos)) - 1
    if idx < 0:
        idx = 0
    if idx >= len(page_map):
        idx = len(page_map) - 1
    return page_map[idx][0]


import bisect

def _charpos_to_page(page_map, char_pos: int) -> int:
    """Map a global char offset to 1-based page using page_map=[(page_num, start_offset), ...]."""
    if not page_map:
        return 1
    starts = [s for (_, s) in page_map]
    i = bisect.bisect_right(starts, max(0, char_pos)) - 1
    if i < 0: i = 0
    if i >= len(page_map): i = len(page_map) - 1
    return page_map[i][0]

def _proportional_page_map(full_text: str, pymupdf_pages: List[str]) -> List[Tuple[int, int]]:
    """
    Create a page_map = [(page_number, start_char_offset_in_full_text), ...]
    by proportionally allocating the concatenated Docling full_text across the
    number of PyMuPDF pages using their text lengths as weights.
    """
    n_pages = len(pymupdf_pages)
    if n_pages == 0:
        return [(1, 0)]

    total_docling = max(1, len(full_text))
    lens = [max(1, len(t)) for t in pymupdf_pages]
    total_weight = float(sum(lens))

    # cumulative starts
    starts = [0]
    acc = 0.0
    for w in lens[:-1]:  # last page start = computed; last page implicitly runs to end
        acc += (w / total_weight) * total_docling
        starts.append(int(round(acc)))

    # Ensure monotonic and within bounds
    starts = [min(max(0, s), total_docling - 1) for s in starts]
    # Make strictly non-decreasing
    for i in range(1, len(starts)):
        if starts[i] < starts[i-1]:
            starts[i] = starts[i-1]

    page_map = [(i + 1, starts[i]) for i in range(n_pages)]
    return page_map

_HEADING_RE_NUM = re.compile(r"^(?P<num>\d+(\.\d+)*[\.\)]?)\s+(?P<title>[A-Z][^\n]{1,80})$")
_HEADING_RE_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 \-/&]{6,80}$")

def _is_heading(line: str) -> Optional[Tuple[str,str]]:
    s = line.strip()
    if not s or s.endswith("."):  # don’t accept sentences
        return None
    m = _HEADING_RE_NUM.match(s)
    if m:
        return (m.group("num").rstrip(".)"), m.group("title").strip())
    # ALL-CAPS style
    if _HEADING_RE_ALLCAPS.match(s):
        return ("", s)
    return None


import re
from typing import Optional, Tuple, List

# Markdown-style headings (if Docling exported MD)

_MD_H = re.compile(r"^(#{1,6})\s+(?P<title>.+)$")

# Numbered headings: "2", "2.1", "2.1.3)", etc.
_NUM_H = re.compile(r"^(?P<num>\d+(\.\d+)*[\.\)]?)\s+(?P<title>[A-Z][^\n]{1,100})$")

# Roman numeral headings: "IV.", "V)", etc.
_ROMAN_H = re.compile(r"^(?P<num>[IVXLCM]+[\.\)])\s+(?P<title>[A-Z][^\n]{1,100})$", re.IGNORECASE)

# Loud ALL-CAPS heading (≥2 words; avoid single codes)
_ALLCAPS_H = re.compile(r"^[A-Z][A-Z0-9 \-/,&]{6,}$")

# Appendix headings
_APPX_H = re.compile(r"^(Appendix|ANNEX)\s+([A-Z0-9]+)\b[:\.\-\s]*(?P<title>[A-Z].{0,100})$", re.IGNORECASE)

# Canonical FDA SSED anchors (case-insensitive exact line match)
_FDA_ANCHORS = {
    "SUMMARY OF SAFETY AND EFFECTIVENESS DATA",
    "INDICATIONS FOR USE",
    "DEVICE DESCRIPTION",
    "ALTERNATIVES",
    "NONCLINICAL LABORATORY STUDIES",
    "CLINICAL STUDIES",
    "RISK/BENEFIT ANALYSIS",
    "CONTRAINDICATIONS",
    "WARNINGS",
    "ADVERSE EVENTS",
    "STERILIZATION",
    "PATIENT LABELING",
    "CONCLUSIONS",
    "LABELING",
    "MANUFACTURING",
}

# Lines that should never be headings (IDs/dates/codes)
_IGNORE = [
    re.compile(r"^P\d{6}$", re.IGNORECASE),
    re.compile(r"^K\d{6}$", re.IGNORECASE),
    re.compile(r"^[A-Z]\d{5,}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]

def _is_probable_heading(line: str, prev_blank: bool, next_blank: bool) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s: return None
    if s.endswith("."):  # sentences: reject
        return None
    for pat in _IGNORE:
        if pat.match(s):
            return None

    # 1) Markdown-style "# ..." headings
    m = _MD_H.match(s)
    if m:
        return ("", m.group("title").strip())


    # 2) Numbered outline
    m = _NUM_H.match(s)
    if m:
        return (m.group("num").rstrip(".)"), m.group("title").strip())

    # 3) Roman numerals
    m = _ROMAN_H.match(s)
    if m:
        return (m.group("num").rstrip(".)").upper(), m.group("title").strip())

    # 4) Appendix
    m = _APPX_H.match(s)
    if m:
        return ("APP", s)

    # 5) ALL-CAPS line with blank lines around (visual block heading)
    if _ALLCAPS_H.match(s) and prev_blank and next_blank and len(s.split()) >= 2:
        return ("", s)

    # 6) FDA canonical anchors (exact line match ignoring case)
    if s.upper() in _FDA_ANCHORS:
        return ("", s.title())

    return None



def _heuristic_sections_from_text(full_text: str, page_map: List[Tuple[int, int]]) -> List[Section]:
    # split into lines + track char offsets for each line start
    lines = full_text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # +1 for the newline that was split

    # detect candidate headings
    # expect _is_probable_heading(line, prev_blank, next_blank) -> (num, title) or None
    candidates: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_blank = (i == len(lines) - 1) or (lines[i + 1].strip() == "")
        hit = _is_probable_heading(line, prev_blank, next_blank)
        if hit:
            num, title = hit
            candidates.append((offsets[i], num or "", title))

    # fallback: no headings → single "Document" section
    if not candidates:
        s0, e0 = 0, len(full_text)
        p_start = _charpos_to_page(page_map, s0)
        p_end = _charpos_to_page(page_map, e0 - 1 if e0 > 0 else 0)
        return [Section(
            section_id="1",
            title="Document",
            page_start=p_start,
            page_end=p_end,
            text=full_text.strip(),
            char_start=s0,
            char_end=e0,
        )]

    # simplest ID strategy: flat counter by appearance
    simple_cands: List[Tuple[int, str, str]] = []
    for idx, (off, _tok, title) in enumerate(candidates, start=1):
        simple_cands.append((off, str(idx), title))

    # slice sections using simple_cands
    sections: List[Section] = []
    for i, (start_off, sec_id, title) in enumerate(simple_cands):
        start = start_off
        end = simple_cands[i + 1][0] if i + 1 < len(simple_cands) else len(full_text)
        sect_text = full_text[start:end].strip()
        if not sect_text:
            continue
        p_start = _charpos_to_page(page_map, start)
        p_end = _charpos_to_page(page_map, max(start, end - 1))
        sections.append(Section(
            section_id=sec_id,
            title=title,
            page_start=p_start,
            page_end=p_end,
            text=sect_text,
            char_start=start,
            char_end=end,
        ))

    return sections

def _chunk_text(
    text: str,
    max_chars: int,
    overlap_ratio: float,
) -> List[Tuple[int, int]]:
    """Return list of (char_start, char_end) windows with overlap."""
    n = len(text)
    if n <= max_chars:
        return [(0, n)]
    stride = int(max_chars * (1 - overlap_ratio))
    stride = max(1, stride)
    windows = []
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        windows.append((start, end))
        if end == n:
            break
        start += stride
    return windows


# --------------------------- Doc Parsers ---------------------------


def _parse_with_pymupdf(pdf_path: str, enable_ocr: bool = False) -> Tuple[str, List[Tuple[int, int]]]:
    """
    PyMuPDF fallback: extract per-page text; if OCR enabled and page text is very short, rasterize + OCR.
    Returns (full_text, page_map).
    """
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF is not installed. Please install 'pymupdf' or use Docling.")
    doc = fitz.open(pdf_path)
    pages = []
    page_map: List[Tuple[int, int]] = []
    cursor = 0

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if enable_ocr and (not text or len(text.strip()) < 50):
            if not _HAS_TESSERACT:
                logging.warning("OCR requested but pytesseract/Pillow not installed; skipping OCR for page %d.", i + 1)
            else:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img) or ""
                text = ocr_text if len(ocr_text.strip()) > len(text.strip()) else text

        text = normalize_text(text)
        page_map.append((i + 1, cursor))
        pages.append(text)
        cursor += len(text) + 1

    full_text = "\n".join(pages).strip()
    return full_text, page_map



# replace your _parse_with_docling() with  robust version
def _parse_with_docling(pdf_path: str, include_tables: bool, include_captions: bool) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Returns (full_text, page_map) where page_map = list of (page_number, start_char_offset_in_full_text).
    Primary path: iterate page.blocks if available.
    Fallback path: export full document to markdown/text and return a single-page map.
    """
    if not _HAS_DOCLING:
        raise RuntimeError("Docling is not installed. Please set ingest.engine to 'pymupdf' or install docling.")

    conv = DocumentConverter()
    result = conv.convert(pdf_path)
    doc = result.document

    # ---- Primary path: page -> blocks (when available) ----
    if _has_page_blocks(doc):
        pages_text: List[str] = []
        page_map: List[Tuple[int, int]] = []
        cursor = 0

        for i, page in enumerate(doc.pages):
            page_text_parts = []
            # Iterate blocks defensively
            for block in getattr(page, "blocks", []) or []:
                cat = getattr(block, "category", None)
                if cat == "Text":
                    page_text_parts.append(getattr(block, "text", "") or "")
                elif include_tables and cat == "Table":
                    # Try to linearize; fall back to any text representation available
                    table_md = getattr(block, "as_markdown", None)
                    if callable(table_md):
                        try:
                            table_text = table_md()  # newer APIs may require call
                        except TypeError:
                            table_text = block.as_markdown  # property form
                    else:
                        table_text = getattr(block, "text", "") or ""
                    if table_text:
                        page_text_parts.append("\n[Table]\n" + str(table_text).strip())
                elif include_captions and cat in ("Figure", "Image"):
                    cap = getattr(block, "caption", None)
                    if cap:
                        page_text_parts.append(f"[Caption] {cap}")

            joined = "\n".join([t for t in page_text_parts if t and str(t).strip()])
            joined = normalize_text(joined)
            page_map.append((i + 1, cursor))
            pages_text.append(joined)
            cursor += len(joined) + 1  # newline join

        full_text = "\n".join(pages_text).strip()
        return full_text, page_map

    # ---- Fallback path: export whole document ----
    # Try markdown, then text; some versions expose different export methods.
    exported = None
    for attr in ("export_to_markdown", "export_markdown", "export_to_text", "export_text"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                exported = fn()
                break
            except Exception:
                continue

    # Last-resort: string conversion
    if exported is None:
        exported = str(doc)

    full_text = normalize_text(str(exported or ""))

    # Coarse page_map: single page starting at 0 if we cannot access per-page blocks
    page_map = [(1, 0)]
    return full_text, page_map









def _parse_with_hybrid(pdf_path: str, include_tables: bool, include_captions: bool) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Hybrid strategy:
      1) Try Docling with page blocks → precise per-page map.
      2) Else export Docling full text (markdown/text) → build page_map
         using PyMuPDF per-page lengths (proportional mapping).
    """
    if not _HAS_DOCLING:
        # If Docling missing, fall back directly to PyMuPDF path
        return _parse_with_pymupdf(pdf_path, enable_ocr=False)

    conv = DocumentConverter()
    result = conv.convert(pdf_path)
    doc = result.document

    # Primary: Docling with page.blocks
    if _has_page_blocks(doc):
        pages_text = []
        page_map = []
        cursor = 0
        for i, page in enumerate(doc.pages):
            parts = []
            for block in getattr(page, "blocks", []) or []:
                cat = getattr(block, "category", None)
                if cat == "Text":
                    parts.append(getattr(block, "text", "") or "")
                elif include_tables and cat == "Table":
                    md_attr = getattr(block, "as_markdown", None)
                    if callable(md_attr):
                        try:
                            table_text = md_attr()
                        except TypeError:
                            table_text = block.as_markdown
                    else:
                        table_text = getattr(block, "text", "") or ""
                    if table_text:
                        parts.append("\n[Table]\n" + str(table_text).strip())
                elif include_captions and cat in ("Figure", "Image"):
                    cap = getattr(block, "caption", None)
                    if cap:
                        parts.append(f"[Caption] {cap}")
            joined = normalize_text("\n".join([t for t in parts if t and str(t).strip()]))
            page_map.append((i + 1, cursor))
            pages_text.append(joined)
            cursor += len(joined) + 1
        full_text = "\n".join(pages_text).strip()
        return full_text, page_map

    # Fallback: Docling export + PyMuPDF page mapping
    exported = None
    for attr in ("export_to_markdown", "export_markdown", "export_to_text", "export_text"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                exported = fn()
                break
            except Exception:
                continue
    if exported is None:
        exported = str(doc)
    full_text = normalize_text(str(exported or ""))

    # Build proportional map from PyMuPDF per-page text
    pymupdf_pages = _pymupdf_page_texts(pdf_path)
    page_map = _proportional_page_map(full_text, pymupdf_pages)
    return full_text, page_map

# --------------------------- Public Ingestor ---------------------------

class Ingestor:
    """
    High-level ingestion orchestrator used by CLI.

    Usage:
        ing = Ingestor(config, run_dir="data/runs/run_YYYYMMDD_HHMM")
        outputs = ing.run_all()
    """

    def __init__(self, config: Dict, run_dir: str):
        self.cfg = config
        self.run_dir = run_dir

        self.docs_dir = os.path.abspath(self.cfg["paths"]["documents"])
        self.parsed_dir = os.path.join(self.run_dir, "parsed")
        self.chunks_dir = os.path.join(self.run_dir, "chunks")
        self.logs_dir = os.path.join(self.run_dir, "logs")

        for d in (self.parsed_dir, self.chunks_dir, self.logs_dir):
            _ensure_dir(d)

        ingest_cfg = self.cfg.get("ingest", {})
        self.engine = ingest_cfg.get("engine", "docling").lower()
        self.include_tables = bool(ingest_cfg.get("include_tables", True))
        self.include_captions = bool(ingest_cfg.get("include_captions", True))
        self.enable_ocr = bool(ingest_cfg.get("enable_ocr", False))
        self.max_chars = int(ingest_cfg.get("max_chars_per_chunk", 1600))
        self.overlap = float(ingest_cfg.get("overlap_ratio", 0.15))

        # Token model for estimates
        self.token_model = self.cfg.get("models", {}).get("completion", "gpt-4o-mini")

        # Logger
        self.logger = logging.getLogger("ingest")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

        # File log
        flog_path = os.path.join(self.logs_dir, "ingest.log")
        fh = logging.FileHandler(flog_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

    # ----------- Main entrypoint -----------

    def run_all(self) -> Dict[str, Dict[str, str]]:
        """
        Parse all PDFs under documents dir, produce sections + chunks files.
        Returns a dict {doc_id: {"sections": path, "chunks": path}}.
        """
        pdf_paths = sorted(glob.glob(os.path.join(self.docs_dir, "*.pdf")))
        if not pdf_paths:
            self.logger.warning("No PDFs found in %s", self.docs_dir)

        outputs: Dict[str, Dict[str, str]] = {}

        for pdf in pdf_paths:
            doc_id = os.path.splitext(os.path.basename(pdf))[0]
            self.logger.info("Ingesting %s", doc_id)

            try:
                sections = self._parse_and_section(pdf)
                sections_path = os.path.join(self.parsed_dir, f"{doc_id}_sections.json")
                with open(sections_path, "w", encoding="utf-8") as f:
                    json.dump({"doc_id": doc_id, "sections": [asdict(s) for s in sections]}, f, ensure_ascii=False, indent=2)

                chunks = self._chunk_sections(doc_id, sections)
                chunks_path = os.path.join(self.chunks_dir, f"{doc_id}_chunks.jsonl")
                with open(chunks_path, "w", encoding="utf-8") as f:
                    for ch in chunks:
                        f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

                outputs[doc_id] = {"sections": sections_path, "chunks": chunks_path}
                self.logger.info("Ingested %s: %d sections, %d chunks", doc_id, len(sections), len(chunks))
            except Exception as e:
                self.logger.exception("Failed to ingest %s: %s", pdf, e)

        return outputs

    # ----------- Parsing & Sectioning -----------

    def _parse_and_section(self, pdf_path: str) -> List[Section]:
        t0 = time.time()
        engine = self.engine
        if engine == "hybrid":
            full_text, page_map = _parse_with_hybrid(pdf_path, self.include_tables, self.include_captions)
        elif engine == "docling":
            full_text, page_map = _parse_with_docling(pdf_path, self.include_tables, self.include_captions)
        elif engine == "pymupdf":
            full_text, page_map = _parse_with_pymupdf(pdf_path, enable_ocr=self.enable_ocr)
        else:
            full_text, page_map = _parse_with_hybrid(pdf_path, self.include_tables, self.include_captions)
        self._page_map = page_map
        self.logger.info("Parsed %s in %.2fs (%d chars)", os.path.basename(pdf_path), time.time() - t0, len(full_text))
        sections = _heuristic_sections_from_text(full_text, page_map)
        cleaned_sections: List[Section] = []
        for s in sections:
            txt = (s.text or "").strip()
            if not txt:
                continue
            cleaned_sections.append(Section(
                section_id=s.section_id or "1",
                title=(s.title or "").strip() or "Section",
                page_start=s.page_start,
                page_end=s.page_end,
                text=txt,
                # PRESERVE global offsets for correct per-chunk page mapping
                char_start=s.char_start,
                char_end=s.char_end,
            ))
        return cleaned_sections

    # ----------- Chunking -----------

    def _chunk_sections(self, doc_id: str, sections: List[Section]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for s in sections:
            windows = _chunk_text(s.text, self.max_chars, self.overlap)
            for idx, (a, b) in enumerate(windows):
                sub = s.text[a:b]
                token_est = _estimate_tokens(sub, model=self.token_model)
                chunk_id = _stable_id(doc_id, s.section_id, str(idx))

                # map section-relative chars -> global char positions
                global_start = s.char_start + a
                global_end   = s.char_start + b

                p_start = _charpos_to_page(self._page_map, global_start)
                p_end   = _charpos_to_page(self._page_map, max(global_start, global_end - 1))

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_id=s.section_id,
                    section_title=s.title,
                    page_start=p_start,
                    page_end=p_end,
                    char_start=a,           # keep section-relative for readability
                    char_end=b,
                    text=sub,
                    token_estimate=token_est
                ))
        return chunks
