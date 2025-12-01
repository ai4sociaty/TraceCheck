# app/ingest_viz.py
# -*- coding: utf-8 -*-
"""
Enhanced ingestion visualization:
- Structure: section treemap, chunk density per page, chunk length & overlap histograms
- Linguistic: sentence length, keyword freq, (optional) spaCy NER, (optional) readability
- Semantic: 2D map of chunk embeddings, section cosine heatmap, outlier scores
- Regulatory/Domain: modal verbs, regulatory term counts, citation density
- Quality table: per-section KPIs + flags, exported to findings JSON

Outputs (unchanged):
  viz/ingest_dashboard.html
  viz/ingest_summary.md
  viz/ingest_findings.json

Offline by default: embeds Plotly JS inside each figure (no CDN).
Falls back to Matplotlib PNGs if Plotly is unavailable.
Skips optional panels gracefully when deps are missing.
"""

from __future__ import annotations

import os
import io
import re
import json
import math
import base64
import traceback
from dataclasses import dataclass, fields as dc_fields
from typing import List, Dict, Tuple, Optional, Any, Iterable
from collections import Counter, defaultdict

# ------------------------- Optional imports (guarded) -------------------------

def _safe_imports():
    mods = {}
    flags = {}

    # core viz libs
    try:
        import plotly
        from plotly.offline import plot as plotly_offline_plot
        import plotly.graph_objs as go
        flags["has_plotly"] = True
        mods["plotly_offline_plot"] = plotly_offline_plot
        mods["plotly_go"] = go
    except Exception:
        flags["has_plotly"] = False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        flags["has_mpl"] = True
        mods["mpl_plt"] = plt
    except Exception:
        flags["has_mpl"] = False

    # data
    try:
        import numpy as np
        flags["has_np"] = True
        mods["np"] = np
    except Exception:
        flags["has_np"] = False
    try:
        import pandas as pd
        flags["has_pd"] = True
        mods["pd"] = pd
    except Exception:
        flags["has_pd"] = False

    # nlp optional
    try:
        import spacy
        flags["has_spacy"] = True
        mods["spacy"] = spacy
    except Exception:
        flags["has_spacy"] = False
    try:
        import textstat
        flags["has_textstat"] = True
        mods["textstat"] = textstat
    except Exception:
        flags["has_textstat"] = False

    # embeddings + dim red + ml
    try:
        from sentence_transformers import SentenceTransformer
        flags["has_sbert"] = True
        mods["SentenceTransformer"] = SentenceTransformer
    except Exception:
        flags["has_sbert"] = False
    try:
        import umap  # noqa
        from umap import UMAP
        flags["has_umap"] = True
        mods["UMAP"] = UMAP
    except Exception:
        flags["has_umap"] = False
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.ensemble import IsolationForest
        flags["has_sklearn"] = True
        mods["PCA"] = PCA
        mods["cosine_similarity"] = cosine_similarity
        mods["IsolationForest"] = IsolationForest
    except Exception:
        flags["has_sklearn"] = False

    return mods, flags

_MODS, _FLAGS = _safe_imports()

# ------------------------- Data models -------------------------

@dataclass
class Section:
    section_id: str
    title: str
    page_start: int
    page_end: int
    text: str
    # tolerate newer schema carrying global char positions
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
    token_estimate: int = 0

# ------------------------- IO helpers -------------------------

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------- Small text utils -------------------------

_STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or than that the their this to was were will with without within
we they you he she i our your shall must should may can could would
""".split())

_REG_TERMS = [
    "adverse", "contraindication", "warning", "precaution", "clinical", "nonclinical",
    "biocompatibility", "sterilization", "labeling", "risk", "benefit", "trial",
    "complication", "rupture", "leak", "hazard", "mitigation", "validation", "verification"
]

_CITATION_PATTERNS = [
    r"\bISO\b", r"\bIEC\b", r"\bFDA\b", r"\b21\s*CFR\b", r"\bASTM\b", r"\bEN\s?\d+",
]

def _simple_tokens(text: str) -> List[str]:
    # letters only ≥3 chars, lowercased
    return [t for t in re.findall(r"[A-Za-z]{3,}", text.lower()) if t not in _STOPWORDS]

def _split_sentences(text: str) -> List[str]:
    # crude but robust: split on ., !, ? followed by space+capital or EOL
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[])', text.strip())
    return [s.strip() for s in parts if s.strip()]

def _pct(n: float) -> str:
    return f"{100.0*n:.1f}%"

# ------------------------- Plot helpers -------------------------

def _plotly_div(fig) -> str:
    if not _FLAGS.get("has_plotly"):
        return ""
    plot = _MODS["plotly_offline_plot"]
    return plot(fig, include_plotlyjs=True, output_type="div")

def _mpl_png_uri(draw_fn) -> str:
    if not _FLAGS.get("has_mpl"):
        return ""
    plt = _MODS["mpl_plt"]
    fig = plt.figure(figsize=(6, 4), dpi=150)
    try:
        ax = fig.add_subplot(111)
        draw_fn(fig, ax)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        plt.close(fig)
        return ""

def _html_card(title: str, body_html: str) -> str:
    return f"""
    <div class="card">
      <h2>{title}</h2>
      <div class="card-body">{body_html}</div>
    </div>
    """

def _html_notice(msg: str) -> str:
    return f'<p><i>{msg}</i></p>'


# --- Heuristics for figures/tables and caption stripping ---

# counts: detect common patterns and docling linearization tags
_FIG_PATTERNS = [
    r"\[FIGURE[^\]]*\]",         # e.g., [FIGURE 3] ...
    r"^figure\s+\d+[\.:]",       # Figure 3:
    r"\bfig\.\s*\d+\b",          # Fig. 3
    r"\bfigure\s+\d+\b",         # Figure 3
]
_TAB_PATTERNS = [
    r"\[TABLE[^\]]*\]",
    r"^table\s+\d+[\.:]",
    r"\btab\.\s*\d+\b",
    r"\btable\s+\d+\b",
]

_FIG_RE = re.compile("|".join(_FIG_PATTERNS), re.IGNORECASE | re.MULTILINE)
_TAB_RE = re.compile("|".join(_TAB_PATTERNS), re.IGNORECASE | re.MULTILINE)

_CAPTION_LINE_RE = re.compile(
    r"^\s*(\[FIGURE[^\]]*\]|Figure\s+\d+[\.:]|Fig\.\s*\d+[\.:]|Table\s+\d+[\.:]|Tab\.\s*\d+[\.:]).*$",
    re.IGNORECASE | re.MULTILINE
)

def _count_fig_tab(text: str) -> Tuple[int, int]:
    if not text:
        return 0, 0
    figs = len(_FIG_RE.findall(text))
    tabs = len(_TAB_RE.findall(text))
    return figs, tabs

def _strip_caption_lines(text: str) -> str:
    # remove lines that look like figure/table captions
    return _CAPTION_LINE_RE.sub("", text or "")

# ------------------------- Core visualizer -------------------------

class IngestVisualizer:
    def __init__(
        self,
        run_dir: str,
        doc_id: str,
        overlap_tolerance: float = 0.05,
        low_char_threshold: int = 100,
        tiny_chunk_threshold: int = 400,
        show_tables: bool = True, show_captions: bool = True,
        exclude_captions_in_keywords: bool = True) -> None:
        self.run_dir = os.path.abspath(run_dir)
        self.doc_id = doc_id
        self.viz_dir = os.path.join(self.run_dir, "viz")
        _ensure_dir(self.viz_dir)

        # inputs
        self.sections_path = os.path.join(self.run_dir, "parsed", f"{doc_id}_sections.json")
        self.chunks_path = os.path.join(self.run_dir, "chunks", f"{doc_id}_chunks.jsonl")
        if not os.path.exists(self.sections_path):
            raise FileNotFoundError(f"Sections not found: {self.sections_path}")
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks not found: {self.chunks_path}")

        # load sections
        sections_json = _read_json(self.sections_path)
        allowed = {f.name for f in dc_fields(Section)}
        self.sections: List[Section] = [
            Section(**{k: v for k, v in s.items() if k in allowed})
            for s in sections_json.get("sections", [])
        ]

        # load chunks
        chunks_jsonl = _read_jsonl(self.chunks_path)
        allowed_c = {f.name for f in dc_fields(Chunk)}
        # tolerate missing fields with defaults
        self.chunks: List[Chunk] = []
        for c in chunks_jsonl:
            c = {**{"token_estimate": 0}, **c}
            self.chunks.append(Chunk(**{k: v for k, v in c.items() if k in allowed_c}))

        # params
        self.overlap_tol = overlap_tolerance
        self.low_char_threshold = low_char_threshold
        self.tiny_chunk_threshold = tiny_chunk_threshold
        self.show_tables = show_tables
        self.show_captions = show_captions

        # derived
        self.page_max = max([s.page_end for s in self.sections], default=1)
        self._mods = _MODS
        self._flags = _FLAGS
        self.exclude_captions_in_keywords = exclude_captions_in_keywords

        # --- add these two lines at the very end of __init__ ---
        self._skip_figtab = getattr(self, "_skip_figtab", False)
        self._skip_ocr = getattr(self, "_skip_ocr", False)

    # --------------------- Builders for each panel ---------------------

    def _panel_structure(self) -> str:
        # Section treemap
        length_by_sec = []
        for s in self.sections:
            length_by_sec.append((s.section_id, s.title, len(s.text), f"{s.page_start}-{s.page_end}"))

        divs = []

        if self._flags.get("has_plotly") and length_by_sec:
            go = self._mods["plotly_go"]
            labels = [f"{sid} {title}" for sid, title, _, _ in length_by_sec]
            values = [v for _, _, v, _ in length_by_sec]
            hover = [f"Pages {pg} · {v} chars" for _, _, v, pg in length_by_sec]
            fig = go.Figure(go.Treemap(labels=labels, parents=[""] * len(labels), values=values, hovertext=hover))
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title="Section Treemap (by characters)")
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl") and length_by_sec:
            # simple horizontal bar of top 20 sections by length
            top = sorted(length_by_sec, key=lambda x: x[2], reverse=True)[:20]
            def draw(fig, ax):
                ax.barh([f"{sid} {title[:24]}" for sid, title, _, _ in top], [v for _, _, v, _ in top])
                ax.invert_yaxis()
                ax.set_xlabel("Characters")
                ax.set_title("Top 20 Sections by Characters")
            uri = _mpl_png_uri(draw)
            divs.append(f"<img src='{uri}' alt='Sections bar'/>")
        else:
            divs.append(_html_notice("No plotting backend found. Install plotly or matplotlib."))

        # Chunk density per page
        chunk_count = [0] * (self.page_max + 1)
        char_count = [0] * (self.page_max + 1)
        for ch in self.chunks:
            ps, pe = max(1, ch.page_start), max(1, ch.page_end)
            for p in range(ps, min(self.page_max, pe) + 1):
                chunk_count[p] += 1
                char_count[p] += len(ch.text)

        if self._flags.get("has_plotly") and self.page_max > 0:
            go = self._mods["plotly_go"]
            x = list(range(1, self.page_max + 1))
            fig = go.Figure()
            fig.add_bar(x=x, y=[chunk_count[p] for p in x], name="#Chunks")
            fig.add_scatter(x=x, y=[char_count[p] for p in x], mode="lines+markers", name="Chars")
            fig.update_layout(title="Chunk Density per Page", xaxis_title="Page", yaxis_title="Count / Chars")
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl") and self.page_max > 0:
            def draw(fig, ax):
                x = list(range(1, self.page_max + 1))
                ax.bar(x, [chunk_count[p] for p in x], alpha=0.7)
                ax2 = ax.twinx()
                ax2.plot(x, [char_count[p] for p in x], marker="o")
                ax.set_xlabel("Page")
                ax.set_ylabel("#Chunks")
                ax2.set_ylabel("Characters")
                ax.set_title("Chunk Density per Page")
            uri = _mpl_png_uri(draw)
            divs.append(f"<img src='{uri}' alt='Chunk density'/>")

        # Chunk length & overlap histograms
        lengths = [len(ch.text) for ch in self.chunks]
        overlaps = []
        # estimate overlap within each section by comparing consecutive chunks
        sec_to_chunks: Dict[str, List[Chunk]] = defaultdict(list)
        for ch in self.chunks:
            sec_to_chunks[ch.section_id].append(ch)
        for sec, chs in sec_to_chunks.items():
            chs_sorted = sorted(chs, key=lambda c: c.char_start)
            for i in range(1, len(chs_sorted)):
                prev, cur = chs_sorted[i-1], chs_sorted[i]
                ov = max(0, prev.char_end - cur.char_start)
                if len(cur.text) > 0:
                    overlaps.append(ov / max(1, len(cur.text)))

        if self._flags.get("has_plotly") and lengths:
            go = self._mods["plotly_go"]
            fig1 = go.Figure(go.Histogram(x=lengths, nbinsx=40))
            fig1.update_layout(title="Chunk Lengths", xaxis_title="Characters", yaxis_title="Count")
            divs.append(_plotly_div(fig1))
            if overlaps:
                fig2 = go.Figure(go.Histogram(x=overlaps, nbinsx=30))
                fig2.update_layout(title="Estimated Inter-Chunk Overlap (ratio)", xaxis_title="Overlap", yaxis_title="Count")
                divs.append(_plotly_div(fig2))
        elif self._flags.get("has_mpl") and lengths:
            def draw1(fig, ax):
                ax.hist(lengths, bins=40)
                ax.set_title("Chunk Lengths")
                ax.set_xlabel("Characters")
                ax.set_ylabel("Count")
            divs.append(f"<img src='{_mpl_png_uri(draw1)}' alt='Lengths'/>")
            if overlaps:
                def draw2(fig, ax):
                    ax.hist(overlaps, bins=30)
                    ax.set_title("Estimated Inter-Chunk Overlap (ratio)")
                    ax.set_xlabel("Overlap")
                    ax.set_ylabel("Count")
                divs.append(f"<img src='{_mpl_png_uri(draw2)}' alt='Overlap'/>")

        return "".join(divs)

    def _panel_linguistic(self) -> str:
        divs = []
        # sentence length distribution
        sent_lens = []
        for ch in self.chunks[:1000]:  # cap for speed
            for s in _split_sentences(ch.text):
                sent_lens.append(len(s))

        if self._flags.get("has_plotly") and sent_lens:
            go = self._mods["plotly_go"]
            fig = go.Figure(go.Histogram(x=sent_lens, nbinsx=40))
            fig.update_layout(title="Sentence Length Distribution", xaxis_title="Characters", yaxis_title="Count")
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl") and sent_lens:
            def draw(fig, ax):
                ax.hist(sent_lens, bins=40)
                ax.set_title("Sentence Length Distribution")
                ax.set_xlabel("Characters")
                ax.set_ylabel("Count")
            divs.append(f"<img src='{_mpl_png_uri(draw)}' alt='Sentence lengths'/>")

        # keyword freq (top 20)
        kw_counter = Counter()
        for ch in self.chunks[:2000]:
            text = ch.text
            if self.exclude_captions_in_keywords:
                text = _strip_caption_lines(text)
            kw_counter.update(_simple_tokens(text))
        top_kws = kw_counter.most_common(20)

        if self._flags.get("has_plotly") and top_kws:
            go = self._mods["plotly_go"]
            terms, counts = zip(*top_kws)
            fig = go.Figure(go.Bar(x=list(terms), y=list(counts)))
            fig.update_layout(title="Top Keywords", xaxis_title="Term", yaxis_title="Count")
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl") and top_kws:
            def draw(fig, ax):
                terms, counts = zip(*top_kws)
                ax.bar(terms, counts)
                ax.set_title("Top Keywords")
                ax.set_xlabel("Term")
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
            divs.append(f"<img src='{_mpl_png_uri(draw)}' alt='Keywords'/>")

        # spaCy NER (optional)
        if self._flags.get("has_spacy"):
            try:
                nlp = self._mods["spacy"].load("en_core_web_sm")
                ent_counter = Counter()
                # sample up to 60 chunks for speed
                for ch in self.chunks[:60]:
                    doc = nlp(ch.text[:4000])
                    ent_counter.update([ent.label_ for ent in doc.ents])
                if ent_counter:
                    if self._flags.get("has_plotly"):
                        go = self._mods["plotly_go"]
                        labels, vals = zip(*ent_counter.most_common())
                        fig = go.Figure(go.Bar(x=list(labels), y=list(vals)))
                        fig.update_layout(title="Named Entities by Type (sampled)", xaxis_title="Entity Type", yaxis_title="Count")
                        divs.append(_plotly_div(fig))
                    elif self._flags.get("has_mpl"):
                        def draw(fig, ax):
                            labels, vals = zip(*ent_counter.most_common())
                            ax.bar(labels, vals)
                            ax.set_title("Named Entities by Type (sampled)")
                            ax.tick_params(axis='x', rotation=45)
                        divs.append(f"<img src='{_mpl_png_uri(draw)}' alt='NER'/>")
            except Exception:
                divs.append(_html_notice("spaCy not ready (en_core_web_sm missing). Skipping NER."))

        # readability (optional)
        if self._flags.get("has_textstat") and self.sections:
            import random
            scores = []
            for s in self.sections[:100]:
                txt = s.text[:8000]
                try:
                    score = self._mods["textstat"].flesch_reading_ease(txt)
                    scores.append((s.section_id, s.title, score))
                except Exception:
                    continue
            if scores:
                if self._flags.get("has_plotly"):
                    go = self._mods["plotly_go"]
                    fig = go.Figure(go.Scatter(
                        x=list(range(len(scores))),
                        y=[sc for _, _, sc in scores],
                        mode="markers+lines",
                        text=[f"{sid} {title}" for sid, title, _ in scores]
                    ))
                    fig.update_layout(title="Readability (Flesch Reading Ease) by Section (sampled)",
                                      xaxis_title="Section index", yaxis_title="Score (higher=easier)")
                    divs.append(_plotly_div(fig))
                elif self._flags.get("has_mpl"):
                    def draw(fig, ax):
                        ax.plot([sc for _, _, sc in scores], marker="o")
                        ax.set_title("Readability (Flesch) by Section (sampled)")
                        ax.set_xlabel("Section index")
                        ax.set_ylabel("Score (higher=easier)")
                    divs.append(f"<img src='{_mpl_png_uri(draw)}' alt='Readability'/>")
        elif not self._flags.get("has_textstat"):
            divs.append(_html_notice("textstat not installed → skipping readability panel."))

        return "".join(divs)

    def _panel_semantic(self) -> str:
        divs = []
        if not (self._flags.get("has_sbert") and self._flags.get("has_np")):
            return _html_notice("sentence-transformers / numpy not installed → skipping semantic panel.")

        np = self._mods["np"]
        # sample chunks for embeddings
        MAX_EMB_CHUNKS = 800
        texts = []
        meta = []
        for ch in self.chunks[:MAX_EMB_CHUNKS]:
            if ch.text.strip():
                texts.append(ch.text)
                meta.append((ch.section_id, ch.page_start, ch.page_end, ch.chunk_id))

        if not texts:
            return _html_notice("No chunk texts for embeddings.")

        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            sbert = self._mods["SentenceTransformer"](model_name)
            X = np.asarray(sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        except Exception:
            return _html_notice("Embedding model failed to load/encode. Skipping semantic panel.")

        # 2D reduction
        Y = None
        if self._flags.get("has_umap"):
            try:
                Y = self._mods["UMAP"](n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(X)
            except Exception:
                Y = None
        if Y is None and self._flags.get("has_sklearn"):
            try:
                Y = self._mods["PCA"](n_components=2).fit_transform(X)
            except Exception:
                Y = None

        if Y is not None and self._flags.get("has_plotly"):
            go = self._mods["plotly_go"]
            hover = [f"{sid} · p{ps}-{pe} · {cid}" for sid, ps, pe, cid in meta]
            fig = go.Figure(go.Scatter(
                x=Y[:, 0], y=Y[:, 1], mode="markers",
                text=hover
            ))
            fig.update_layout(title="Semantic Map of Chunks (2D)", xaxis_title="dim-1", yaxis_title="dim-2")
            divs.append(_plotly_div(fig))

        # section cosine heatmap (top K sections by chars)
        sec_lengths = defaultdict(int)
        for ch in self.chunks:
            sec_lengths[ch.section_id] += len(ch.text)
        top_secs = [sid for sid, _ in sorted(sec_lengths.items(), key=lambda x: x[1], reverse=True)[:25]]

        if top_secs and self._flags.get("has_sklearn") and self._flags.get("has_plotly"):
            # mean vector per section
            sec_vecs = []
            sec_labels = []
            for sid in top_secs:
                idxs = [i for i, (msid, *_rest) in enumerate(meta) if msid == sid]
                if not idxs:
                    continue
                vec = X[idxs].mean(axis=0)
                sec_vecs.append(vec)
                # choose one title exemplar
                title = ""
                for s in self.sections:
                    if s.section_id == sid:
                        title = s.title
                        break
                sec_labels.append(f"{sid} {title[:40]}")
            if len(sec_vecs) >= 2:
                sims = self._mods["cosine_similarity"](np.vstack(sec_vecs))
                go = self._mods["plotly_go"]
                fig = go.Figure(data=go.Heatmap(
                    z=sims, x=sec_labels, y=sec_labels, colorscale="Viridis"))
                fig.update_layout(title="Section Cosine Similarity (top by length)")
                divs.append(_plotly_div(fig))

        # outlier scores
        if self._flags.get("has_sklearn") and Y is not None and self._flags.get("has_plotly"):
            Iso = self._mods["IsolationForest"]
            iso = Iso(n_estimators=200, contamination=0.05, random_state=42)
            scores = -iso.fit_predict(X)  # 1 = inlier, -1 = outlier → invert sign-ish
            go = self._mods["plotly_go"]
            fig = go.Figure(go.Histogram(x=list(scores), nbinsx=20))
            fig.update_layout(title="Outlier Scores (IsolationForest on embeddings)")
            divs.append(_plotly_div(fig))

        return "".join(divs) if divs else _html_notice("Semantic charts unavailable (missing deps or empty).")

    def _panel_regulatory(self) -> str:
        divs = []
        # per-section counts
        sec_stats = []
        for s in self.sections:
            txt = s.text.lower()
            modal = {
                "shall": len(re.findall(r"\bshall\b", txt)),
                "must": len(re.findall(r"\bmust\b", txt)),
                "should": len(re.findall(r"\bshould\b", txt)),
                "may": len(re.findall(r"\bmay\b", txt)),
            }
            reg = {t: txt.count(t) for t in _REG_TERMS}
            cits = {}
            for pat in _CITATION_PATTERNS:
                cits[pat] = len(re.findall(pat, s.text))
            sec_stats.append((s.section_id, s.title, modal, reg, cits))

        # modal verbs stacked bar
        if self._flags.get("has_plotly") and sec_stats:
            go = self._mods["plotly_go"]
            labels = [f"{sid} {title[:24]}" for sid, title, *_ in sec_stats]
            mod_matrix = {k: [m[k] for _, _, m, _, _ in sec_stats] for k in ["shall", "must", "should", "may"]}
            fig = go.Figure()
            for k in ["shall", "must", "should", "may"]:
                fig.add_bar(x=labels, y=mod_matrix[k], name=k)
            fig.update_layout(barmode="stack", title="Modal Verbs per Section (requirement strength)",
                              xaxis_title="Section", yaxis_title="Count", xaxis_tickangle=-45)
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl") and sec_stats:
            def draw(fig, ax):
                labels = [f"{sid} {title[:18]}" for sid, title, *_ in sec_stats][:20]
                idx = range(len(labels))
                mod_matrix = {k: [m[k] for _, _, m, _, _ in sec_stats][:20] for k in ["shall", "must", "should", "may"]}
                bottom = [0]*len(labels)
                for k in ["shall", "must", "should", "may"]:
                    ax.bar(idx, mod_matrix[k], bottom=bottom, label=k)
                    bottom = [a+b for a,b in zip(bottom, mod_matrix[k])]
                ax.set_title("Modal Verbs per Section (top 20)")
                ax.set_xticks(list(idx))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.legend()
            divs.append(f"<img src='{_mpl_png_uri(draw)}' alt='Modal verbs'/>")

        # regulatory term freq (top 12 terms overall)
        total_reg = Counter()
        for _, _, _, reg, _ in sec_stats:
            total_reg.update(reg)
        top_reg = total_reg.most_common(12)
        if self._flags.get("has_plotly") and top_reg:
            go = self._mods["plotly_go"]
            terms, counts = zip(*top_reg)
            fig = go.Figure(go.Bar(x=list(terms), y=list(counts)))
            fig.update_layout(title="Regulatory Term Frequency (overall)", xaxis_title="Term", yaxis_title="Count")
            divs.append(_plotly_div(fig))

        # citation density per section
        cit_counts = []
        for sid, title, _, _, cits in sec_stats:
            cit_counts.append((f"{sid} {title[:24]}", sum(cits.values())))
        if self._flags.get("has_plotly") and cit_counts:
            go = self._mods["plotly_go"]
            labels, vals = zip(*cit_counts)
            fig = go.Figure(go.Bar(x=list(labels[:30]), y=list(vals[:30])))
            fig.update_layout(title="Citation Density per Section (top 30)", xaxis_title="Section", yaxis_title="#Citations",
                              xaxis_tickangle=-45)
            divs.append(_plotly_div(fig))

        return "".join(divs) if divs else _html_notice("No regulatory/citation signals detected.")
    
    
    def _panel_figtab(self) -> str:
        """Bar charts: figures per section, tables per section (top 25)."""
        divs = []
        rows = []   
        for s in self.sections:
            f, t = _count_fig_tab(s.text)
            rows.append((f"{s.section_id} {s.title[:40]}", f, t))
        rows = [r for r in rows if (r[1] + r[2]) > 0]
        if not rows:
            return _html_notice("No figures or tables detected (by heuristic).")

        # top 25 by total count
        rows.sort(key=lambda r: (r[1] + r[2]), reverse=True)
        rows = rows[:25]

        labels = [r[0] for r in rows]
        figs = [r[1] for r in rows]
        tabs = [r[2] for r in rows]

        if self._flags.get("has_plotly"):
            go = self._mods["plotly_go"]
            fig = go.Figure()
            fig.add_bar(x=labels, y=figs, name="Figures")
            fig.add_bar(x=labels, y=tabs, name="Tables")
            fig.update_layout(barmode="group", title="Figures/Tables per Section (top 25)",
                            xaxis_tickangle=-45, xaxis_title="Section", yaxis_title="Count")
            divs.append(_plotly_div(fig))
        elif self._flags.get("has_mpl"):
            def draw(fig, ax):
                import numpy as np
                idx = np.arange(len(labels))
                ax.bar(idx-0.2, figs, width=0.4, label="Figures")
                ax.bar(idx+0.2, tabs, width=0.4, label="Tables")
                ax.set_xticks(idx)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_title("Figures/Tables per Section (top 25)")
                ax.legend()
            uri = _mpl_png_uri(draw)
            divs.append(f"<img src='{uri}' alt='Figures Tables'/>")
        else:
            divs.append(_html_notice("No plotting backend found. Install plotly or matplotlib."))

        return "".join(divs)
    
    def _panel_ocr_density(self) -> str:
        """
        Heuristic OCR/low-text density panel:
        - Bar of characters per page
        - Highlights pages below a low threshold
        - Shows count of low-density pages (likely OCR/image-only pages)
        """
        divs = []
        # chars per page
        char_count = [0] * (self.page_max + 1)
        for ch in self.chunks:
            ps, pe = max(1, ch.page_start), max(1, ch.page_end)
            add = len(ch.text) // max(1, (pe - ps + 1))
            for p in range(ps, min(self.page_max, pe) + 1):
                char_count[p] += add

        # threshold: choose 25th percentile of nonzero pages or fixed 400
        nonzero = [c for c in char_count[1:] if c > 0]
        if nonzero:
            import numpy as np
            thresh = max(400, int(np.percentile(nonzero, 25)))
        else:
            thresh = 400

        low_pages = [p for p in range(1, self.page_max + 1) if char_count[p] < thresh]
        info_html = f"<p>Low-density pages (&lt; {thresh} chars): <b>{len(low_pages)}</b> / {self.page_max}</p>"

        if self._flags.get("has_plotly"):
            go = self._mods["plotly_go"]
            x = list(range(1, self.page_max + 1))
            colors = ["#d9534f" if char_count[p] < thresh else "#5bc0de" for p in x]
            fig = go.Figure(go.Bar(x=x, y=[char_count[p] for p in x], marker_color=colors))
            fig.update_layout(title="OCR / Low-Text Density (chars per page)",
                            xaxis_title="Page", yaxis_title="Characters")
            divs.append(info_html + _plotly_div(fig))
        elif self._flags.get("has_mpl"):
            def draw(fig, ax):
                x = list(range(1, self.page_max + 1))
                y = [char_count[p] for p in x]
                ax.bar(x, y)
                ax.axhline(thresh, color="red", linestyle="--", linewidth=1)
                ax.set_title("OCR / Low-Text Density (chars per page)")
                ax.set_xlabel("Page"); ax.set_ylabel("Characters")
            divs.append(info_html + f"<img src='{_mpl_png_uri(draw)}' alt='OCR density'/>")
        else:
            divs.append(_html_notice("No plotting backend found."))

        return "".join(divs)


    def _build_quality_table(self) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (html_table, findings_dict)
        """
        has_pd = self._flags.get("has_pd")
        if not has_pd:
            return _html_notice("pandas not installed → skipping quality table."), {}

        pd = self._mods["pd"]

        # aggregate per section
        rows = []
        sec_to_chunks: Dict[str, List[Chunk]] = defaultdict(list)
        for ch in self.chunks:
            sec_to_chunks[ch.section_id].append(ch)

        for s in self.sections:
            chs = sorted(sec_to_chunks.get(s.section_id, []), key=lambda c: c.char_start)
            lens = [len(c.text) for c in chs]
            tiny = sum(1 for L in lens if L < self.tiny_chunk_threshold)
            long = sum(1 for L in lens if L > 1600)  # default from config
            # repetition via Jaccard on crude 3-gram sets between neighbors
            rep_hits = 0
            rep_total = 0
            for i in range(1, len(chs)):
                a = set(re.findall(r"[A-Za-z]{3,}", chs[i-1].text.lower()))
                b = set(re.findall(r"[A-Za-z]{3,}", chs[i].text.lower()))
                if a and b:
                    j = len(a & b) / max(1, len(a | b))
                    rep_total += 1
                    if j >= 0.5:
                        rep_hits += 1
            rep_ratio = rep_hits / rep_total if rep_total else 0.0

            flags = []
            if lens and (sum(lens) / max(1, len(lens))) > 1700:
                flags.append("long")
            if rep_ratio > 0.3:
                flags.append("repetitive")
            if (s.page_end - s.page_start + 1) == 0:
                flags.append("no-pages")

            rows.append({
                "section_id": s.section_id,
                "title": s.title,
                "pages": f"{s.page_start}-{s.page_end}",
                "chunks": len(chs),
                "avg_len": int(sum(lens)/max(1, len(lens))) if lens else 0,
                "pct_tiny": _pct(tiny / max(1, len(lens))) if lens else "0.0%",
                "pct_long": _pct(long / max(1, len(lens))) if lens else "0.0%",
                "repeat_ratio": f"{rep_ratio:.2f}",
                "flags": ", ".join(flags) if flags else "-",
            })

        df = pd.DataFrame(rows)
        # sort by section_id (numeric if possible)
        def _sid_key(s):
            parts = [p for p in re.split(r"[^\d]+", str(s)) if p.isdigit()]
            return tuple(int(p) for p in parts) if parts else (1e9,)
        df = df.sort_values(by="section_id", key=lambda col: col.map(_sid_key))

        # HTML table (minimal CSS applied in page)
        html = df.to_html(index=False, escape=True)

        # findings JSON
        findings = {
            "doc_id": self.doc_id,
            "run_dir": self.run_dir,
            "sections": len(self.sections),
            "chunks": len(self.chunks),
            "page_span": f"1-{self.page_max}",
            "quality": rows,
            "kpis": {
                "avg_chunk_len": int(sum(len(c.text) for c in self.chunks)/max(1, len(self.chunks))) if self.chunks else 0,
                "tiny_chunk_threshold": self.tiny_chunk_threshold,
            }
        }
        return html, findings

    # --------------------- Compose HTML & write files ---------------------

    def _compose_html(self, structure_html: str, ling_html: str, sem_html: str,
                  reg_html: str, table_html: str, figtab_html: str, ocr_html: str) -> str:
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Ingest Dashboard – {self.doc_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 1200px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .card {{ background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .card h2 {{ margin-top: 0; font-size: 18px; }}
    .kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }}
    .kpis .pill {{ background: #f5f5f5; border-radius: 999px; padding: 6px 12px; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #eee; padding: 6px 8px; text-align: left; }}
    th {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Ingest Dashboard – {self.doc_id}</h1>
  <div class="kpis">
    <div class="pill">Sections: {len(self.sections)}</div>
    <div class="pill">Chunks: {len(self.chunks)}</div>
    <div class="pill">Pages: 1–{self.page_max}</div>
  </div>

  <div class="grid">
        {_html_card("Structure", structure_html)}
        {_html_card("Linguistic Signals", ling_html)}
        {_html_card("Semantic Map & Similarity", sem_html)}
        {_html_card("Regulatory / Domain Signals", reg_html)}
        {_html_card("Figures & Tables", figtab_html)}
        {_html_card("OCR / Low-Text Density", ocr_html)}
        {_html_card("Quality Table", table_html)}
  </div>
</body>
</html>
"""
    def _compose_summary(self, findings: dict) -> str:
        """Return Markdown summary string for ingest_summary.md."""
        # Top keywords (caption lines optionally stripped)
        from collections import Counter
        kw = Counter()
        for ch in self.chunks[:2000]:
            text = ch.text
            if getattr(self, "exclude_captions_in_keywords", True):
                text = _strip_caption_lines(text)
            kw.update(_simple_tokens(text))
        top_kw = ", ".join([f"{t}({c})" for t, c in kw.most_common(12)])

        # Top regulatory terms
        reg_counter = Counter()
        for s in self.sections:
            txt = s.text.lower()
            reg_counter.update({t: txt.count(t) for t in _REG_TERMS})
        top_reg = ", ".join([f"{t}({c})" for t, c in reg_counter.most_common(8)])

        # Basic KPIs
        sections_n = len(self.sections)
        chunks_n = len(self.chunks)
        page_span = f"1–{self.page_max}"

        lines = [
            f"# Ingest Summary – {self.doc_id}",
            "",
            f"- Sections: **{sections_n}**",
            f"- Chunks: **{chunks_n}**",
            f"- Page span: **{page_span}**",
            f"- Top keywords: {top_kw or '—'}",
            f"- Top regulatory terms: {top_reg or '—'}",
        ]

        if findings and isinstance(findings, dict):
            avg_len = findings.get("kpis", {}).get("avg_chunk_len")
            if avg_len is not None:
                lines.append(f"- Avg chunk length: **{avg_len} chars**")

        lines.append("")
        return "\n".join(lines)

    # def run(self) -> Dict[str, str]:
    #     # build all panels

    #     try:
    #         structure_html = self._panel_structure()
    #     except Exception:
    #         structure_html = _html_notice("Structure panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     try:
    #         ling_html = self._panel_linguistic()
    #     except Exception:
    #         ling_html = _html_notice("Linguistic panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     try:
    #         sem_html = self._panel_semantic()
    #     except Exception:
    #         sem_html = _html_notice("Semantic panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     try:
    #         reg_html = self._panel_regulatory()
    #     except Exception:
    #         reg_html = _html_notice("Regulatory panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     try:
    #         table_html, findings = self._build_quality_table()
    #     except Exception:
    #         table_html, findings = _html_notice("Quality table error."), {}

    #     try:
    #         figtab_html = self._panel_figtab()
    #     except Exception:
    #         figtab_html = _html_notice("Figures/Tables panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     try:
    #         ocr_html = self._panel_ocr_density()
    #     except Exception:
    #         ocr_html = _html_notice("OCR density panel error:\n<pre>" + traceback.format_exc() + "</pre>")

    #     # write files
    #     html_out = os.path.join(self.viz_dir, "ingest_dashboard.html")
    #     md_out = os.path.join(self.viz_dir, "ingest_summary.md")
    #     json_out = os.path.join(self.viz_dir, "ingest_findings.json")

    #     with open(html_out, "w", encoding="utf-8") as f:
    #         f.write(self._compose_html(structure_html, ling_html, sem_html, reg_html, table_html, figtab_html, ocr_html))

    #     # summary markdown
    #     top_kw = Counter()
    #     for ch in self.chunks[:2000]:
    #         top_kw.update(_simple_tokens(ch.text))
    #     top_kw_list = ", ".join([f"{t}({c})" for t, c in top_kw.most_common(12)])

    #     with open(md_out, "w", encoding="utf-8") as f:
    #         f.write(f"# Ingest Summary – {self.doc_id}\n\n")
    #         f.write(f"- Sections: **{len(self.sections)}**\n")
    #         f.write(f"- Chunks: **{len(self.chunks)}**\n")
    #         f.write(f"- Page span: **1–{self.page_max}**\n")
    #         f.write(f"- Top keywords: {top_kw_list}\n")

    #     if findings:
    #         with open(json_out, "w", encoding="utf-8") as f:
    #             json.dump(findings, f, indent=2)

    #     return {
    #         "findings_json": json_out,
    #         "dashboard_html": html_out,
    #         "summary_md": md_out,
    #     }
    def run(self) -> Dict[str, str]:
        # build all panels

        try:
            structure_html = self._panel_structure()
        except Exception:
            structure_html = _html_notice("Structure panel error:\n<pre>" + traceback.format_exc() + "</pre>")

        try:
            ling_html = self._panel_linguistic()
        except Exception:
            ling_html = _html_notice("Linguistic panel error:\n<pre>" + traceback.format_exc() + "</pre>")

        try:
            sem_html = self._panel_semantic()
        except Exception:
            sem_html = _html_notice("Semantic panel error:\n<pre>" + traceback.format_exc() + "</pre>")

        try:
            reg_html = self._panel_regulatory()
        except Exception:
            reg_html = _html_notice("Regulatory panel error:\n<pre>" + traceback.format_exc() + "</pre>")

        try:
            table_html, findings = self._build_quality_table()
        except Exception:
            table_html, findings = _html_notice("Quality table error."), {}

        # -------------------- NEW LOGIC HERE --------------------
        # only build the new panels if not explicitly skipped
        if not getattr(self, "_skip_figtab", False):
            try:
                figtab_html = self._panel_figtab()
            except Exception:
                figtab_html = _html_notice("Figures/Tables panel error:\n<pre>" + traceback.format_exc() + "</pre>")
        else:
            figtab_html = _html_notice("(Skipped Figures/Tables panel)")

        if not getattr(self, "_skip_ocr", False):
            try:
                ocr_html = self._panel_ocr_density()
            except Exception:
                ocr_html = _html_notice("OCR density panel error:\n<pre>" + traceback.format_exc() + "</pre>")
        else:
            ocr_html = _html_notice("(Skipped OCR density panel)")
        # --------------------------------------------------------

        # now pass these to your HTML composition function
        html_path = os.path.join(self.viz_dir, "ingest_dashboard.html")
        md_path = os.path.join(self.viz_dir, "ingest_summary.md")
        json_path = os.path.join(self.viz_dir, "ingest_findings.json")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._compose_html(
                structure_html,
                ling_html,
                sem_html,
                reg_html,
                table_html,
                figtab_html,
                ocr_html
            ))

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._compose_summary(findings))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=2)

        return {
            "findings_json": os.path.abspath(json_path),
            "dashboard_html": os.path.abspath(html_path),
            "summary_md": os.path.abspath(md_path),
        }
