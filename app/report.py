# app/report.py
# -*- coding: utf-8 -*-
"""
Report generator: aggregates ingest, index, and assess artifacts into Markdown.
- One consolidated run-level report (reports/report_YYYYMMDD_HHMM.md)
- Per-document mini reports (reports/per_doc/<DOC_ID>.md)

Inputs it looks for (if present):
- Ingest:
  - parsed/<DOC_ID>_sections.json
  - chunks/<DOC_ID>_chunks.jsonl
  - viz/ingest_dashboard.html  (link)
  - viz/ingest_findings.json   (optional; created by ingest_viz)
- Index:
  - data/index/<DOC_ID>/faiss.index (presence)
  - data/index/<DOC_ID>/mapping.json
  - data/index/<DOC_ID>/viz/index_dashboard.html (link)
- Assess:
  - assess/<DOC_ID>/results.jsonl
  - assess/<DOC_ID>/summary.json
  - assess/<DOC_ID>/viz/assessment_dashboard.html (link)

"""

from __future__ import annotations
import os, json, glob, datetime, math
from typing import Dict, Any, List, Optional

# Optional: pandas for compact tables (fallback to plain)
try:
    import pandas as pd
except Exception:
    pd = None


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_jsonl(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)
    except Exception:
        return
        yield  # pragma: no cover


def _md_h1(s: str) -> str: return f"# {s}\n"
def _md_h2(s: str) -> str: return f"\n## {s}\n"
def _md_h3(s: str) -> str: return f"\n### {s}\n"
def _md_kv(k: str, v: Any) -> str: return f"- **{k}:** {v}"
def _pct(x: float, d: int = 1) -> str: return f"{round(100.0 * x, d)}%"


class ReportGenerator:
    def __init__(self, cfg: dict, run_dir: str, doc_ids: Optional[List[str]] = None):
        self.cfg = cfg or {}
        self.run_dir = os.path.abspath(run_dir)
        self.index_root = os.path.abspath(self.cfg["paths"]["index_dir"])
        # Discover doc_ids if not provided
        if not doc_ids:
            pattern = os.path.join(self.run_dir, "parsed", "*_sections.json")
            doc_ids = [os.path.basename(p)[:-len("_sections.json")] for p in glob.glob(pattern)]
        self.doc_ids = sorted(doc_ids or [])
        # Outputs
        self.reports_dir = os.path.join(self.run_dir, "reports")
        self.per_doc_dir = os.path.join(self.reports_dir, "per_doc")
        _ensure_dir(self.per_doc_dir)

    # -------------------- INGEST --------------------

    def _ingest_stats(self, doc_id: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"doc_id": doc_id}
        sections_path = os.path.join(self.run_dir, "parsed", f"{doc_id}_sections.json")
        chunks_path = os.path.join(self.run_dir, "chunks", f"{doc_id}_chunks.jsonl")
        viz_dir = os.path.join(self.run_dir, "viz")
        ingest_dash = os.path.join(viz_dir, "ingest_dashboard.html")
        ingest_findings = os.path.join(viz_dir, "ingest_findings.json")

        stats["sections_json"] = sections_path if os.path.exists(sections_path) else None
        stats["chunks_jsonl"] = chunks_path if os.path.exists(chunks_path) else None
        stats["ingest_dashboard"] = ingest_dash if os.path.exists(ingest_dash) else None
        findings = _read_json(ingest_findings) if os.path.exists(ingest_findings) else None

        # Fallback counts
        n_sections = None
        if stats["sections_json"]:
            sj = _read_json(stats["sections_json"])
            if sj and "sections" in sj:
                n_sections = len(sj["sections"])
        stats["sections_count"] = n_sections

        n_chunks = 0
        total_chars = 0
        approx_tokens = 0
        if stats["chunks_jsonl"]:
            for rec in _iter_jsonl(stats["chunks_jsonl"]):
                n_chunks += 1
                txt = rec.get("text") or ""
                total_chars += len(txt)
                approx_tokens += max(1, len(txt) // 4)
        stats["chunks_count"] = n_chunks or None
        stats["avg_chunk_chars"] = round(total_chars / n_chunks, 1) if n_chunks else None
        stats["avg_chunk_tokens_approx"] = round(approx_tokens / n_chunks, 1) if n_chunks else None

        # Pull figure/table/OCR stats if available
        if findings:
            stats["figures_total"] = findings.get("figures_total")
            stats["tables_total"] = findings.get("tables_total")
            stats["ocr_pages"] = findings.get("ocr_pages")
            stats["ocr_pages_count"] = findings.get("ocr_pages_count")
        return stats

    # -------------------- INDEX --------------------

    def _index_stats(self, doc_id: str) -> Dict[str, Any]:
        d = {"doc_id": doc_id}
        doc_index_dir = os.path.join(self.index_root, doc_id)
        d["index_dir"] = doc_index_dir if os.path.isdir(doc_index_dir) else None
        d["faiss_index"] = os.path.join(doc_index_dir, "faiss.index") if os.path.exists(os.path.join(doc_index_dir, "faiss.index")) else None
        d["mapping_json"] = os.path.join(doc_index_dir, "mapping.json") if os.path.exists(os.path.join(doc_index_dir, "mapping.json")) else None
        dash = os.path.join(doc_index_dir, "viz", "index_dashboard.html")
        d["index_dashboard"] = dash if os.path.exists(dash) else None

        # If mapping exists, count chunks
        n_map = None
        if d["mapping_json"]:
            m = _read_json(d["mapping_json"])
            if m and "chunks" in m:
                n_map = len(m["chunks"])
        d["mapped_chunks"] = n_map
        return d

    # -------------------- ASSESS --------------------

    def _assess_stats(self, doc_id: str) -> Dict[str, Any]:
        out = {"doc_id": doc_id}
        assess_dir = os.path.join(self.run_dir, "assess", doc_id)
        res = os.path.join(assess_dir, "results.jsonl")
        summ = os.path.join(assess_dir, "summary.json")
        dash = os.path.join(assess_dir, "viz", "assessment_dashboard.html")
        out["results_jsonl"] = res if os.path.exists(res) else None
        out["summary_json"] = summ if os.path.exists(summ) else None
        out["assessment_dashboard"] = dash if os.path.exists(dash) else None

        summary = _read_json(summ) if out["summary_json"] else None
        if summary:
            out["total_items"] = summary.get("total_items")
            out["counts"] = summary.get("counts", {})
            out["percents"] = summary.get("percents", {})
        else:
            out["total_items"] = None
            out["counts"] = {}
            out["percents"] = {}

        # Token budget overview from snippets (max/avg prompt_tokens)
        max_prompt = None
        avg_prompt = None
        if out["results_jsonl"]:
            pts = []
            for row in _iter_jsonl(out["results_jsonl"]):
                for sn in row.get("snippets", []) or []:
                    pt = sn.get("prompt_tokens")
                    if isinstance(pt, int):
                        pts.append(pt)
            if pts:
                max_prompt = max(pts)
                avg_prompt = round(sum(pts)/len(pts), 1)
        out["max_prompt_tokens"] = max_prompt
        out["avg_prompt_tokens"] = avg_prompt
        return out

    # -------------------- RENDERING --------------------

    def _render_doc_md(self, doc_id: str, ing: Dict[str, Any], idx: Dict[str, Any], ass: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(_md_h1(f"Compliance Report — {doc_id}"))

        # Ingest
        lines.append(_md_h2("Ingest"))
        lines.append(_md_kv("Sections file", ing.get("sections_json") or "—"))
        lines.append(_md_kv("Chunks file", ing.get("chunks_jsonl") or "—"))
        if ing.get("sections_count") is not None:
            lines.append(_md_kv("Sections", ing["sections_count"]))
        if ing.get("chunks_count") is not None:
            lines.append(_md_kv("Chunks", ing["chunks_count"]))
        if ing.get("avg_chunk_chars") is not None:
            lines.append(_md_kv("Avg chunk length (chars)", ing["avg_chunk_chars"]))
        if ing.get("avg_chunk_tokens_approx") is not None:
            lines.append(_md_kv("Avg chunk tokens (≈)", ing["avg_chunk_tokens_approx"]))
        if ing.get("figures_total") is not None or ing.get("tables_total") is not None:
            lines.append(_md_kv("Figures (total)", ing.get("figures_total", "—")))
            lines.append(_md_kv("Tables (total)", ing.get("tables_total", "—")))
        if ing.get("ocr_pages_count") is not None:
            lines.append(_md_kv("OCR pages", ing["ocr_pages_count"]))
        if ing.get("ingest_dashboard"):
            lines.append(_md_kv("Ingest dashboard", ing["ingest_dashboard"]))

        # Index
        lines.append(_md_h2("Index"))
        lines.append(_md_kv("Index dir", idx.get("index_dir") or "—"))
        lines.append(_md_kv("FAISS index", idx.get("faiss_index") or "—"))
        lines.append(_md_kv("Mapping", idx.get("mapping_json") or "—"))
        if idx.get("mapped_chunks") is not None:
            lines.append(_md_kv("Mapped chunks", idx["mapped_chunks"]))
        if idx.get("index_dashboard"):
            lines.append(_md_kv("Index dashboard", idx["index_dashboard"]))

        # Assess
        lines.append(_md_h2("Assessment"))
        lines.append(_md_kv("Results", ass.get("results_jsonl") or "—"))
        lines.append(_md_kv("Summary", ass.get("summary_json") or "—"))
        if ass.get("total_items") is not None:
            lines.append(_md_kv("Checklist items", ass["total_items"]))
        if ass.get("counts"):
            lines.append("\n**Verdicts**")
            for k, v in ass["counts"].items():
                pc = ass["percents"].get(k)
                if pc is not None:
                    lines.append(f"- {k}: **{v}** ({pc}%)")
                else:
                    lines.append(f"- {k}: **{v}**")
        if ass.get("avg_prompt_tokens") is not None:
            lines.append(_md_kv("Avg prompt tokens (est.)", ass["avg_prompt_tokens"]))
        if ass.get("max_prompt_tokens") is not None:
            lines.append(_md_kv("Max prompt tokens (est.)", ass["max_prompt_tokens"]))
        if ass.get("assessment_dashboard"):
            lines.append(_md_kv("Assessment dashboard", ass["assessment_dashboard"]))

        # Top “Not Covered” / “Partially Covered”
        rows = []
        res_path = ass.get("results_jsonl")
        if res_path and os.path.exists(res_path):
            for r in _iter_jsonl(res_path):
                rows.append({
                    "id": r.get("item_id"),
                    "category": r.get("category"),
                    "verdict": r.get("verdict"),
                    "confidence": r.get("confidence"),
                    "question": r.get("question"),
                })
        if rows:
            lines.append(_md_h3("Items needing attention"))
            table_rows = [r for r in rows if r["verdict"] in ("Partially Covered", "Not Covered")]
            # sort: Not Covered first, then Partial, low confidence first
            order = {"Not Covered": 0, "Partially Covered": 1, "Covered": 2}
            table_rows.sort(key=lambda r: (order.get(r["verdict"], 9), r.get("confidence") or 9))
            table_rows = table_rows[:20]
            if pd is not None and table_rows:
                df = pd.DataFrame(table_rows, columns=["id", "verdict", "confidence", "category", "question"])
                lines.append(df.to_markdown(index=False))
            elif table_rows:
                lines.append("| ID | Verdict | Conf. | Category | Question |")
                lines.append("|---:|---|:---:|---|---|")
                for r in table_rows:
                    lines.append(f"| {r['id']} | {r['verdict']} | {r.get('confidence','')} | {r['category']} | {r['question']} |")
        return "\n".join(lines) + "\n"

    def _render_run_md(self, per_doc_paths: List[str]) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = []
        lines.append(_md_h1("Matrix Compliance — Run Summary"))
        lines.append(_md_kv("Run directory", self.run_dir))
        lines.append(_md_kv("Generated at", now))
        lines.append(_md_kv("Documents", ", ".join(self.doc_ids) if self.doc_ids else "—"))

        # Aggregate verdict totals across docs
        grand_counts: Dict[str, int] = {}
        total_items = 0
        for doc_id in self.doc_ids:
            s = _read_json(os.path.join(self.run_dir, "assess", doc_id, "summary.json")) or {}
            total_items += int(s.get("total_items", 0))
            for k, v in (s.get("counts") or {}).items():
                grand_counts[k] = grand_counts.get(k, 0) + int(v or 0)
        if total_items:
            lines.append(_md_h2("Overall Verdicts"))
            for k, v in grand_counts.items():
                pc = f"{round(100.0*v/total_items, 1)}%"
                lines.append(f"- {k}: **{v}** ({pc})")

        # Link per-doc reports
        if per_doc_paths:
            lines.append(_md_h2("Per-document reports"))
            for p in per_doc_paths:
                rel = os.path.relpath(p, start=self.run_dir)
                lines.append(f"- {rel}")
        return "\n".join(lines) + "\n"

    # -------------------- MAIN --------------------

    def run(self) -> Dict[str, str]:
        per_doc_paths: List[str] = []
        for doc_id in self.doc_ids:
            ing = self._ingest_stats(doc_id)
            idx = self._index_stats(doc_id)
            ass = self._assess_stats(doc_id)
            md = self._render_doc_md(doc_id, ing, idx, ass)
            out_path = os.path.join(self.per_doc_dir, f"{doc_id}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
            per_doc_paths.append(out_path)

        run_md = self._render_run_md(per_doc_paths)
        run_md_path = os.path.join(self.reports_dir, f"report_{os.path.basename(self.run_dir)}.md")
        with open(run_md_path, "w", encoding="utf-8") as f:
            f.write(run_md)

        return {
            "run_report_md": os.path.abspath(run_md_path),
            "per_doc": [os.path.abspath(p) for p in per_doc_paths],
        }
