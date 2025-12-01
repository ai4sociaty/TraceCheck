# app/assess.py
# -*- coding: utf-8 -*-
"""
Compliance assessment pipeline (single-stage, per-chunk):

- Load checklist (JSON)
- For each atomic checklist question:
    - Hybrid retrieval → strict top-N chunks (no diversification)
    - For each chunk: build a small context (question + one chunk)
    - Single LLM call returns {"verdict": "Covered|Partially Covered|Not Covered", "confidence": 1-5, "rationale": "..."}
- Aggregate per-chunk verdicts into a per-question verdict (priority Covered > Partial > Not Covered)
- Persist results as JSONL + summary.json + report.md

Notes:
- No trimming is performed (assumes chunks <~300 tokens).
- Works in dry-run (no LLM calls) with a simple heuristic.

Outputs (per doc_id):
data/runs/<run_id>/assess/<doc_id>/
  ├─ results.jsonl   # per-question consolidated results
  ├─ summary.json    # aggregate counts
  └─ report.md       # human-readable summary
"""

from __future__ import annotations

import os
import json
import math
import glob
import logging
from typing import List, Dict, Tuple, Optional

from .retrieve import HybridSearcher
from . import utils

# Optional LLM import (graceful fallback if missing)
try:
    from .llm import (
        SingleStageJudge,  # class with .run(requirement, context) -> dict
        count_tokens,      # optional tokenizer
    )
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False
    SingleStageJudge = None
    count_tokens = None





# ---- ANSI colors for logs ----
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_RESET = "\x1b[0m"

WARN_PROMPT_YELLOW = 1000    # yellow warning if prompt tokens exceed
WARN_PROMPT_RED = 15000      # red warning if prompt tokens exceed

def _tok_count_safe(text: str, model: Optional[str], fallback_ratio: float = 4.0) -> int:
    """Count tokens using provided tokenizer, fallback to char/ratio."""
    if not text:
        return 0
    if count_tokens:
        try:
            return int(count_tokens(text, model or "gpt-4o-mini"))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / fallback_ratio))

# ============================= Helpers =============================

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))

def _as_tokens(text: str, model: Optional[str] = None) -> int:
    if count_tokens:
        try:
            return int(count_tokens(text, model or "gpt-4o-mini"))
        except Exception:
            pass
    return _approx_tokens(text)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def _write_jsonl_line(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set(); out=[]
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _find_chunks_jsonl(run_dir: str, doc_id: str) -> Optional[str]:
    cand = os.path.join(run_dir, "chunks", f"{doc_id}_chunks.jsonl")
    if os.path.exists(cand):
        return cand
    pattern = os.path.join(run_dir, "chunks", "*_chunks.jsonl")
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        if base == f"{doc_id}_chunks.jsonl":
            return p
    return None


# ============================= Main Class =============================

class ComplianceAssessor:
    """
    Single-stage per-chunk compliance classification.
    """

    def __init__(
        self,
        cfg: dict,
        run_dir: str,
        doc_id: str,
        checklist_path: str = "documents/compliance_checklist.json",
        out_root: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        # Retrieval knobs
        k_bm25: int = 40,
        k_vec: int = 40,
        topn_merge: int = 50,
        topn_strict: int = 10,    # strictly select top-N by hybrid score
        # LLM / budget knobs (kept for logging/guardrails)
        llm_model: str = "gpt-4o-mini",
        #llm_model=args.model or "gpt-4o-mini",
        total_token_budget: int = 1000,
        prompt_budget: int = 800,
        # Execution
        dry_run: bool = False,
    ):
        self.cfg = cfg or {}
        self.run_dir = os.path.abspath(run_dir)
        self.doc_id = doc_id
        self.checklist_path = checklist_path
        self.out_dir = out_root or os.path.join(self.run_dir, "assess", doc_id)
        _ensure_dir(self.out_dir)

        self.log = logger or logging.getLogger(__name__)
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Retrieval config
        self.k_bm25 = int(k_bm25)
        self.k_vec = int(k_vec)
        self.topn_merge = int(topn_merge)
        self.topn_strict = int(topn_strict)

        # LLM / budget
        self.llm_model = llm_model
        self.total_token_budget = int(total_token_budget)
        self.prompt_budget = int(prompt_budget)

        # Execution mode
        self.dry_run = bool(dry_run)

        # Index + searcher
        self.index_root = os.path.abspath(self.cfg["paths"]["index_dir"])
        self.searcher = HybridSearcher(self.index_root, self.doc_id)

        # Load chunk texts for prompts
        self.chunk_texts: Dict[str, str] = self._load_chunk_texts()

        # ---- LLM tools (LAZY INIT) ----
        # Do NOT instantiate OpenAI-backed classes here; dry-run must not require API keys.
        self.evidence_extractor = None
        self.verdict_assessor = None
        self.single_judge = None  # will be created on first real LLM call in judge_chunk()

        # Outputs
        self.results_jsonl = os.path.join(self.out_dir, "results.jsonl")
        self.summary_json = os.path.join(self.out_dir, "summary.json")
        self.report_md = os.path.join(self.out_dir, "report.md")
        if os.path.exists(self.results_jsonl):
            os.remove(self.results_jsonl)


    # ---------------- checklist ----------------

    def load_checklist(self) -> List[dict]:
        obj = _read_json(self.checklist_path)
        items = obj.get("items", [])
        flat: List[dict] = []
        for it in items:
            base_id = it.get("id") or it.get("title") or "C?"
            title = it.get("title", "").strip()
            qs = it.get("questions", []) or []
            if not qs:
                flat.append({"id": str(base_id), "title": title, "question": title})
                continue
            for i, q in enumerate(qs, start=1):
                qid = f"{base_id}.{i}"
                flat.append({"id": qid, "title": title, "question": q.strip()})
        return flat

    # ---------------- retrieval ----------------

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        hits = self.searcher.search_hybrid(
            query=query,
            k_bm25=self.k_bm25,
            k_vec=self.k_vec,
            topn_merge=self.topn_merge,
            w_bm25=1.0,
            w_vec=1.0,
        )
        return hits

    def select_topn_strict(self, hits: List[Tuple[str, float]], n: int) -> List[Tuple[str, float]]:
        return hits[:max(0, n)]

    def fetch_chunk_meta(self, chunk_id: str) -> dict:
        return self.searcher.mapping.get(chunk_id, {})

    def _load_chunk_texts(self) -> Dict[str, str]:
        path = _find_chunks_jsonl(self.run_dir, self.doc_id)
        if not path or not os.path.exists(path):
            self.log.warning("Chunks jsonl not found; contexts will show placeholders. (%s)", path)
            return {}
        out: Dict[str, str] = {}
        for rec in _iter_jsonl(path):
            cid = rec.get("chunk_id")
            txt = rec.get("text") or ""
            if cid and txt:
                out[str(cid)] = txt
        self.log.info("Loaded chunk texts: %d", len(out))
        return out

    # ---------------- per-chunk context ----------------

    def _token_budget_report(
        self,
        header: str,
        requirement: str,
        chunk_text: str,
        completion_expect: int = 200,  # your expected completion size
    ) -> dict:
        """Compute token breakdown and emit warnings."""
        t_header = _tok_count_safe(header, self.llm_model)
        t_req    = _tok_count_safe(requirement, self.llm_model)
        t_chunk  = _tok_count_safe(chunk_text, self.llm_model)
        t_prompt = t_header + t_req + t_chunk
        t_comp   = int(completion_expect)
        t_total  = t_prompt + t_comp

        # proportions
        def pct(x, denom): return round(100.0 * (x / denom), 1) if denom else 0.0
        p_header = pct(t_header, t_prompt)
        p_req    = pct(t_req,    t_prompt)
        p_chunk  = pct(t_chunk,  t_prompt)
        p_comp   = pct(t_comp,   t_total)

        # color warnings
        if t_prompt > WARN_PROMPT_RED:
            self.log.error(_RED + "[TOKENS] Prompt est ~%d exceeds HARD limit %d." + _RESET, t_prompt, WARN_PROMPT_RED)
        elif t_prompt > WARN_PROMPT_YELLOW:
            self.log.warning(_YELLOW + "[TOKENS] Prompt est ~%d exceeds soft budget %d." + _RESET, t_prompt, WARN_PROMPT_YELLOW)

        # Optional trace
        self.log.debug(
            "[TOKENS] prompt≈%d (header=%d, req=%d, chunk=%d) | completion≈%d | total≈%d",
            t_prompt, t_header, t_req, t_chunk, t_comp, t_total
        )

        return {
            "prompt_tokens": t_prompt,
            "header_tokens": t_header,
            "requirement_tokens": t_req,
            "chunk_tokens": t_chunk,
            "completion_tokens_expect": t_comp,
            "total_tokens_expect": t_total,
            "pct_header_of_prompt": p_header,
            "pct_requirement_of_prompt": p_req,
            "pct_chunk_of_prompt": p_chunk,
            "pct_completion_of_total": p_comp,
            "warn_soft_exceeded": t_prompt > WARN_PROMPT_YELLOW,
            "warn_hard_exceeded": t_prompt > WARN_PROMPT_RED,
        }


    # def build_single_chunk_context(self, requirement: str, cid: str) -> Tuple[str, dict]:
    #     """
    #     Build the prompt context for one chunk (no trimming).
    #     """
    #     meta = self.fetch_chunk_meta(cid)
    #     sect = meta.get("section_title") or ""
    #     pstart = meta.get("page_start")
    #     pend = meta.get("page_end") if meta.get("page_end") is not None else meta.get("page_start")
    #     pages = f"p{pstart}–p{pend}"
    #     raw_text = self.chunk_texts.get(cid) or f"[Text not available; len={int(meta.get('text_len') or 0)} chars]"

    #     # Instruction: single-stage classification (3-way)
    #     header = (
    #         "You are a regulatory reviewer.\n"
    #         "Task: Given the requirement and ONE excerpt, classify coverage as exactly one of:\n"
    #         " - Covered\n - Partially Covered\n - Not Covered\n"
    #         "Base your decision ONLY on the excerpt (ignore outside knowledge).\n"
    #         "Return valid JSON with keys: verdict (string), confidence (1-5), rationale (<=80 tokens).\n\n"
    #         f"Requirement:\n{requirement}\n\n"
    #         "Excerpt:\n"
    #     )
    #     block = f"[Section: {sect} | {pages}]\n{raw_text}\n"
    #     context = (header + block).strip()

    #     snippet = {
    #         "chunk_id": cid,
    #         "section_title": sect,
    #         "pages": pages,
    #         "text_tokens_est": _as_tokens(raw_text, self.llm_model),
    #     }
    #     return context, snippet



    def build_single_chunk_context(self, requirement: str, cid: str) -> Tuple[str, dict]:
        """
        Build the prompt context for one chunk (no trimming).
        Also computes token budget and logs yellow/red warnings.
        """
        meta = self.fetch_chunk_meta(cid)
        sect = meta.get("section_title") or ""
        pstart = meta.get("page_start")
        pend = meta.get("page_end") if meta.get("page_end") is not None else meta.get("page_start")
        pages = f"p{pstart}–p{pend}"
        raw_text = self.chunk_texts.get(cid)
        if not raw_text:
            self.log.warning("Chunk text missing for %s; using placeholder.", cid)
            raw_text = f"[Text not available; len={int(meta.get('text_len') or 0)} chars]"

        # Instruction header
        header = (
            "You are a regulatory reviewer.\n"
            "Task: Given the requirement and ONE excerpt, classify coverage as exactly one of:\n"
            " - Covered\n - Partially Covered\n - Not Covered\n"
            "Base your decision ONLY on the excerpt (ignore outside knowledge).\n"
            "Return valid JSON with keys: verdict (string), confidence (1-5), rationale (<=80 tokens).\n\n"
            "Requirement:\n"
        )

        # Build context
        context = (header + requirement + "\n\nExcerpt:\n" + f"[Section: {sect} | {pages}]\n{raw_text}\n").strip()

        # Token budget report (+ warnings)
        # You can tune completion expectation by model; here we use self.prompt_budget as guidance if you prefer.
        completion_expect = 200
        tstats = self._token_budget_report(header, requirement, raw_text, completion_expect=completion_expect)
        if getattr(self, "log_tokens", False):
            self.log.info(
                "[TOKENS] prompt≈%d | header=%d (%.1f%%), req=%d (%.1f%%), chunk=%d (%.1f%%) | completion≈%d | total≈%d%s%s",
                tstats["prompt_tokens"],
                tstats["header_tokens"], tstats["pct_header_of_prompt"],
                tstats["requirement_tokens"], tstats["pct_requirement_of_prompt"],
                tstats["chunk_tokens"], tstats["pct_chunk_of_prompt"],
                tstats["completion_tokens_expect"],
                tstats["total_tokens_expect"],
                "  ⚠️" if tstats["warn_soft_exceeded"] else "",
                "  ❌" if tstats["warn_hard_exceeded"] else "",
            )


        snippet = {
            "chunk_id": cid,
            "section_title": sect,
            "pages": pages,
            "text_tokens_est": tstats["chunk_tokens"],

            # NEW: enrich snippet with detailed token stats
            "prompt_tokens": tstats["prompt_tokens"],
            "header_tokens": tstats["header_tokens"],
            "requirement_tokens": tstats["requirement_tokens"],
            "chunk_tokens": tstats["chunk_tokens"],
            "completion_tokens_expect": tstats["completion_tokens_expect"],
            "total_tokens_expect": tstats["total_tokens_expect"],
            "pct_header_of_prompt": tstats["pct_header_of_prompt"],
            "pct_requirement_of_prompt": tstats["pct_requirement_of_prompt"],
            "pct_chunk_of_prompt": tstats["pct_chunk_of_prompt"],
            "pct_completion_of_total": tstats["pct_completion_of_total"],
            "warn_soft_exceeded": tstats["warn_soft_exceeded"],
            "warn_hard_exceeded": tstats["warn_hard_exceeded"],
        }
        return context, snippet

    # ---------------- single-stage LLM ----------------

    def judge_chunk(self, requirement: str, context: str) -> dict:
        # Dry-run path: no LLM at all
        if self.dry_run:
            low = context.lower()
            verdict = "Not Covered"
            if any(k in low for k in [
                "indication", "intended use", "contraindicat", "steril", "mri",
                "warning", "precaution", "label", "ifu", "risk", "iso 14971",
                "62304", "biocompat"
            ]):
                verdict = "Partially Covered"
            return {"verdict": verdict, "confidence": 2, "rationale": "Heuristic (dry-run)."}

        # Real LLM path: lazy init here to avoid requiring key in dry-run
        if self.single_judge is None:
            if not _HAS_LLM or SingleStageJudge is None:
                raise RuntimeError("LLM judging unavailable. Install/openai SDK or run with --dry-run.")
            self.single_judge = SingleStageJudge(self.llm_model, logger=self.log)

        try:
            return self.single_judge.run(requirement=requirement, context=context)
        except Exception as e:
            self.log.error("Single-stage judge failed: %s", e)
            return {"verdict": "Not Covered", "confidence": 1, "rationale": "LLM error."}


    # ---------------- aggregation / reporting ----------------

    def _aggregate_chunk_verdicts(self, chunk_results: List[dict]) -> dict:
        """
        Priority: Covered > Partially Covered > Not Covered
        Confidence: mean of chunks at or above the final level
        Rationale: first concise rationale (optionally concatenated)
        """
        order = {"Covered": 2, "Partially Covered": 1, "Not Covered": 0}
        if not chunk_results:
            return {"verdict": "Not Covered", "confidence": 1, "rationale": "No chunk results."}

        # Best class by priority
        best = max(chunk_results, key=lambda r: order.get(r.get("verdict","Not Covered"), 0))
        final_label = best.get("verdict", "Not Covered")
        min_level = order.get(final_label, 0)

        # Confidence: average over chunks whose class >= final level
        confs = [r.get("confidence", 1) for r in chunk_results if order.get(r.get("verdict","Not Covered"),0) >= min_level]
        mean_conf = round(sum(confs)/len(confs), 1) if confs else 1.0

        # Rationale: take the first non-empty, or join top-2
        rats = [r.get("rationale","").strip() for r in chunk_results if r.get("rationale")]
        rationale = " ".join(rats[:2])
        if len(rationale) > 240:
            rationale = rationale[:240] + " …"

        return {
            "verdict": final_label,
            "confidence": mean_conf,
            "rationale": rationale or f"Aggregated from {len(chunk_results)} chunks."
        }

    def _save_item_row(
        self,
        item: dict,
        per_chunk: List[dict],
        snippets: List[dict],
        agg_verdict: dict,
        topn_used: int
    ) -> dict:
        out_obj = {
            "item_id": item["id"],
            "category": item["title"],
            "question": item["question"],
            "mode": "per_chunk_single_stage",
            "topn": topn_used,
            "per_chunk": per_chunk,   # [{chunk_id, verdict, confidence, rationale}]
            "snippets": snippets,     # [{chunk_id, section_title, pages, text_tokens_est}]
            "verdict": agg_verdict.get("verdict", "Not Covered"),
            "confidence": agg_verdict.get("confidence", 1),
            "rationale": agg_verdict.get("rationale", ""),
        }
        _write_jsonl_line(self.results_jsonl, out_obj)
        return out_obj

    def _aggregate(self, rows: List[dict]) -> dict:
        by_status = {}
        for r in rows:
            v = r.get("verdict", "Not Covered")
            by_status[v] = by_status.get(v, 0) + 1

        total = len(rows)
        pct = {k: (v / total if total else 0.0) for k, v in by_status.items()}

        return {
            "doc_id": self.doc_id,
            "total_items": total,
            "counts": by_status,
            "percents": {k: round(100 * v, 1) for k, v in pct.items()},
        }

    def _write_report_md(self, rows: List[dict], summary: dict) -> None:
        lines = []
        lines.append(f"# Compliance Assessment — {self.doc_id}\n")
        lines.append("## Summary\n")
        lines.append(f"- Total items: **{summary['total_items']}**")
        for k in sorted(summary["counts"].keys()):
            lines.append(f"- {k}: **{summary['counts'][k]}** ({summary['percents'][k]}%)")
        lines.append("")

        lines.append("## Results\n")
        lines.append("| ID | Category | Question | Verdict | Conf. | Notes |")
        lines.append("|---:|---|---|---|:---:|---|")
        for r in rows:
            lines.append(
                f"| {_md_escape(r['item_id'])} "
                f"| {_md_escape(r['category'])} "
                f"| {_md_escape(r['question'])} "
                f"| {_md_escape(r['verdict'])} "
                f"| {r.get('confidence', '')} "
                f"| {_md_escape(r.get('rationale',''))} |"
            )

        lines.append("\n## Diagnostics (per-item)\n")
        for r in rows:
            lines.append(f"### {r['item_id']} — {r['category']}")
            lines.append(f"**Question:** {r['question']}")
            lines.append(f"**Final verdict:** {r['verdict']} (conf: {r.get('confidence','')})")
            lines.append("**Per-chunk decisions:**")
            for pc in r.get("per_chunk", []):
                lines.append(f"- `{pc.get('chunk_id')}` → {pc.get('verdict')} (conf {pc.get('confidence', '')}) — {pc.get('rationale','')}")
            lines.append("")
        with open(self.report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ---------------- main ----------------

    def run_all(self) -> dict:
        self.log.info("Starting assessment for %s", self.doc_id)
        items = self.load_checklist()
        self.log.info("Loaded checklist items: %d", len(items))

        all_rows: List[dict] = []

        for item in items:
            qtext = item["question"]
            hits = self.retrieve(qtext)
            chosen = self.select_topn_strict(hits, self.topn_strict)

            per_chunk_rows = []
            snippets_meta = []

            for rank, (cid, score) in enumerate(chosen, start=1):
                # Build per-chunk micro-context (no trimming)
                context, snippet = self.build_single_chunk_context(qtext, cid)
                snippet["rank"] = rank
                snippet["score"] = float(score)
                snippets_meta.append(snippet)

                # Single-stage LLM judgment (or heuristic)
                result = self.judge_chunk(qtext, context)  # {"verdict": "...", "confidence": ..., "rationale": "..."}
                per_chunk_rows.append({
                    "chunk_id": cid,
                    "verdict": result.get("verdict", "Not Covered"),
                    "confidence": result.get("confidence", 1),
                    "rationale": result.get("rationale", "")
                })

            # Aggregate per-chunk → per-question
            agg = self._aggregate_chunk_verdicts(per_chunk_rows)

            # Save row
            row = self._save_item_row(
                item=item,
                per_chunk=per_chunk_rows,
                snippets=snippets_meta,
                agg_verdict=agg,
                topn_used=len(chosen),
            )
            all_rows.append(row)

        # Aggregate + report
        summary = self._aggregate(all_rows)
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._write_report_md(all_rows, summary)

        out = {
            "out_dir": os.path.abspath(self.out_dir),
            "results_jsonl": os.path.abspath(self.results_jsonl),
            "summary_json": os.path.abspath(self.summary_json),
            "report_md": os.path.abspath(self.report_md),
            "items": len(all_rows),
        }
        self.log.info("Assessment complete → %s", out["out_dir"])
        # after building all_rows and writing summary
        try:
            all_prompts = []
            for r in all_rows:
                for sn in r.get("snippets", []):
                    pt = sn.get("prompt_tokens")
                    if isinstance(pt, int):
                        all_prompts.append(pt)
            if all_prompts:
                mx = max(all_prompts); av = round(sum(all_prompts)/len(all_prompts), 1)
                msg = f"[TOKENS] Prompt est — max≈{mx}, avg≈{av}"
                if mx > WARN_PROMPT_RED:
                    self.log.error(_RED + msg + _RESET)
                elif mx > WARN_PROMPT_YELLOW:
                    self.log.warning(_YELLOW + msg + _RESET)
                else:
                    self.log.info(msg)
        except Exception:
            pass

        return out
