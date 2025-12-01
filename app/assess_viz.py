# app/assess_viz.py
# -*- coding: utf-8 -*-
"""
Assessment visualization: reads results.jsonl / summary.json produced by assess.py
and generates a Plotly HTML dashboard + a compact markdown summary.

Outputs (per doc_id):
data/runs/<run>/assess/<doc_id>/viz/
  ├─ assessment_dashboard.html
  └─ assessment_summary.md
"""

from __future__ import annotations
import os, json, math, glob
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _load_results(run_dir: str, doc_id: str) -> pd.DataFrame:
    assess_dir = os.path.join(run_dir, "assess", doc_id)
    r_path = os.path.join(assess_dir, "results.jsonl")
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"results.jsonl not found: {r_path}")
    rows = []
    with open(r_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    if not rows:
        raise RuntimeError("results.jsonl is empty.")
    df = pd.DataFrame(rows)
    return df

def _load_summary(run_dir: str, doc_id: str) -> Dict[str, Any]:
    s_path = os.path.join(run_dir, "assess", doc_id, "summary.json")
    if not os.path.exists(s_path):
        return {}
    with open(s_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _explode_per_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # Expand per-chunk rows to a flat frame
    base_cols = ["item_id", "category", "question", "verdict", "confidence", "topn"]
    df_base = df[base_cols].copy()
    df_chunks = df[["per_chunk", "snippets"]].copy()

    # Normalize lists
    pc = df_chunks["per_chunk"].apply(lambda x: x if isinstance(x, list) else [])
    sn = df_chunks["snippets"].apply(lambda x: x if isinstance(x, list) else [])

    # Align lengths (if any mismatch)
    maxlen = np.maximum(pc.apply(len), sn.apply(len))
    # Safest: iterate rows
    flat_rows = []
    for i, (pch, snp, base) in enumerate(zip(pc, sn, df_base.to_dict("records"))):
        for j in range(max(len(pch), len(snp))):
            pr = pch[j] if j < len(pch) else {}
            sr = snp[j] if j < len(snp) else {}
            flat_rows.append({
                **base,
                "chunk_rank": sr.get("rank"),
                "chunk_score": sr.get("score"),
                "chunk_id": pr.get("chunk_id") or sr.get("chunk_id"),
                "chunk_section": sr.get("section_title"),
                "chunk_pages": sr.get("pages"),
                "chunk_text_tokens": sr.get("text_tokens_est"),
                "chunk_verdict": pr.get("verdict"),
                "chunk_confidence": pr.get("confidence"),
                "chunk_rationale": pr.get("rationale"),
            })
    return pd.DataFrame(flat_rows)

def _coverage_color_map() -> Dict[str, str]:
    return {
        "Covered": "#2ca02c",            # green
        "Partially Covered": "#ff7f0e",  # orange
        "Not Covered": "#d62728",        # red
        "Information Not Found": "#7f7f7f"  # grey (if ever present)
    }

def _mk_coverage_pie(df: pd.DataFrame) -> go.Figure:
    by_v = df["verdict"].value_counts().reset_index()
    by_v.columns = ["verdict", "count"]
    cmap = _coverage_color_map()
    fig = px.pie(by_v, names="verdict", values="count",
                 color="verdict", color_discrete_map=cmap,
                 title="Checklist Verdict Distribution")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def _mk_confidence_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="confidence", nbins=5, title="Final Confidence (per item)")
    fig.update_layout(bargap=0.1)
    return fig

def _mk_retrieval_scatter(df_chunks: pd.DataFrame) -> go.Figure:
    # chunk_score vs chunk_rank, colored by chunk_verdict
    tmp = df_chunks.dropna(subset=["chunk_score", "chunk_rank"]).copy()
    if tmp.empty:
        return go.Figure()
    cmap = _coverage_color_map()
    fig = px.scatter(
        tmp,
        x="chunk_rank", y="chunk_score",
        color="chunk_verdict",
        hover_data=["item_id", "chunk_id", "chunk_section", "chunk_pages"],
        color_discrete_map=cmap,
        title="Retrieval Quality: Score vs Rank (per chunk)"
    )
    fig.update_xaxes(title="Rank (1 = best)")
    fig.update_yaxes(title="Hybrid score")
    return fig

def _mk_section_heatmap(df_chunks: pd.DataFrame, top_k_sections: int = 20) -> go.Figure:
    # items (rows) vs sections (columns): count of selected chunks
    tmp = df_chunks.copy()
    tmp["chunk_section"] = tmp["chunk_section"].fillna("(unknown)")
    # pick top sections by frequency
    top_sections = tmp["chunk_section"].value_counts().head(top_k_sections).index.tolist()
    tmp = tmp[tmp["chunk_section"].isin(top_sections)]
    if tmp.empty:
        return go.Figure()
    pivot = pd.pivot_table(
        tmp,
        index="item_id",
        columns="chunk_section",
        values="chunk_id",
        aggfunc="count",
        fill_value=0
    )
    fig = px.imshow(
        pivot.values,
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale='Blues',
        aspect="auto",
        title=f"Where evidence lives: Items × Sections (top {len(top_sections)})"
    )
    fig.update_coloraxes(colorbar_title="#chunks")
    return fig

def _mk_item_bar(df: pd.DataFrame) -> go.Figure:
    # bar per item_id with verdict color + confidence as text
    ordered = df.sort_values(["verdict", "confidence"], ascending=[True, False]).copy()
    cmap = _coverage_color_map()
    fig = px.bar(
        ordered,
        x="item_id",
        y=[1]*len(ordered),  # constant height, we color by verdict
        color="verdict",
        hover_data=["question", "confidence", "category", "rationale"],
        color_discrete_map=cmap,
        title="Per-Item Verdict (ordered by class, conf desc)"
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=True)
    return fig

def _render_html(figs: List[go.Figure], titles: List[str], out_html: str) -> None:
    # Compose a very simple HTML with multiple figures stacked vertically
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Assessment Dashboard</title></head><body>",
        "<h1>Assessment Dashboard</h1>"
    ]
    for title, fig in zip(titles, figs):
        html_parts.append(f"<h2>{title}</h2>")
        if fig and len(fig.data) > 0:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        else:
            html_parts.append("<p><em>No data available.</em></p>")
    html_parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

def _render_summary_md(df_items: pd.DataFrame, df_chunks: pd.DataFrame, summary: Dict[str, Any], out_md: str) -> None:
    lines = []
    lines.append(f"# Assessment Summary — {df_items.get('doc_id',[None])[0] or ''}\n")
    if summary:
        lines.append("## Totals")
        lines.append(f"- Items: **{summary.get('total_items','?')}**")
        counts = summary.get("counts", {})
        pcts = summary.get("percents", {})
        for k in ["Covered", "Partially Covered", "Not Covered", "Information Not Found"]:
            if k in counts:
                lines.append(f"- {k}: **{counts[k]}** ({pcts.get(k, 0)}%)")
        lines.append("")
    lines.append("## Notes")
    lines.append("- *Confidence* is the aggregated value per item (mean over supporting chunks at/above final level).")
    lines.append("- *Retrieval scatter* helps spot if high-rank chunks have low scores or vice versa.")
    lines.append("- *Section heatmap* shows where evidence clusters (useful for reviewer navigation).")
    lines.append("")
    lines.append("## Per-item table (first 20)")
    head = df_items[["item_id", "category", "verdict", "confidence", "question"]].head(20)
    lines.append(head.to_markdown(index=False))
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

class AssessVisualizer:
    """
    Build a compact dashboard from assessment outputs.
    """
    def __init__(self, run_dir: str, doc_id: str):
        self.run_dir = os.path.abspath(run_dir)
        self.doc_id = doc_id
        self.assess_dir = os.path.join(self.run_dir, "assess", self.doc_id)
        self.out_dir = os.path.join(self.assess_dir, "viz")
        _ensure_dir(self.out_dir)

        # paths
        self.results_path = os.path.join(self.assess_dir, "results.jsonl")
        self.summary_path = os.path.join(self.assess_dir, "summary.json")
        self.dashboard_html = os.path.join(self.out_dir, "assessment_dashboard.html")
        self.summary_md = os.path.join(self.out_dir, "assessment_summary.md")

    def run(self) -> Dict[str, str]:
        df_items = _load_results(self.run_dir, self.doc_id)
        df_items["doc_id"] = self.doc_id
        summary = _load_summary(self.run_dir, self.doc_id)

        # Explode per-chunk rows
        df_chunks = _explode_per_chunk(df_items)

        # Figures
        figs = []
        titles = []

        fig_pie = _mk_coverage_pie(df_items)
        figs.append(fig_pie); titles.append("Verdict Distribution")

        fig_conf = _mk_confidence_hist(df_items)
        figs.append(fig_conf); titles.append("Confidence Histogram")

        fig_scatter = _mk_retrieval_scatter(df_chunks)
        figs.append(fig_scatter); titles.append("Retrieval Score vs Rank")

        fig_heat = _mk_section_heatmap(df_chunks, top_k_sections=20)
        figs.append(fig_heat); titles.append("Items × Sections (Heatmap)")

        fig_items = _mk_item_bar(df_items)
        figs.append(fig_items); titles.append("Per-Item Verdict Overview")

        _render_html(figs, titles, self.dashboard_html)
        _render_summary_md(df_items, df_chunks, summary, self.summary_md)

        return {
            "dashboard_html": os.path.abspath(self.dashboard_html),
            "summary_md": os.path.abspath(self.summary_md),
        }
