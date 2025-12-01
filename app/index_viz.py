# app/index_viz.py
# -*- coding: utf-8 -*-
"""
Index-stage dashboard (advanced yet compact):
- Embedding health: norms, pairwise cosine (anisotropy), participation ratio (intrinsic dimension)
- kNN diagnostics: distance profile, hubness (k-occurrence across kNN lists)
- 2D PCA map (colored by section)
- Section coverage: chunks per section/page span
- BM25/token signals: doc length, IDF variance (if available)

Inputs (from data/index/<doc_id>/):
  - vectors.npy
  - embed_meta.json
  - mapping.jsonl
  - bm25.pkl (optional)
Outputs:
  - data/index/<doc_id>/viz/index_dashboard.html
  - data/index/<doc_id>/viz/index_summary.md
"""

from __future__ import annotations

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

import os, json, pickle, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# light deps only
import numpy as np

# Optional plotting + ML
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from sklearn.decomposition import PCA
    _HAS_SK = True
except Exception:
    _HAS_SK = False


@dataclass
class MapRow:
    chunk_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    text_len: int

def _mpl_png_uri(draw_fn) -> str:
    if not _HAS_MPL:
        return ""
    fig = plt.figure(figsize=(5.5, 3.2), dpi=140)
    ax = fig.add_subplot(111)
    try:
        draw_fn(fig, ax)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    finally:
        plt.close(fig)



def _read_json(p:str)->dict:
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def _read_jsonl(p:str)->List[dict]:
    out=[]
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if ln: out.append(json.loads(ln))
    return out

def _ensure_dir(p:str)->None:
    os.makedirs(p, exist_ok=True)

def _notice(msg:str)->str:
    return f"<div style='padding:10px;border:1px dashed #999;background:#fafafa'>{msg}</div>"

def _participation_ratio(vecs: np.ndarray) -> float:
    """Estimate intrinsic dimensionality via spectrum PR (PCA)."""
    if vecs.shape[0] < 5:
        return float(vecs.shape[1])
    k = min(128, vecs.shape[1], vecs.shape[0]-1)
    pca = PCA(n_components=k, svd_solver="randomized")
    s = pca.fit(vecs).explained_variance_
    num = (s.sum()**2)
    den = (s**2).sum() + 1e-12
    return float(num/den)

def _cosine_anisotropy(vecs: np.ndarray, sample: int = 2000) -> Tuple[float,float]:
    """Mean & std of cosine similarity on a sample (higher mean => more anisotropy)."""
    n = vecs.shape[0]
    if n < 3: return 0.0, 0.0
    idx = np.random.choice(n, size=min(sample, n), replace=False)
    X = vecs[idx]
    sims = (X @ X.T)
    iu = np.triu_indices(X.shape[0], k=1)
    vals = sims[iu]
    return float(vals.mean()), float(vals.std())

def _knn_dist_profile(vecs: np.ndarray, k: int = 10, sample:int=2000) -> Tuple[np.ndarray,np.ndarray]:
    """Return mean/median k-th neighbor distance across a sample (IP assumed if normalized)."""
    n = vecs.shape[0]
    m = min(sample, n)
    idx = np.random.choice(n, size=m, replace=False)
    X = vecs[idx]
    # cosine/IP → use (1 - sim) as distance proxy (sorted descending sim)
    sims = X @ vecs.T
    sims.sort(axis=1)         # ascending
    sims = sims[:, -k-1:-1]   # top-k (skip self at last col)
    dists = 1.0 - sims        # 1 - cosine
    mean_k = dists.mean(axis=0)
    med_k  = np.median(dists, axis=0)
    return mean_k, med_k

def _hubness(vecs: np.ndarray, k: int = 10, sample:int=10000) -> Tuple[float, np.ndarray]:
    """
    Hubness: count how often a point appears in others' kNN lists.
    Returns (gini, counts_dist)
    """
    n = vecs.shape[0]
    m = min(sample, n)
    idx = np.random.choice(n, size=m, replace=False)
    X = vecs[idx]
    sims = X @ vecs.T
    nn_idx = np.argpartition(-sims, kth=range(1,k+1), axis=1)[:, :k]  # top-k approx
    counts = np.zeros(n, dtype=np.int32)
    for row in nn_idx:
        counts[row] += 1
    # Gini coefficient of hubness counts
    c = counts.astype(np.float64)
    if c.sum() == 0: return 0.0, counts
    c_sorted = np.sort(c)
    i = np.arange(1, n+1)
    gini = (np.sum((2*i - n - 1) * c_sorted) / (n * np.sum(c))) if n>0 else 0.0
    gini = float(gini)
    return gini, counts

class IndexVisualizer:
    def __init__(self, index_root: str, doc_id: str) -> None:
        self.index_dir = os.path.join(os.path.abspath(index_root), doc_id)
        self.doc_id = doc_id
        self.viz_dir = os.path.join(self.index_dir, "viz")
        _ensure_dir(self.viz_dir)

        self.mapping_path = os.path.join(self.index_dir, "mapping.jsonl")
        self.embed_meta_path = os.path.join(self.index_dir, "embed_meta.json")
        self.vecs_path = os.path.join(self.index_dir, "vectors.npy")
        self.bm25_path = os.path.join(self.index_dir, "bm25.pkl")

        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"mapping.jsonl not found: {self.mapping_path}")
        if not os.path.exists(self.embed_meta_path):
            raise FileNotFoundError(f"embed_meta.json not found: {self.embed_meta_path}")
        if not os.path.exists(self.vecs_path):
            raise FileNotFoundError(f"vectors.npy not found: {self.vecs_path}")

        self.embed_meta = _read_json(self.embed_meta_path)
        self.vecs: np.ndarray = np.load(self.vecs_path)          # (N, D)
        self.maps_raw = _read_jsonl(self.mapping_path)
        self.maps: List[MapRow] = [
            MapRow(
                chunk_id=m.get("chunk_id",""),
                section_id=str(m.get("section_id","")),
                section_title=m.get("section_title","") or "",
                page_start=int(m.get("page_start") or 1),
                page_end=int(m.get("page_end") or m.get("page_start") or 1),
                text_len=int(m.get("text_len") or 0),
            ) for m in self.maps_raw
        ]
        self.has_bm25 = os.path.exists(self.bm25_path)

    # ---------------- panels ----------------
    def _panel_meta(self) -> str:
        rows = [
            ("Doc ID", self.doc_id),
            ("#Chunks", f"{len(self.maps):,}"),
            ("Embeddings", f"{self.embed_meta.get('backend')} / {self.embed_meta.get('model')}"),
            ("Dim", str(self.embed_meta.get('dim'))),
            ("Metric", self.embed_meta.get('metric')),
            ("Normalized", str(self.embed_meta.get('normalized'))),
            ("BM25", "yes" if self.has_bm25 else "no"),
        ]
        html = ["<table class='kvt'><tbody>"]
        for k,v in rows:
            html.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html.append("</tbody></table>")
        return "".join(html)

    def _panel_pca(self) -> str:
        if not (_HAS_PLOTLY and _HAS_SK):
            return _notice("Install plotly and scikit-learn for the PCA map.")
        n = self.vecs.shape[0]
        samp = min(5000, n)
        idx = np.random.choice(n, size=samp, replace=False)
        X = self.vecs[idx]
        sect = [self.maps[i].section_id for i in idx]
        pca = PCA(n_components=2, svd_solver="randomized").fit_transform(X)
        fig = go.Figure(go.Scattergl(
            x=pca[:,0], y=pca[:,1], mode="markers",
            marker=dict(size=4, opacity=0.7),
            text=[f"{self.maps[i].section_id}: {self.maps[i].section_title}" for i in idx]
        ))
        fig.update_layout(title="PCA map of chunks (sampled)", xaxis_title="PC1", yaxis_title="PC2")
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})

    def _panel_anisotropy(self) -> str:
        mean_cos, std_cos = _cosine_anisotropy(self.vecs)
        pr = _participation_ratio(self.vecs)
        expl = (
            "<ul>"
            f"<li><b>Cosine mean</b> ≈ {mean_cos:.3f} (higher ⇒ more anisotropy / less angular spread)</li>"
            f"<li><b>Cosine std</b> ≈ {std_cos:.3f}</li>"
            f"<li><b>Participation ratio</b> ≈ {pr:.1f} dimensions (intrinsic)</li>"
            "</ul>"
            "<p><i>Interpretation:</i> Highly anisotropic embeddings (high mean cosine) may hurt recall; "
            "PR gives a sense of effective dimension vs model dim.</p>"
        )
        return expl

    def _panel_knn(self) -> str:
        if not _HAS_PLOTLY:
            return _notice("Install plotly for kNN diagnostics.")
        mean_k, med_k = _knn_dist_profile(self.vecs, k=10)
        x = list(range(1, len(mean_k)+1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=mean_k.tolist(), mode="lines+markers", name="Mean 1-cos(k)"))
        fig.add_trace(go.Scatter(x=x, y=med_k.tolist(), mode="lines+markers", name="Median 1-cos(k)"))
        fig.update_layout(title="kNN distance profile (lower is tighter clusters)", xaxis_title="k", yaxis_title="1 - cosine")
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})

    def _panel_hubness(self) -> str:
        if not _HAS_PLOTLY:
            return _notice("Install plotly for hubness plot.")
        gini, counts = _hubness(self.vecs, k=10)
        hist = np.histogram(counts, bins=min(50, max(5, int(np.sqrt(len(counts)))) ))
        x = hist[1][:-1].tolist(); y = hist[0].tolist()
        fig = go.Figure(go.Bar(x=x, y=y))
        fig.update_layout(title=f"Hubness counts (Gini ≈ {gini:.3f})", xaxis_title="kNN appearances", yaxis_title="#Chunks")
        interpret = (
            f"<p>Hubness measures how often some points become neighbors of many others. "
            f"Higher Gini ⇒ few hubs dominate retrieval, which can harm diversity.</p>"
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}) + interpret

    def _panel_sections(self) -> str:
        # chunks per section & avg text_len
        from collections import defaultdict
        cnt = defaultdict(int); tl = defaultdict(list); pages = defaultdict(set)
        for m in self.maps:
            cnt[m.section_id] += 1
            tl[m.section_id].append(m.text_len)
            for p in range(m.page_start, m.page_end+1):
                pages[m.section_id].add(p)
        labels = []
        ch_counts = []
        pg_spans = []
        avg_len = []
        for sid in cnt:
            labels.append(f"{sid}")
            ch_counts.append(cnt[sid])
            pg_spans.append(len(pages[sid]))
            avg_len.append(int(np.mean(tl[sid]) if tl[sid] else 0))
        if not _HAS_PLOTLY:
            return _notice("Install plotly for section coverage.")
        fig = go.Figure()
        fig.add_bar(x=labels, y=ch_counts, name="Chunks")
        fig.add_bar(x=labels, y=pg_spans, name="Pages spanned")
        fig.update_layout(barmode="group", title="Section coverage", xaxis_title="Section", yaxis_title="Count")
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})
    
    
    def _panel_bm25(self) -> str:
        if not self.has_bm25:
            return _notice("BM25 index not present.")
        try:
            blob = pickle.load(open(self.bm25_path, "rb"))
            avgdl = blob.get("avgdl", None)
            idf = blob.get("idf", None)

            # Normalize idf to a list of floats
            vals = None
            terms = None
            if isinstance(idf, dict):
                terms, vals = zip(*idf.items()) if idf else ([], [])
                vals = list(vals)
                terms = list(terms)
            elif isinstance(idf, (list, tuple, np.ndarray)):
                vals = list(idf)
            else:
                return _notice("BM25 loaded, but IDF field is missing or unknown format.")

            if not vals:
                return _notice("BM25 loaded, but IDF is empty (check tokenizer / stopword settings).")

            title = f"BM25 IDF distribution (avgdl={avgdl:.1f})" if isinstance(avgdl, (int, float)) else "BM25 IDF distribution"

            # Prefer Plotly
            if _HAS_PLOTLY:
                import plotly.graph_objects as go
                fig = go.Figure(go.Histogram(x=vals, nbinsx=50))
                fig.update_layout(title=title, xaxis_title="IDF", yaxis_title="Frequency")
                html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})
            # Fallback: Matplotlib → inline PNG
            elif "_HAS_MPL" in globals() and _HAS_MPL:
                def draw(fig, ax):
                    ax.hist(vals, bins=50)
                    ax.set_title(title)
                    ax.set_xlabel("IDF"); ax.set_ylabel("Frequency")
                uri = _mpl_png_uri(draw)
                html = f"<img alt='BM25 IDF' src='{uri}'/>" if uri else ""
            else:
                html = ""  # no plotting libs; fall through to stats

            # Always add summary stats + top/bottom examples (helpful even with plots)
            arr = np.asarray(vals, dtype=float)
            stats = (
                f"<ul>"
                f"<li>min={arr.min():.3f}, median={np.median(arr):.3f}, mean={arr.mean():.3f}, max={arr.max():.3f}</li>"
                f"</ul>"
            )

            examples = ""
            if isinstance(idf, dict) and terms:
                # Show a few rare/common terms for intuition
                order = np.argsort(arr)  # ascending by idf
                low_terms = [terms[i] for i in order[:5]]
                high_terms = [terms[i] for i in order[-5:]]
                examples = (
                    "<div style='font-size:12px;color:#666'>"
                    f"<b>Common (low IDF):</b> {', '.join(low_terms)}<br>"
                    f"<b>Rare (high IDF):</b> {', '.join(high_terms)}"
                    "</div>"
                )

            return (html or "") + f"<p><b>{title}</b></p>" + stats + examples

        except Exception as e:
            return _notice(f"Could not load BM25 stats: {e}")


    # ---------------- compose & run ----------------
    def _compose(self, meta_html, pca_html, aniso_html, knn_html, hub_html, sect_html, bm25_html) -> str:
        css = """
        <meta charset="utf-8">
        <style>
        :root {
            --bg: #ffffff; --fg:#111; --muted:#666; --card:#fff; --border:#e6e6e6; --shadow:rgba(0,0,0,.04);
        }
        @media (prefers-color-scheme: dark) {
            :root { --bg:#0f1114; --fg:#e9edf1; --muted:#a6adbb; --card:#171a1f; --border:#2a2f38; --shadow:rgba(0,0,0,.25); }
        }
        html, body { background: var(--bg); color: var(--fg); }
        body { margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
        header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom:1px solid var(--border); padding: 14px 18px;}
        h1 { font-size: 20px; margin: 0; }
        .sub { color: var(--muted); margin-top: 4px; font-size: 13px; }
        main { padding: 16px 18px 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
        .card { background: var(--card); border:1px solid var(--border); border-radius: 12px; box-shadow: 0 1px 3px var(--shadow); }
        .card h3 { margin: 0; padding: 12px 14px; font-size: 15px; border-bottom:1px solid var(--border); }
        .card .body { padding: 12px 14px; }
        table.kvt { width: 100%; border-collapse: collapse; }
        table.kvt th, table.kvt td { padding: 4px 6px; vertical-align: top; }
        table.kvt th { text-align: left; color: var(--muted); width: 160px; }
        img { max-width:100%; height:auto; display:block; }
        .notice { padding: 10px; border:1px dashed var(--border); color: var(--muted); border-radius:8px; }
        </style>
        """
        def card(title, inner_html): return f"<div class='card'><h3>{title}</h3><div class='body'>{inner_html}</div></div>"

        head = f"""
        <header>
        <h1>Index Dashboard — {self.doc_id}</h1>
        <div class='sub'>Embedding/BM25 diagnostics with compact research signals</div>
        </header>
        """
        grid = (
        "<main><div class='grid'>"
        + card("Index Metadata", meta_html)
        + card("Anisotropy & Intrinsic Dimension", aniso_html)
        + card("kNN Distance Profile", knn_html)
        + card("Hubness", hub_html)
        + card("PCA Map", pca_html)
        + card("Section Coverage", sect_html)
        + card("BM25 / Token Signals", bm25_html)
        + "</div></main>"
        )
        return css + head + grid


    def _summary_md(self) -> str:
        mean_cos, std_cos = _cosine_anisotropy(self.vecs)
        pr = _participation_ratio(self.vecs)
        lines = [
            f"# Index Summary — {self.doc_id}",
            "",
            f"- Chunks: **{len(self.maps)}**",
            f"- Embeddings: **{self.embed_meta.get('backend')} / {self.embed_meta.get('model')}**",
            f"- Dim: **{self.embed_meta.get('dim')}**, Normalized: **{self.embed_meta.get('normalized')}**, Metric: **{self.embed_meta.get('metric')}**",
            f"- Cosine mean/std: **{mean_cos:.3f} / {std_cos:.3f}**",
            f"- Participation ratio (intrinsic dim): **{pr:.1f}**",
            "",
            "### Interpretation",
            "- Higher cosine mean ⇒ more anisotropy; can hurt recall diversity.",
            "- Lower kNN distances ⇒ tighter clusters; good for precision but watch for hubs.",
            "- Higher hubness Gini ⇒ few points dominate neighbors; consider rebalancing or reducing overlap.",
        ]
        return "\n".join(lines)

    def run(self) -> Dict[str,str]:
        meta_html = self._panel_meta()
        pca_html = self._panel_pca()
        aniso_html = self._panel_anisotropy()
        knn_html = self._panel_knn()
        hub_html = self._panel_hubness()
        sect_html = self._panel_sections()
        bm25_html = self._panel_bm25()

        html = self._compose(meta_html, pca_html, aniso_html, knn_html, hub_html, sect_html, bm25_html)

        out_html = os.path.join(self.viz_dir, "index_dashboard.html")
        out_md   = os.path.join(self.viz_dir, "index_summary.md")
        #with open(out_html, "w", encoding="utf-8") as f: f.write(html)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        with open(out_md, "w", encoding="utf-8") as f: f.write(self._summary_md())
        return {"dashboard_html": os.path.abspath(out_html), "summary_md": os.path.abspath(out_md)}
