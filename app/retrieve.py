# app/retrieve.py
# -*- coding: utf-8 -*-
"""
Build + use BM25 and Embedding (FAISS) indices for chunk-level retrieval.

Config (config.yaml)
--------------------
index:
  bm25: true
  embedding_backend: auto         # one of: auto | sbert | openai
  defaults:
    sbert_model: sentence-transformers/all-MiniLM-L6-v2
    openai_model: text-embedding-3-small
  faiss_metric: ip                # "ip" or "l2"
  normalize: true                 # normalize vectors if ip (recommended)
  save_vectors: true
  batch_size: 64

Artifacts written to: data/index/<doc_id>/
  - mapping.jsonl         # chunk_id -> metadata (section, pages, etc.)
  - bm25.pkl              # tokens + idf/avgdl (if BM25 enabled)
  - bm25_meta.json        # metadata for BM25
  - vectors.npy           # (N, D) float32 embeddings
  - embed_meta.json       # backend/model/dim/metric/normalized
  - faiss.index           # FAISS index (if faiss installed)

APIs
----
IndexBuilder(run_dir, doc_id, index_root, cfg).build_all() -> Dict[str, str]
HybridSearcher(index_root, doc_id)
  .search_bm25(query, k=50)
  .search_faiss(query, k=50)     # uses FAISS if available
  .search_numpy(query, k=50)     # fallback if FAISS missing
  .search_hybrid(query, k_bm25=50, k_vec=50, topn_merge=50, w_bm25=1.0, w_vec=1.0)
"""

from __future__ import annotations

import os
import re
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

# ---------------- Optional deps (guarded imports) ----------------
try:
    import numpy as np
    _HAS_NP = True
except Exception:
    _HAS_NP = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ---------------- Small text utils ----------------
_STOP = set("""
a an and are as at be by for from has have in into is it its of on or than that the their this to was were will with without within
we they you he she i our your shall must should may can could would
""".split())

def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z]{2,}", (text or "").lower()) if t not in _STOP]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------- Embedding provider (pluggable) ----------------
class EmbeddingProvider:
    """
    Wraps embedding backends:
      - "sbert"  : sentence-transformers (CPU-friendly)
      - "openai" : OpenAI embeddings (requires OPENAI_API_KEY)
    Returns np.ndarray (N, D) float32.
    """
    def __init__(self, backend: str, model_name: str, batch_size: int = 64, logger: Optional[logging.Logger] = None):
        if not _HAS_NP:
            raise RuntimeError("numpy is required for embeddings.")
        self.backend = (backend or "sbert").lower()
        self.model_name = model_name
        self.batch = max(1, int(batch_size))
        self.log = logger or logging.getLogger(__name__)
        self.dim: Optional[int] = None

        self._sbert = None
        self._client = None
        self._openai_model = None

        if self.backend == "sbert":
            if not _HAS_SBERT:
                raise RuntimeError("sentence-transformers not installed; cannot use backend='sbert'.")
            self._sbert = SentenceTransformer(self.model_name)
            try:
                self.dim = int(self._sbert.get_sentence_embedding_dimension())
            except Exception:
                self.dim = None

        elif self.backend == "openai":
            if not _HAS_OPENAI:
                raise RuntimeError("openai package not installed; cannot use backend='openai'.")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set; cannot use backend='openai'.")
            self._client = OpenAI(api_key=api_key)
            self._openai_model = self.model_name

        else:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

    def encode(self, texts: List[str]) -> "np.ndarray":
        import numpy as _np

        if self.backend == "sbert":
            arr = self._sbert.encode(
                texts,
                batch_size=self.batch,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            arr = _np.asarray(arr, dtype="float32")
            if self.dim is None:
                self.dim = arr.shape[1]
            return arr

        elif self.backend == "openai":
            vecs: List[List[float]] = []
            for i in range(0, len(texts), self.batch):
                batch = texts[i:i + self.batch]
                resp = self._client.embeddings.create(model=self._openai_model, input=batch)
                vecs.extend([d.embedding for d in resp.data])
            arr = _np.asarray(vecs, dtype="float32")
            if self.dim is None:
                self.dim = arr.shape[1]
            return arr

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


# ---------------- Index Builder ----------------
class IndexBuilder:
    """
    Builds BM25 + Embedding indices for a given run/doc_id.
    """
    def __init__(self, run_dir: str, doc_id: str, index_root: str, cfg: dict) -> None:
        self.run_dir = os.path.abspath(run_dir)
        self.doc_id = doc_id
        self.index_dir = os.path.join(os.path.abspath(index_root), doc_id)
        _ensure_dir(self.index_dir)
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        # inputs
        self.chunks_path = os.path.join(self.run_dir, "chunks", f"{doc_id}_chunks.jsonl")
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks not found: {self.chunks_path}")

        # data
        self.chunks = _read_jsonl(self.chunks_path)
        self.doc_ids: List[str] = [c["chunk_id"] for c in self.chunks]
        self.texts: List[str] = [c.get("text", "") for c in self.chunks]
        self.tokens: List[List[str]] = [_simple_tokens(t) for t in self.texts]

        # outputs
        self.mapping_path = os.path.join(self.index_dir, "mapping.jsonl")

    def _resolve_backend_and_model(self) -> Tuple[str, str, int, bool, str]:
        """
        Resolve backend/model/params using config-only (the CLI resolves earlier).
        Returns: backend, model, batch, normalize, metric
        """
        idx = self.cfg.get("index", {}) if isinstance(self.cfg, dict) else {}
        backend = (idx.get("embedding_backend", "auto") or "auto").lower()
        defaults = idx.get("defaults", {}) if isinstance(idx.get("defaults", {}), dict) else {}
        default_sbert = defaults.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2")
        default_openai = defaults.get("openai_model", "text-embedding-3-small")
        # explicit model override in cfg wins
        model_name = idx.get("embedding_model")
        metric = (idx.get("faiss_metric", "ip") or "ip").lower()
        batch = int(idx.get("batch_size", 64))
        normalize = bool(idx.get("normalize", True if metric == "ip" else False))

        if backend == "auto":
            if os.getenv("OPENAI_API_KEY"):
                backend = "openai"
                model_name = model_name or default_openai
            else:
                backend = "sbert"
                model_name = model_name or default_sbert
        elif backend == "openai":
            model_name = model_name or default_openai
        else:
            backend = "sbert"
            model_name = model_name or default_sbert

        return backend, model_name, batch, normalize, metric

    def build_all(self) -> Dict[str, str]:
        paths: Dict[str, str] = {}

        # write mapping.jsonl (preserve order = row order)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({
                    "chunk_id": c["chunk_id"],
                    "section_id": c.get("section_id"),
                    "section_title": c.get("section_title"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "text_len": len(c.get("text", "")),
                    "run_dir": self.run_dir,
                    "doc_id": self.doc_id
                }) + "\n")
        paths["mapping"] = self.mapping_path

        # ---- BM25 (optional) ----
        idx_cfg = self.cfg.get("index", {}) if isinstance(self.cfg, dict) else {}
        if idx_cfg.get("bm25", True) and _HAS_BM25:
            bm25 = BM25Okapi(self.tokens)
            blob = {
                "doc_ids": self.doc_ids,
                "tokens": self.tokens,
                "idf": bm25.idf,
                "avgdl": bm25.avgdl,
            }
            with open(os.path.join(self.index_dir, "bm25.pkl"), "wb") as f:
                pickle.dump(blob, f)
            with open(os.path.join(self.index_dir, "bm25_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "type": "bm25",
                    "tokenizer": "simple_alpha_lower_stop",
                    "n_docs": len(self.doc_ids)
                }, f, indent=2)
            paths["bm25"] = os.path.join(self.index_dir, "bm25.pkl")
        elif not _HAS_BM25:
            self.log.warning("rank_bm25 not installed; skipping BM25 index.")

        # ---- Embeddings + FAISS ----
        if not _HAS_NP:
            self.log.warning("numpy not available; skipping embeddings/FAISS.")
            return paths

        backend, model_name, batch, normalize, metric = self._resolve_backend_and_model()
        prov = EmbeddingProvider(backend=backend, model_name=model_name, batch_size=batch, logger=self.log)
        vecs = prov.encode(self.texts).astype("float32")

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms

        dim = int(vecs.shape[1])
        if idx_cfg.get("save_vectors", True):
            np.save(os.path.join(self.index_dir, "vectors.npy"), vecs)
            paths["vectors"] = os.path.join(self.index_dir, "vectors.npy")

        with open(os.path.join(self.index_dir, "embed_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "type": "embeddings",
                "backend": backend,
                "model": model_name,
                "dim": dim,
                "normalized": normalize,
                "metric": metric,
                "n_docs": len(self.doc_ids)
            }, f, indent=2)
        paths["embed_meta"] = os.path.join(self.index_dir, "embed_meta.json")

        if not _HAS_FAISS:
            self.log.warning("faiss not installed; vectors saved but no FAISS index created.")
            return paths

        index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        index.add(vecs)
        faiss.write_index(index, os.path.join(self.index_dir, "faiss.index"))
        paths["faiss"] = os.path.join(self.index_dir, "faiss.index")
        return paths


# ---------------- Hybrid Searcher ----------------
class HybridSearcher:
    """
    Loads indices for a given <doc_id> and supports BM25, vector, and hybrid search.
    """
    def __init__(self, index_root: str, doc_id: str) -> None:
        self.index_dir = os.path.join(os.path.abspath(index_root), doc_id)
        self.doc_id = doc_id

        # mapping (keeps insertion order == row order)
        self.mapping: Dict[str, dict] = {}
        mp = os.path.join(self.index_dir, "mapping.jsonl")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Index mapping not found: {mp}")
        with open(mp, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.mapping[j["chunk_id"]] = j

        # BM25
        self._bm25 = None
        self._bm25_doc_ids: List[str] = []
        self._bm25_tokens: List[List[str]] = []
        if _HAS_BM25 and os.path.exists(os.path.join(self.index_dir, "bm25.pkl")):
            blob = pickle.load(open(os.path.join(self.index_dir, "bm25.pkl"), "rb"))
            self._bm25_tokens = blob["tokens"]
            self._bm25_doc_ids = blob["doc_ids"]
            self._bm25 = BM25Okapi(self._bm25_tokens)
            try:
                self._bm25.idf = blob["idf"]
                self._bm25.avgdl = blob["avgdl"]
            except Exception:
                pass

        # Embeddings / FAISS
        self._faiss = None
        self._vecs = None
        self._embed_meta = None
        self._metric = "ip"
        self._sbert = None
        self._openai_client = None
        self._openai_model = None

        if _HAS_NP and os.path.exists(os.path.join(self.index_dir, "vectors.npy")):
            self._vecs = np.load(os.path.join(self.index_dir, "vectors.npy"))
            self._embed_meta = _read_json(os.path.join(self.index_dir, "embed_meta.json"))
            self._metric = (self._embed_meta.get("metric") or "ip").lower()

            # FAISS index (if present)
            if _HAS_FAISS and os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                self._faiss = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))

            # prepare query encoder based on backend used at build time
            backend = (self._embed_meta.get("backend") or "sbert").lower()
            model_name = self._embed_meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            if backend == "sbert" and _HAS_SBERT:
                self._sbert = SentenceTransformer(model_name)
            elif backend == "openai" and _HAS_OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._openai_client = OpenAI(api_key=api_key)
                    self._openai_model = model_name

        # cached chunk_id list in consistent row order (dict preserves insertion order)
        self._row_cids = list(self.mapping.keys())

    # ----- BM25 -----
    def search_bm25(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        if not (self._bm25 and self._bm25_doc_ids):
            return []
        toks = _simple_tokens(query)
        scores = self._bm25.get_scores(toks)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._bm25_doc_ids[i], float(scores[i])) for i in idxs]

    # ----- Vector encoders -----
    def _encode_query(self, query: str) -> Optional["np.ndarray"]:
        if not _HAS_NP or self._embed_meta is None:
            return None
        backend = (self._embed_meta.get("backend") or "sbert").lower()

        if backend == "sbert" and self._sbert is not None:
            v = self._sbert.encode([query], normalize_embeddings=False)[0].astype("float32")
        elif backend == "openai" and self._openai_client is not None and self._openai_model:
            resp = self._openai_client.embeddings.create(model=self._openai_model, input=[query])
            v = np.asarray(resp.data[0].embedding, dtype="float32")
        else:
            return None

        if self._embed_meta.get("normalized", False):
            n = np.linalg.norm(v) + 1e-12
            v = v / n
        return v.reshape(1, -1)

    # ----- Vector search -----
    def search_faiss(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        if self._faiss is None or self._vecs is None:
            return []
        v = self._encode_query(query)
        if v is None:
            return []
        D, I = self._faiss.search(v, min(k, len(self._vecs)))
        out: List[Tuple[str, float]] = []
        for d, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            out.append((self._row_cids[int(idx)], float(d)))
        return out

    def search_numpy(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        if self._vecs is None:
            return []
        v = self._encode_query(query)
        if v is None:
            return []
        if self._metric == "ip":
            sims = (self._vecs @ v.T).ravel()
            idxs = np.argsort(-sims)[:k]
            return [(self._row_cids[int(i)], float(sims[int(i)])) for i in idxs]
        else:
            d = ((self._vecs - v) ** 2).sum(axis=1)
            idxs = np.argsort(d)[:k]
            return [(self._row_cids[int(i)], float(-d[int(i)])) for i in idxs]  # negative distance as score

    # ----- Hybrid merge -----
    def search_hybrid(
        self,
        query: str,
        k_bm25: int = 50,
        k_vec: int = 50,
        topn_merge: int = 50,
        w_bm25: float = 1.0,
        w_vec: float = 1.0,
    ) -> List[Tuple[str, float]]:
        bm = self.search_bm25(query, k_bm25) if self._bm25 else []
        ve = self.search_faiss(query, k_vec)
        if not ve:  # FAISS missing or no index → numpy fallback
            ve = self.search_numpy(query, k_vec)

        def _norm(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
            if not pairs:
                return {}
            vals = [s for _, s in pairs]
            lo, hi = min(vals), max(vals)
            if hi <= lo:
                return {cid: 1.0 for cid, _ in pairs}
            return {cid: (s - lo) / (hi - lo + 1e-12) for cid, s in pairs}

        bmN = _norm(bm)
        veN = _norm(ve)

        agg = Counter()
        for cid, s in bmN.items():
            agg[cid] += w_bm25 * s
        for cid, s in veN.items():
            agg[cid] += w_vec * s

        merged = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topn_merge]
        return [(cid, float(score)) for cid, score in merged]
