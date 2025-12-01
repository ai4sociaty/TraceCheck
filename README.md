# TraceCheck — Automated Document Compliance Verification

A production-ready reference implementation that assists RA/QA professionals in verifying whether a **medical device technical document** covers a **compliance checklist**. It ingests large PDFs, builds **hybrid retrieval** (BM25 + embeddings), and runs **single-stage per-chunk LLM judging** to classify each checklist question as **Covered / Partially Covered / Not Covered**, with supporting evidence and a clean report — all under strict **per-call token limits**.

---

## Key Features

- **Robust ingestion** of complex PDFs
  - Primary parser: **Docling** (layout-aware, optional OCR).
  - Fallback: **PyMuPDF** (fast, light).
  - Extracts headings/sections, tables/captions (optional), and **splits into overlapping chunks** (configurable by characters).
- **Hybrid retrieval**
  - **BM25** lexical + **FAISS** vector search; **hybrid scoring** and strict top-N selection.
  - Embeddings backends: **OpenAI** or **SBERT (CPU)**.
- **Assessment (LLM)**
  - Single-stage, **per-chunk judgment** with JSON outputs (verdict, confidence, rationale).
  - Token budgeting with **soft/hard thresholds** (warnings).
  - **Dry-run** heuristic available for debugging (optional).
- **Visual QA**
  - Ingest dashboard (structure, sections, pages, figures/tables, OCR density).
  - Index dashboard (IDF, embedding anisotropy, nearest neighbors).
  - Assess dashboard (coverage distribution, per-item evidence).
- **Deliverables**
  - `results.jsonl`, `summary.json`, and `report.md` per document.
  - Streamlit app to upload PDF and **run the full pipeline**.
- **Reproducible CLI**
  - Single-file `cli.py` with subcommands: `ingest`, `index`, `assess`, `viz-*`, `report`.
  - Optional `cli_runall.py` to execute everything in one shot.

---

## Project Structure

```
TraceCheck/
├─ README.md                        # ← you are here
├─ requirements.txt
├─ config.yaml                      # central configuration (models, tokens, flags, paths)
├─ documents/
│  ├─ Compliance_checklist.Json
│  └─ PDFs/                         # drop your PDFs here (used by CLI) 
├─ data/
│  ├─ index/                        # FAISS/BM25 indices + mappings
│  └─ runs/                         # run_YYYYMMDD_HHMM/

├─ app/
│  ├─ __init__.py
│  ├─ io.py                         # parsing, OCR, sectioning, chunking
│  ├─ retrieve.py                   # BM25/FAISS build & hybrid search
│  ├─ assess.py                     # single-stage judging & aggregation
│  ├─ llm.py                        # OpenAI wrapper + token logging/guards
│  ├─ ingest_viz.py                 # ingest QA dashboard generator
│  ├─ index_viz.py                  # index QA dashboard generator
│  ├─ assess_viz.py                 # assess QA dashboard generator
│  ├─ report.py                     # compile Markdown/PDF summary
│  └─ utils.py                      # helpers (yaml, tokens, ids, logging)
├─ cli.py                           # CLI: ingest/index/assess/report/viz-*
└─ web/
   └─ app.py                        # Streamlit front-end, calls CLI underneath
```

> **Token rule in assignment:** The brief mentions both 1000 and 1500 tokens. We treat **1000 as the soft limit** and **1500 as the hard cap**. Each **individual LLM call** (prompt+completion) should stay under the **hard limit**.

---

## Prerequisites

- Python **3.11** recommended.
- macOS / Linux / Windows.
- Optional GPU:
  - Apple Silicon `mps` works for Docling OCR & SBERT.
  - CPU-only works across the stack; SBERT runs fine on CPU for this scale.
- For OpenAI:
  - Set environment variable: `export OPENAI_API_KEY=sk-...`

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

If FAISS/PyYAML issues arise:

```bash
pip install faiss-cpu
pip install PyYAML
```

---

## Quickstart (CLI)

1) **Drop PDFs** in `documents/PDFs/` or point to them via `config.yaml`.
2) **Ingest** (parse + chunk):

```bash
python cli.py ingest --config config.yaml
```

3) **Index** (BM25 + embeddings; choose backend):

```bash
# OpenAI embeddings (requires OPENAI_API_KEY)
python cli.py index --config config.yaml --backend openai --model text-embedding-3-small  --doc "<DOC_ID>" 

# SBERT embeddings (CPU-friendly)
python cli.py index --config config.yaml --backend sbert --model sentence-transformers/all-MiniLM-L6-v2  --doc "<DOC_ID>" 
```

4) **Assess** (LLM judging, strict top-N hybrid hits; with token logs):

```bash
python cli.py assess --config config.yaml   --doc "<DOC_ID>"   --checklist documents/compliance_checklist.json   --log-tokens
```

5) **Visualize**:

```bash
# Ingest
#latest run and id 
LATEST_RUN=$(ls -dt data/runs/run_* | head -n 1)
DOC_ID=$(basename "$(ls "$LATEST_RUN"/parsed/*_sections.json | head -n 1)" _sections.json)

python cli.py viz-ingest --run "$LATEST_RUN" --doc "$DOC_ID"
# Index
python cli.py viz-index --config config.yaml --doc "$DOC_ID"
# Assess
python cli.py viz-assess --run "$LATEST_RUN" --doc "$DOC_ID"
```

6) **Pipeline runner (bash)**:

```bash

# 1) make executable once
chmod +x run_pipeline_openai.sh

# 2) set your API key (only if using OpenAI for embeddings/judging)
export OPENAI_API_KEY="sk-...your-key..."

# 3) run
./run_pipeline_openai.sh
```

> Outputs live under `data/runs/run_*/`, especially `assess/<doc_id>/report.md`.

---

## Streamlit Web App

A minimal front-end that uploads a PDF to `documents/`, then runs the same CLI steps and shows:

- Progress bars & timings
- Links to HTML dashboards
- Assessment **summary** in-app (download full `report.md`)

Run:

```bash
export OPENAI_API_KEY=sk-...     # if using OpenAI
streamlit run web/app.py
```

---

## Ingestion Details

- **Docling**: better structure recovery (headings, tables, figures). Optional **OCR** for low-text pages.
- **PyMuPDF**: fast fallback.
- **Sectioning**: robust heuristics + Docling metadata. Fixes earlier bugs (e.g., repeated page ranges, heading regex).
- **Chunking**: sentence-aware, **character-based target size** with overlap; keeps chunks <~300 tokens (typical), so **no trimming** needed later.

**Known parsing pitfalls addressed:**

- Some PDFs report corrupted layer config → handled with warning.
- OCR model downloads & accelerator selection logged.
- Page mapping errors fixed by using cumulative char offsets.

---

## Retrieval & Indexing

- **BM25** index built from chunks (elastic signals).
- **FAISS** index built from chunk embeddings.
- **Hybrid search** merges BM25 & vector scores; **strict top-N** is selected by final hybrid rank (no per-section diversification).
- Configurable backends:
  - **OpenAI** (accurate, cloud).
  - **SBERT** (local, CPU-friendly).

**Index Artifacts (per doc):**

```
data/index/<doc_id>/
  ├─ bm25.jsonl / metadata
  ├─ faiss.index
  ├─ vectors.npy (optional)
  └─ mapping.jsonl   # chunk_id → section_title, page range, text_len, ...
```

---

## Assessment (Single-Stage, Per-Chunk)

For each **atomic checklist question**:

1. Run hybrid search → take **top N** chunks (config `assess.topn` or `--topn`).
2. For **each chunk**, build a micro-prompt: **(requirement + only that chunk)**.
3. Send **one** LLM call that must return strict JSON:
   ```json
   {"verdict":"Covered|Partially Covered|Not Covered","confidence":1-5,"rationale":"..."}
   ```
4. Aggregate across chunks with priority: **Covered > Partial > Not Covered** (mean confidence among final-level chunks).

**Token discipline:**

- We estimate tokens for system+user content; warnings when:
  - **> soft (1000)** → yellow
  - **> hard (1500)** → red
- Since chunks are small (<~300 tokens) and prompts simple, calls remain under cap.

**Outputs per question:** stored in `results.jsonl`, including:

- top-N chunk IDs, hybrid scores, pages/section titles
- per-chunk verdicts & rationales
- final aggregated verdict

**Summaries:**

- `summary.json`: counts + percentages per verdict class
- `report.md`: readable summary with diagnostics

---

## 📊 Visualization

- **Ingest viz**: sections table, chunk histograms, figures/tables per section, OCR density, quality flags.
- **Index viz**: IDF plots (if Plotly available), embedding cosine stats, nearest-neighbor examples.
- **Assess viz**: stacked bars of verdicts, question-level drill-down, token logs, evidence snippets.

Open via the CLI viz commands or the Streamlit link buttons.

---

## Troubleshooting

- **`faiss-cpu not installed`** → `pip install faiss-cpu`
- **`PyYAML is not installed`** but `pip show` says it is:
  - Verify **active venv**: `which python; which pip`
  - Reinstall: `pip install --upgrade PyYAML`
- **Docling page mapping / headings odd**:
  - Ensure you’re on the latest `docling-core` and our `io.py` heuristics are in place.
- **`RecursionError` in Streamlit** (from earlier): don’t call a helper from itself; use `os.environ.copy()` in env helpers.
- **Streamlit progress error (`invalid type: function`)**: don’t read private attributes like `prog._value`; maintain your own float and call `prog.progress(value)`.
- **OpenSSL/LibreSSL warning on macOS**: harmless; consider upgrading Python or `urllib3` if needed.
- **Token overrun**: reduce `assess.topn`, chunk size, or prompt verbosity.

---

## 🔐 Security Notes

- Never commit API keys; use environment variables.
- The Streamlit app does **not** store the key; it uses your shell environment.
- `.env` (if you use one) should be `.gitignored`

---

## Performance Tips

- Prefer **SBERT** locally if network is limiting and you’re okay with slightly reduced recall vs OpenAI embeddings.
- Increase `index.batch_size` when embedding large docs (watch RAM).
- For OCR-heavy docs, enable `ingest.parser.enable_ocr: true` but expect slower ingest.

---

## 🧪 Testing

- Use **dry-run** mode for quick plumbing tests (no LLM required):
  ```bash
  python cli.py assess --config config.yaml --doc "<DOC_ID>" --checklist documents/compliance_checklist.json --dry-run
  ```
- Unit-test ingestion by asserting number of sections/chunks and page maps.
- Compare hybrid retrieval hits vs BM25-only for sanity.

## 📄 License

AGPL 3.0
