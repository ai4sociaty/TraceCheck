#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
CONFIG_FILE="config.yaml"
CHECKLIST="documents/compliance_checklist.json"
BACKEND="openai"
EMBED_MODEL="text-embedding-3-small"
LLM_MODEL="gpt-4o-mini"

# -----------------------------
# Presentation
# -----------------------------
C_BOLD="\033[1m"; C_RED="\033[31m"; C_GRN="\033[32m"; C_YEL="\033[33m"; C_RST="\033[0m"
info() { echo -e "${C_BOLD}[$(date +%H:%M:%S)]${C_RST} $*"; }
ok()   { echo -e "${C_GRN}[OK]${C_RST} $*"; }
warn() { echo -e "${C_YEL}[WARN]${C_RST} $*"; }
err()  { echo -e "${C_RED}[ERR]${C_RST} $*" >&2; }

# -----------------------------
# Load API key
# -----------------------------
if [[ -f ".env" ]]; then
  source .env
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  err "❌ OPENAI_API_KEY not found. Please set it in your environment or .env file."
  exit 1
else
  ok "Using OpenAI API key (env detected)"
fi

# -----------------------------
# Step 1 — Ingest PDFs
# -----------------------------
info "Step 1/6: Ingest PDFs"
python cli.py ingest --config "$CONFIG_FILE"

LATEST_RUN="$(ls -dt data/runs/run_* | head -n 1)"
[[ -n "$LATEST_RUN" ]] || { err "No run directories found after ingest."; exit 1; }
ok "Latest run: $LATEST_RUN"

DOC_ID="$(basename "$(ls "$LATEST_RUN"/parsed/*_sections.json | head -n 1)" _sections.json)"
[[ -n "$DOC_ID" ]] || { err "No parsed documents found."; exit 1; }
ok "Detected document ID: $DOC_ID"

# -----------------------------
# Step 2 — Build Index
# -----------------------------
info "Step 2/6: Building FAISS + BM25 indices (OpenAI embeddings)"
python cli.py index \
  --config "$CONFIG_FILE" \
  --run "$LATEST_RUN" \
  --doc "$DOC_ID" \
  --backend openai \
  --model "$EMBED_MODEL" \
  --use-openai
ok "Indexing complete."

# -----------------------------
# Step 3 — Visualize Ingest
# -----------------------------
info "Step 3/6: Visualize ingestion"
python cli.py viz-ingest \
  --run "$LATEST_RUN" \
  --doc "$DOC_ID"
ok "Ingest visualization complete."

# -----------------------------
# Step 4 — Visualize Index
# -----------------------------
info "Step 4/6: Visualize index"
python cli.py viz-index \
  --config "$CONFIG_FILE" \
  --doc "$DOC_ID"
ok "Index visualization complete."

# -----------------------------
# Step 5 — Assess Compliance
# -----------------------------
info "Step 5/6: Running OpenAI compliance assessment"
python cli.py assess \
  --config "$CONFIG_FILE" \
  --doc "$DOC_ID" \
  --checklist "$CHECKLIST" \
  --model "$LLM_MODEL" \
  --backend openai\
  --log-tokens 
ok "Assessment complete."

# -----------------------------
# Step 6 — Visualize Assessment
# -----------------------------
info "Step 6/6: Visualize assessment results"
python cli.py viz-assess \
  --run "$LATEST_RUN" \
  --doc "$DOC_ID"
ok "Assessment visualization complete."

# -----------------------------
# Final report
# -----------------------------
info "Generating final report"
python cli.py report \
  --run "$LATEST_RUN" \
  --config "$CONFIG_FILE"
ok "✅ Pipeline complete for $DOC_ID"
echo "📂 Results: $LATEST_RUN"
