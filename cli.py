# cli.py
# -*- coding: utf-8 -*-
"""
Command-line interface for the Matrix Compliance pipeline.

"""

import os
from re import sub
import sys
import time
import json
import shutil
import argparse
from datetime import datetime

# ------------------ Local imports ------------------
from app.io import Ingestor
from app.ingest_viz import IngestVisualizer

# (Optional future imports)
# from app.retrieve import Indexer
# from app.pipeline import CompliancePipeline
# from app.report import ReportGenerator
from app import utils
import os, glob
import  glob
from app.retrieve import IndexBuilder
from app.utils import load_yaml, make_run_dir, snapshot_config, ensure_dir, resolve_embedding_cfg  # keep your existing ones as-is

# ------------------ Helpers ------------------

# ------------------ CLI Main ------------------

# cli.py
import argparse, json, sys, os, glob
from app import utils
from app.io import Ingestor
from app.ingest_viz import IngestVisualizer
from app.utils import load_yaml, make_run_dir, snapshot_config, resolve_embedding_cfg

def main():
    parser = argparse.ArgumentParser(
        prog="matrix-compliance",
        description="Matrix Requirements – Automated Document Compliance Verification"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------- INGEST -------------------------
    p_ingest = sub.add_parser("ingest", help="Parse and chunk PDFs into structured JSON")
    p_ingest.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # -------------------------- INDEX -------------------------
    p_index = sub.add_parser("index", help="Build BM25 + embedding indices")
    p_index.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_index.add_argument("--run", help="Run directory (defaults to latest run)")
    p_index.add_argument("--doc", help="Specific document ID (basename without _sections.json)")
    p_index.add_argument("--backend", choices=["auto", "sbert", "openai"], help="Override embedding backend")
    p_index.add_argument("--model", help="Override embedding model")
    p_index.add_argument("--use-openai", action="store_true", help="Force OpenAI backend")
    p_index.add_argument("--no-openai", action="store_true", help="Disable OpenAI (auto -> sbert)")

    # ------------------------- ASSESS -------------------------
    p_assess = sub.add_parser("assess", help="Run compliance checklist analysis")
    p_assess.add_argument("--config", default="config.yaml", help="Path to config file")
    p_assess.add_argument("--doc", required=True, help="Document ID (same as used during ingest)")
    p_assess.add_argument("--checklist", default="documents/compliance_checklist.json", help="Path to checklist JSON")
    p_assess.add_argument("--backend", choices=["auto", "sbert", "openai"], default=None,
                          help="Embedding backend override (optional)")
    #p_assess.add_argument("--model", default=None, help="Embedding model override (optional)")
    p_assess.add_argument("--model", default=None, help="Embedding or LLM model override")

    p_assess.add_argument("--topn", type=int, default=10, help="Top-N chunks per question")
    p_assess.add_argument("--dry-run", action="store_true", help="Skip LLM calls and use heuristic judgments")
    # assess subparser
    p_assess.add_argument("--log-tokens", action="store_true",
                      help="Print per-chunk token breakdown for each LLM call")


    # ------------------------ REPORT --------------------------
    p_report = sub.add_parser("report", help="Aggregate results and generate final report")
    p_report.add_argument("--run", required=False, help="Specific run directory (defaults to latest)")
    p_report.add_argument("--config", default="config.yaml", help="Path to config.yaml")  # ← add


    # ---------------------- VIZ INGEST ------------------------
    p_viz_ing = sub.add_parser("viz-ingest", help="Visualize and QA the ingest stage")
    p_viz_ing.add_argument("--run", required=True, help="Run directory (e.g., data/runs/run_YYYYMMDD_HHMM)")
    p_viz_ing.add_argument("--doc", required=True, help="Document ID (without extension)")
    p_viz_ing.add_argument("--overlap-tolerance", type=float, default=0.05)
    p_viz_ing.add_argument("--low-char-threshold", type=int, default=100)
    p_viz_ing.add_argument("--tiny-chunk-threshold", type=int, default=400)
    p_viz_ing.add_argument("--no-tables", action="store_true")
    p_viz_ing.add_argument("--no-captions", action="store_true")
    p_viz_ing.add_argument("--keep-captions-in-keywords", action="store_true",
                           help="Do not strip figure/table captions when counting keywords.")
    p_viz_ing.add_argument("--no-figtab-panel", action="store_true", help="Skip Figures/Tables panel.")
    p_viz_ing.add_argument("--no-ocr-panel", action="store_true", help="Skip OCR density panel.")

    # ---------------------- VIZ INDEX -------------------------
    p_viz_idx = sub.add_parser("viz-index", help="Visualize and QA the indexing stage")
    p_viz_idx.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p_viz_idx.add_argument("--doc", required=True, help="Document ID (basename)")

    # ---------------------- VIZ ASSESS -------------------------
    p_viz_assess = sub.add_parser("viz-assess", help="Visualize and QA the assessment stage")
    p_viz_assess.add_argument("--run", required=False, help="Run directory (defaults to latest)")
    p_viz_assess.add_argument("--doc", required=True, help="Document ID used in assess")


    # >>> parse AFTER all subparsers are defined
    args = parser.parse_args()

    # ======================= DISPATCH =========================
    if args.cmd == "ingest":
        cfg = load_yaml(args.config)
        run_dir = make_run_dir(cfg["paths"]["runs_dir"])
        snapshot_config(args.config, run_dir)
        print(f"[INFO] Starting ingestion → {run_dir}")
        ing = Ingestor(cfg, run_dir)
        outputs = ing.run_all()
        print(json.dumps(outputs, indent=2))
        print(f"[INFO] Ingest complete. Results saved to {run_dir}")
        return

    elif args.cmd == "index":
        from app.retrieve import IndexBuilder
        cfg = load_yaml(args.config)
        backend, model, batch, normalize, metric = resolve_embedding_cfg(cfg, args)
        cfg.setdefault("index", {}).update({
            "embedding_backend": backend,
            "embedding_model": model,
            "batch_size": batch,
            "normalize": normalize,
            "faiss_metric": metric,
        })
        run_dir = args.run or utils.get_latest_run_dir(cfg["paths"]["runs_dir"])
        if not run_dir:
            print("No run directory found. Run `ingest` first.")
            sys.exit(1)

        if args.doc:
            docs = [args.doc]
        else:
            pattern = os.path.join(run_dir, "parsed", "*_sections.json")
            docs = [os.path.basename(p)[:-len("_sections.json")] for p in glob.glob(pattern)]
            if not docs:
                print(f"No parsed docs found in {run_dir}/parsed. Run `ingest` first.")
                sys.exit(1)

        index_root = cfg["paths"]["index_dir"]
        os.makedirs(index_root, exist_ok=True)
        results = {}
        for doc_id in docs:
            builder = IndexBuilder(run_dir=run_dir, doc_id=doc_id, index_root=index_root, cfg=cfg)
            paths = builder.build_all()
            results[doc_id] = paths
        print(json.dumps(results, indent=2))
        print(f"[INFO] Indices saved under {index_root}/<doc_id>/")
        print(f"[INFO] Embeddings: backend={backend}, model={model}, normalize={normalize}, metric={metric}")
        return

    elif args.cmd == "assess":
        from app.assess import ComplianceAssessor
        cfg = load_yaml(args.config)
        run_dir = utils.get_latest_run_dir(cfg["paths"]["runs_dir"])
        if not run_dir:
            print("❌ No run directory found. Run `ingest` first.")
            sys.exit(1)

        if args.backend:
            cfg.setdefault("index", {})["embedding_backend"] = args.backend
        if args.model:
            cfg.setdefault("index", {})["embedding_model"] = args.model


        assessor = ComplianceAssessor(
        cfg=cfg,
        run_dir=run_dir,
        doc_id=args.doc,
        checklist_path=args.checklist,
        topn_strict=args.topn,
        dry_run=args.dry_run,
        llm_model=args.model or "gpt-4o-mini",  # 👈 inject the model argument here
        )
        assessor.log_tokens = getattr(args, "log_tokens", False)

        out = assessor.run_all()
        print(json.dumps(out, indent=2))
        print(f"[INFO] Assessment results saved to {out['out_dir']}")
        return
    elif args.cmd == "viz-assess":
        from app.assess_viz import AssessVisualizer
        run_dir = args.run or utils.get_latest_run_dir("data/runs")
        if not run_dir:
            print("No run directory found.")
            sys.exit(1)
        v = AssessVisualizer(run_dir=run_dir, doc_id=args.doc)
        out = v.run()
        print(json.dumps(out, indent=2))
        print(f"[INFO] Assessment dashboard written to {out['dashboard_html']}")
        return


    elif args.cmd == "report":
        from app.report import ReportGenerator
        cfg = load_yaml(args.config)                    # ← add this
        run_dir = args.run or utils.get_latest_run_dir("data/runs")
        if not run_dir:
            print("No run directory found.")
            sys.exit(1)
        # discover docs in this run
        pattern = os.path.join(run_dir, "parsed", "*_sections.json")
        doc_ids = [os.path.basename(p)[:-len("_sections.json")] for p in glob.glob(pattern)]
        rg = ReportGenerator(cfg=cfg, run_dir=run_dir, doc_ids=doc_ids)  # ← pass cfg here
        out = rg.run()
        print(json.dumps(out, indent=2))
        print(f"[INFO] Reports written under {run_dir}/reports/")
        return


    elif args.cmd == "viz-ingest":
        v = IngestVisualizer(
            run_dir=args.run,
            doc_id=args.doc,
            overlap_tolerance=args.overlap_tolerance,
            low_char_threshold=args.low_char_threshold,
            tiny_chunk_threshold=args.tiny_chunk_threshold,
            show_tables=not args.no_tables,
            show_captions=not args.no_captions,
            exclude_captions_in_keywords=not args.keep_captions_in_keywords,
        )
        if hasattr(v, "_panel_figtab"):
            v._skip_figtab = args.no_figtab_panel
        if hasattr(v, "_panel_ocr_density"):
            v._skip_ocr = args.no_ocr_panel
        out = v.run()
        print(json.dumps(out, indent=2))
        print(f"[INFO] Dashboard and summary written to {args.run}/viz/")
        return

    elif args.cmd == "viz-index":
        from app.index_viz import IndexVisualizer
        cfg = load_yaml(args.config)
        index_root = cfg["paths"]["index_dir"]
        v = IndexVisualizer(index_root=index_root, doc_id=args.doc)
        out = v.run()
        print(json.dumps(out, indent=2))
        print(f"[INFO] Index dashboard written to {out['dashboard_html']}")
        return

    else:
        parser.print_help()
        return

if __name__ == "__main__":
    main()


'''
source .venv/bin/activate

python cli.py ingest
LATEST_RUN=$(ls -dt data/runs/run_* | head -n 1)
echo "Using OpenAI API key: $OPENAI_API_KEY"
python cli.py index --config config.yaml --backend openai --model text-embedding-3-small  --run "$LATEST_RUN" 

python cli.py viz-ingest \
  --run "$LATEST_RUN" \
  --doc "5_Memorygel Silicone Gel -Filled Breast Implants_SSED"

DOC_ID=$(basename "$(ls "$LATEST_RUN"/parsed/*_sections.json | head -n 1)" _sections.json)
python cli.py viz-ingest --run "$LATEST_RUN" --doc "$DOC_ID"
echo "$DOC_ID"

python cli.py viz-index --doc "$DOC_ID"



python cli.py assess \
  --config config.yaml \
  --doc "$DOC_ID" \
  --checklist documents/compliance_checklist.json \
  --model gpt-4o-mini \
  --log-tokens


python cli.py viz-assess \
  --run "$LATEST_RUN" \
  --doc "$DOC_ID"

python cli.py report --run "$LATEST_RUN" --config config.yaml
 '''