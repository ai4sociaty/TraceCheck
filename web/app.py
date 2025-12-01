# web/app.py
# -*- coding: utf-8 -*-
"""
Minimal Streamlit front-end that calls your existing CLI (cli.py) for each step:

1) ingest
2) index (OpenAI or SBERT)
3) viz-ingest
4) viz-index
5) assess (with --log-tokens)
6) viz-assess

Adds:
- Danger zone to purge PDFs
- Per-step timing with progress bars (tqdm-like)
- Clickable buttons for HTML visualizations
- Only assessment summary in UI (full report downloadable)
"""

from __future__ import annotations
import os, io, glob, json, subprocess, time
from pathlib import Path
import streamlit as st

# ---------- tiny utils ----------

def load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        st.error(f"Failed to read YAML: {e}")
        return {}

def latest_run_dir(base: str = "data/runs") -> str | None:
    runs = sorted(glob.glob(os.path.join(base, "run_*")))
    return runs[-1] if runs else None

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def run_cli(cmd: list[str], title: str, show_full: bool = False, env: dict | None = None, prog=None) -> tuple[int, float]:
    """
    Run a CLI command, streaming output into an expander.
    Returns (exit_code, elapsed_seconds).
    If 'prog' is a streamlit progress object, it updates during streaming.
    """
    start = time.time()
    buf = []
    local_progress = 0.0  # keep our own progress value

    with st.expander(title, expanded=False):
        st.code(" ".join(cmd))
        out = st.empty()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )
            n_lines = 0
            for line in proc.stdout:
                n_lines += 1
                if show_full:
                    buf.append(line)
                else:
                    buf.append(line)
                    if len(buf) > 120:
                        buf = buf[-120:]
                out.code("".join(buf))

                # tqdm-like feel: bump progress every 10 lines (cap at 0.98)
                if prog and (n_lines % 10 == 0):
                    local_progress = min(0.98, local_progress + 0.02)
                    prog.progress(local_progress)

            rc = proc.wait()
        except Exception as e:
            st.error(str(e))
            rc = 1

        elapsed = time.time() - start
        # finalize progress bar
        if prog:
            prog.progress(1.0 if rc == 0 else local_progress)

        if rc == 0:
            st.success(f"✅ done in {elapsed:.1f}s")
        else:
            st.error(f"❌ exit code {rc} (after {elapsed:.1f}s)")
        return rc, elapsed


def link_button(label: str, path: str):
    if os.path.exists(path):
        uri = Path(path).resolve().as_uri()
        st.link_button(label, uri)
    else:
        st.caption(f"{label}: _not found_")

# ---------- UI ----------

st.set_page_config(page_title="TraceCheck Compliance – CLI Runner", layout="wide")
st.title("🧩 TraceCheck Compliance — CLI Orchestrator")

# Left: upload & run, Right: settings
left, right = st.columns([2, 1])

with right:
    st.subheader("⚙️ Settings")

    cfg_path = st.text_input("config.yaml", value="config.yaml")
    cfg = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
    paths = cfg.get("paths", {})
    docs_dir = paths.get("documents", "documents")
    runs_dir = paths.get("runs_dir", "data/runs")
    index_dir = paths.get("index_dir", "data/index")

    st.caption(f"documents: `{docs_dir}`  |  runs: `{runs_dir}`  |  index: `{index_dir}`")

    # Embedding backend & models
    backend = st.selectbox("Embedding backend", ["openai", "sbert"], index=0)
    embed_model = st.text_input(
        "Embedding model (OpenAI: text-embedding-3-small, SBERT: sentence-transformers/all-MiniLM-L6-v2)",
        value="text-embedding-3-small" if backend == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model = st.text_input("LLM model (assess)", value="gpt-4o-mini")

    # Retrieval / viz args
    topn = st.number_input("Assess: top-N chunks per question", min_value=1, max_value=50, value=10, step=1)
    #overlap_tol = st.number_input("viz-ingest: overlap tolerance", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f")
    #low_char_th = st.number_input("viz-ingest: low-char threshold", min_value=0, max_value=2000, value=100, step=50)
    #tiny_chunk_th = st.number_input("viz-ingest: tiny-chunk threshold", min_value=0, max_value=5000, value=400, step=50)

    # Checklist path
    checklist_path = st.text_input("Checklist JSON", value="documents/compliance_checklist.json")

    # Logging
    show_full_logs = st.checkbox("Show full logs (verbose)", value=False)
    st.caption("If using OpenAI, export OPENAI_API_KEY in your shell before launching Streamlit.")

    st.markdown("---")
    st.subheader("🧨 Danger Zone — Purge PDFs")
    default_purge_dir = docs_dir  # change if your PDFs live elsewhere
    purge_dir = st.text_input("Folder to purge (*.pdf will be deleted)", value=default_purge_dir)
    confirm = st.checkbox("I understand this will permanently delete all *.pdf files in the folder above.")
    if st.button("Delete all PDFs in folder"):
        if not confirm:
            st.warning("Please confirm before deleting.")
        else:
            deleted = 0
            if os.path.isdir(purge_dir):
                for p in glob.glob(os.path.join(purge_dir, "*.pdf")):
                    try:
                        os.remove(p)
                        deleted += 1
                    except Exception as e:
                        st.error(f"Failed to delete {p}: {e}")
                st.success(f"Deleted {deleted} PDF(s) from {purge_dir}")
            else:
                st.error(f"Folder not found: {purge_dir}")

with left:
    st.subheader("📄 Upload PDF")
    ensure_dir(docs_dir)
    up = st.file_uploader("Drop a PDF", type=["pdf"])
    doc_id = None
    if up:
        pdf_path = os.path.join(docs_dir, up.name)
        with open(pdf_path, "wb") as f:
            f.write(up.getbuffer())
        st.success(f"Saved to `{pdf_path}`")
        doc_id = Path(up.name).stem
        st.caption(f"doc_id = `{doc_id}`")

    st.markdown("---")
    st.subheader("🚀 Run Pipeline (CLI)")
    run_now = st.button("Run (ingest → index → viz → assess → viz)")

    if run_now and doc_id:
        # Prepare environment (pass-through current env)
        env = os.environ.copy()

        # Overall tqdm-like progress
        overall = st.progress(0.0, text="Starting…")
        step_weight = 1.0 / 6.0  # 6 stages (incl. post viz-assess)
        timers = {}
        total_start = time.time()

        # 1) INGEST
        overall.progress(0.00, text="Step 1/6 — ingest")
        p1 = st.progress(0.0)
        rc, t = run_cli(
            ["python", "cli.py", "ingest", "--config", cfg_path],
            title="Step 1/6 — ingest",
            show_full=show_full_logs,
            env=env,
            prog=p1
        )
        timers["ingest"] = t
        if rc != 0: st.stop()
        overall.progress(step_weight, text="Step 1/6 — ingest ✓")

        run_dir = latest_run_dir(runs_dir)
        if not run_dir:
            st.error("No run dir found after ingest.")
            st.stop()

        # 2) INDEX (explicit args; force backend)
        overall.progress(step_weight * 1, text="Step 2/6 — index")
        p2 = st.progress(0.0)
        index_cmd = [
            "python", "cli.py", "index",
            "--config", cfg_path,
            "--run", run_dir,
            "--doc", doc_id,
            "--backend", backend,
            "--model", embed_model
        ]
        if backend == "openai":
            index_cmd += ["--use-openai"]
        else:
            index_cmd += ["--no-openai"]

        rc, t = run_cli(index_cmd, title="Step 2/6 — index (BM25 + Embeddings)", show_full=show_full_logs, env=env, prog=p2)
        timers["index"] = t
        if rc != 0: st.stop()
        overall.progress(step_weight * 2, text="Step 2/6 — index ✓")

        # 3) VIZ-INGEST (explicit thresholds)
        overall.progress(step_weight * 2, text="Step 3/6 — viz-ingest")
        p3 = st.progress(0.0)
        rc, t = run_cli([
            "python", "cli.py", "viz-ingest",
            "--run", run_dir,
            "--doc", doc_id,
            #"--overlap-tolerance", str(overlap_tol),
            #"--low-char-threshold", str(low_char_th),
            #"--tiny-chunk-threshold", str(tiny_chunk_th)
        ], title="Step 3/6 — viz-ingest", show_full=show_full_logs, env=env, prog=p3)
        timers["viz_ingest"] = t
        if rc != 0:
            st.info("viz-ingest failed or skipped; continuing.")
        overall.progress(step_weight * 3, text="Step 3/6 — viz-ingest ✓")

        # 4) VIZ-INDEX
        overall.progress(step_weight * 3, text="Step 4/6 — viz-index")
        p4 = st.progress(0.0)
        rc, t = run_cli([
            "python", "cli.py", "viz-index",
            "--config", cfg_path,
            "--doc", doc_id
        ], title="Step 4/6 — viz-index", show_full=show_full_logs, env=env, prog=p4)
        timers["viz_index"] = t
        if rc != 0:
            st.info("viz-index failed or skipped; continuing.")
        overall.progress(step_weight * 4, text="Step 4/6 — viz-index ✓")

        # 5) ASSESS (explicit; include --log-tokens)
        overall.progress(step_weight * 4, text="Step 5/6 — assess")
        p5 = st.progress(0.0)
        assess_cmd = [
            "python", "cli.py", "assess",
            "--config", cfg_path,
            "--doc", doc_id,
            "--checklist", checklist_path,
            "--topn", str(topn),
            "--log-tokens"
        ]
        if backend == "openai":
            assess_cmd += ["--backend", "openai", "--model", llm_model]
        else:
            assess_cmd += ["--backend", "sbert", "--model", llm_model]

        rc, t = run_cli(assess_cmd, title="Step 5/6 — assess (LLM) with --log-tokens", show_full=show_full_logs, env=env, prog=p5)
        timers["assess"] = t
        if rc != 0: st.stop()
        overall.progress(step_weight * 5, text="Step 5/6 — assess ✓")

        # 6) VIZ-ASSESS (post-assess)
        overall.progress(step_weight * 5, text="Step 6/6 — viz-assess")
        p6 = st.progress(0.0)
        rc, t = run_cli([
            "python", "cli.py", "viz-assess",
            "--run", run_dir,
            "--doc", doc_id
        ], title="Step 6/6 — viz-assess", show_full=show_full_logs, env=env, prog=p6)
        timers["viz_assess"] = t
        # continue even if it fails
        overall.progress(1.0, text="All steps complete ✓")

        total_elapsed = time.time() - total_start
        st.success(f"✅ Pipeline complete in {total_elapsed:.1f}s")

        # -------- Results panel --------
        assess_dir = os.path.join(run_dir, "assess", doc_id)
        report_md = os.path.join(assess_dir, "report.md")
        summary_json = os.path.join(assess_dir, "summary.json")
        ingest_dash = os.path.join(run_dir, "viz", "ingest_dashboard.html")
        index_dash = os.path.join(index_dir, doc_id, "viz", "index_dashboard.html")
        assess_dash = os.path.join(assess_dir, "viz", "assessment_dashboard.html")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### 📊 Dashboards")
            link_button("Open Ingest dashboard", ingest_dash)
            link_button("Open Index dashboard", index_dash)
            link_button("Open Assessment dashboard", assess_dash)

        with colB:
            st.markdown("### 📘 Assessment Summary")
            if os.path.exists(summary_json):
                sj = json.loads(Path(summary_json).read_text(encoding="utf-8"))
                counts = sj.get("counts", {})
                perc = sj.get("percents", {})
                total = sj.get("total_items", 0)
                st.write(f"**Total items:** {total}")
                for k in ["Covered", "Partially Covered", "Not Covered"]:
                    if k in counts:
                        st.write(f"- **{k}**: {counts[k]} ({perc.get(k, 0)}%)")
                # Download full report
                if os.path.exists(report_md):
                    text = Path(report_md).read_text(encoding="utf-8")
                    st.download_button("⬇️ Download full report.md", text, file_name="report.md")
                else:
                    st.caption("report.md not found.")
            else:
                st.info(f"summary.json not found at: {summary_json}")

    elif run_now and not doc_id:
        st.warning("Please upload a PDF first.")
