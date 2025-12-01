# app/utils.py
import os, glob
import shutil
import argparse
from datetime import datetime


import os
import sys
import time
import json
import shutil
import argparse
from datetime import datetime


def get_latest_run_dir(base_dir: str) :
    """Return the most recent run_YYYYMMDD_HHMM folder."""
    runs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    return runs[-1] if runs else None




def load_yaml(path: str) -> dict:
    import yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("run_%Y%m%d_%H%M")
    run_dir = os.path.join(base_dir, ts)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "inputs"))
    ensure_dir(os.path.join(run_dir, "logs"))
    return run_dir


def snapshot_config(cfg_path: str, run_dir: str) -> str:
    """Copy the current config.yaml into run/inputs/config.snapshot.yaml"""
    dest_dir = os.path.join(run_dir, "inputs")
    ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, "config.snapshot.yaml")
    shutil.copy(cfg_path, dest_path)
    return dest_path



def resolve_embedding_cfg(cfg: dict, args) -> tuple[str, str, int, bool, str]:
    """
    Decide backend/model and core params from config + CLI + env.
    Returns: (backend, model, batch_size, normalize, metric)
    Precedence:
      1) --use-openai / --no-openai / --backend / --model
      2) cfg['index'] values
      3) env (OPENAI_API_KEY) when backend='auto'
    """
    idx = cfg.setdefault("index", {})
    backend = (args.backend or idx.get("embedding_backend", "auto")).lower()

    # CLI convenience
    if args.use_openai:
        backend = "openai"
    if args.no_openai and backend == "openai":
        backend = "sbert"
    if args.backend:
        backend = args.backend.lower()

    defaults = idx.get("defaults", {}) if isinstance(idx.get("defaults", {}), dict) else {}
    default_sbert = defaults.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2")
    default_openai = defaults.get("openai_model", "text-embedding-3-small")

    model = args.model  # CLI override wins
    metric = (idx.get("faiss_metric", "ip") or "ip").lower()
    batch_size = int(idx.get("batch_size", 64))
    normalize = bool(idx.get("normalize", True if metric == "ip" else False))

    if not model:
        if backend == "auto":
            if os.getenv("OPENAI_API_KEY") and not args.no_openai:
                backend = "openai"
                model = default_openai
            else:
                backend = "sbert"
                model = default_sbert
        elif backend == "openai":
            model = default_openai
        else:
            backend = "sbert"
            model = default_sbert

    return backend, model, batch_size, normalize, metric
