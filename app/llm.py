# app/llm.py
# -*- coding: utf-8 -*-
"""
LLM helpers for compliance assessment.

Exposes:
- SingleStageJudge(model_name): one-shot per-chunk judgment → JSON
- count_tokens(text, model_or_encoding): token counter (tiktoken if available, else heuristic)

Optional:
- EvidenceExtractor / VerdictAssessor stubs kept for compatibility (not used in single-stage flow).
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import logging
from typing import Optional, Dict, Any

# ---------------- Token counting ----------------

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))

def count_tokens(text: str, model_or_encoding: str = "gpt-4o-mini") -> int:
    """
    Use tiktoken if installed; otherwise 4 chars ≈ 1 token heuristic.
    """
    try:
        import tiktoken  # type: ignore
        enc = None
        try:
            enc = tiktoken.encoding_for_model(model_or_encoding)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return _approx_tokens(text or "")

# ---------------- OpenAI client (optional) ----------------

class _OpenAIClient:
    def __init__(self, model: str, timeout: float = 30.0, max_retries: int = 2, logger: Optional[logging.Logger]=None):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = logger or logging.getLogger(__name__)
        self._client = None
        self._ready = False

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # don't raise here; will raise when chat_json is called
            return
        from openai import OpenAI
        self._client = OpenAI(timeout=timeout)
        self._ready = True

    def chat_json(self, system: str, user: str) -> str:

        """
        Call chat/completions and return the text content. We nudge for JSON only.
        """
        if not self._ready:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAI-backed judging.")
        

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                out = resp.choices[0].message.content or ""
                return out.strip()
            except Exception as e:
                last_err = e
                self.log.warning("OpenAI error (attempt %d/%d): %s", attempt+1, self.max_retries+1, e)
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

# ---------------- JSON guards ----------------

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Try strict json.loads; else find first {...} block and parse; else return {}.
    """
    if not text:
        return {}
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) first JSON-looking block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def _coerce_verdict(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    if s in {"covered", "full", "fully covered", "fully"}:
        return "Covered"
    if s in {"partially covered", "partial", "partially"}:
        return "Partially Covered"
    if s in {"not covered", "no", "none"}:
        return "Not Covered"
    # Fallback
    return "Not Covered"

def _clip_rationale(r: Optional[str], max_chars: int = 600) -> str:
    r = (r or "").strip()
    if len(r) > max_chars:
        return r[:max_chars] + " …"
    return r

# ---------------- Single-stage judge ----------------

_SINGLE_STAGE_SYSTEM = (
    "You are a meticulous RA/QA reviewer. "
    "You must return STRICT JSON only. "
    "Decide coverage of the requirement based ONLY on the provided excerpt. "
    "Do not use external knowledge."
)

_SINGLE_STAGE_USER_TMPL = """Return a JSON object with exactly these keys:
- "verdict": one of ["Covered","Partially Covered","Not Covered"]
- "confidence": integer 1-5 (5=high)
- "rationale": <= 80 tokens, concise justification citing phrases from the excerpt

Requirement:
{requirement}

Excerpt:
{excerpt}
"""

class SingleStageJudge:
    """
    One-call classifier: (requirement + one excerpt) -> JSON verdict.
    Uses OpenAI if OPENAI_API_KEY is set; otherwise raise (caller should run dry-run).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None):
        self.model_name = model_name or "gpt-4o-mini"
        self.log = logger or logging.getLogger(__name__)
        self.client = _OpenAIClient(self.model_name, logger=self.log)

    def run(self, requirement: str, context: str) -> Dict[str, Any]:
        """
        'context' should already contain the excerpt block (built by assessor).
        We pass requirement + excerpt explicitly to the LLM with a JSON-only instruction.
        """
        # Extract the excerpt body from the assessor-provided context
        # The assessor formats:
        #   ... "Requirement:\n{requirement}\n\nExcerpt:\n[Section: ...]\n{raw_text}\n"
        # Here we simply reuse `context` as "excerpt" to avoid mismatches.
        # For clarity, we pass 'requirement' separately and 'context' as excerpt.
        user_msg = _SINGLE_STAGE_USER_TMPL.format(requirement=requirement, excerpt=context)

        raw = self.client.chat_json(system=_SINGLE_STAGE_SYSTEM, user=user_msg)
        data = _extract_json_block(raw)

        verdict = _coerce_verdict(data.get("verdict"))
        conf = data.get("confidence")
        try:
            conf = int(conf)
        except Exception:
            conf = 3
        conf = min(5, max(1, conf))

        rationale = _clip_rationale(data.get("rationale"))

        return {
            "verdict": verdict,
            "confidence": conf,
            "rationale": rationale,
        }

# ---------------- Compatibility stubs (optional) ----------------
# If you previously used two-stage flows, these keep imports working.

_EVIDENCE_SYSTEM = (
    "You are a helpful assistant that extracts verbatim evidence. "
    "Return STRICT JSON with an array under key 'quotes'. Each quote should include 'text'."
)

_EVIDENCE_USER_TMPL = """Return JSON of this shape:
{"quotes":[{"text":"<verbatim sentence or short passage from excerpt>"}]}

Requirement:
{requirement}

Excerpt:
{excerpt}
"""

class EvidenceExtractor:
    """
    Legacy extractor stub: extracts quotes as JSON.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.log = logger or logging.getLogger(__name__)
        try:
            self.client = _OpenAIClient(self.model_name, logger=self.log)
        except Exception:
            # If no key, keep a None client; caller's dry-run should bypass calling us.
            self.client = None

    def run(self, requirement: str, context: str) -> Dict[str, Any]:
        if not self.client:
            # Safe fallback for dry-run
            return {"quotes": []}
        raw = self.client.chat_json(system=_EVIDENCE_SYSTEM, user=_EVIDENCE_USER_TMPL.format(requirement=requirement, excerpt=context))
        data = _extract_json_block(raw)
        quotes = data.get("quotes") or []
        # Normalize to [{"text": "..."}]
        out = []
        for q in quotes:
            if isinstance(q, dict) and q.get("text"):
                out.append({"text": str(q["text"])})
            elif isinstance(q, str):
                out.append({"text": q})
        return {"quotes": out}

_ASSESS_SYSTEM = (
    "You are a compliance assessor. Decide coverage using ONLY provided quotes."
)

_ASSESS_USER_TMPL = """Return a JSON object with keys:
- "verdict": one of ["Covered","Partially Covered","Not Covered","Information Not Found"]
- "confidence": integer 1-5
- "rationale": <= 80 tokens

Requirement:
{requirement}

Quotes JSON:
{quotes_json}
"""

class VerdictAssessor:
    """
    Legacy assessor stub: takes quotes and returns coverage.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.log = logger or logging.getLogger(__name__)
        try:
            self.client = _OpenAIClient(self.model_name, logger=self.log)
        except Exception:
            self.client = None

    def run(self, requirement: str, quotes: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            # Safe fallback for dry-run
            if quotes.get("quotes"):
                return {"verdict": "Partially Covered", "confidence": 2, "rationale": "Heuristic — quotes present."}
            return {"verdict": "Information Not Found", "confidence": 1, "rationale": "Heuristic — no quotes."}
        payload = _ASSESS_USER_TMPL.format(requirement=requirement, quotes_json=json.dumps(quotes, ensure_ascii=False))
        raw = self.client.chat_json(system=_ASSESS_SYSTEM, user=payload)
        data = _extract_json_block(raw)
        verdict = data.get("verdict") or "Information Not Found"
        conf = data.get("confidence") or 2
        try:
            conf = int(conf)
        except Exception:
            conf = 2
        return {
            "verdict": verdict,
            "confidence": min(5, max(1, conf)),
            "rationale": _clip_rationale(data.get("rationale")),
        }
