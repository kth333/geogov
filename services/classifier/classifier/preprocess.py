import json
import os
import re
from functools import lru_cache
from typing import List, Tuple

TERMS_PATH = os.getenv("TERMINOLOGY_PATH", "/app/data/terminology.json")

@lru_cache(maxsize=1)
def _load_terms():
    try:
        with open(TERMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def expand_terms(text: str) -> str:
    """Inline-glossary: replace whole-word jargon with 'TERM (meaning)'. Safe no-op if file missing."""
    terms = _load_terms()
    out = text or ""
    for k, v in terms.items():
        kk = re.escape(k)
        out = re.sub(rf"\b{kk}\b", f"{k} ({v})", out)
    return out

def preprocess_payload(title: str, description: str, docs: List[str]) -> Tuple[str, str, List[str]]:
    return expand_terms(title or ""), expand_terms(description or ""), [expand_terms(d) for d in (docs or [])]