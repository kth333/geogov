import re
import logging
from typing import Dict, List, Optional
from .models import GeoDecision, RegulationHit

logger = logging.getLogger("rules")

def to_similarity(distance_or_score: float, score_is_distance: bool = True) -> float:
    """
    Convert Qdrant score to similarity in [0,1].

    If score_is_distance=True (cosine distance), Qdrant can yield up to ~2.0 if vectors
    aren't strictly unit-normalized. Map via: sim = 1 - clamp(d, 0..2)/2.
    If the value is already a similarity, clamp to [0,1].
    """
    s = float(distance_or_score or 0.0)
    if score_is_distance:
        d = max(0.0, min(2.0, s))
        return 1.0 - (d / 2.0)
    # already similarity
    return max(0.0, min(1.0, s))

def build_reg_hits(scored, score_is_distance: bool = True, snippet_len: int = 160) -> List[RegulationHit]:
    """
    Build hits from Qdrant results, using ONLY reg_id in payload.
    Be robust to reg_id being nested or absent:
      - doc.metadata['reg_id']
      - doc.metadata['metadata']['reg_id']
      - doc.metadata['id']
    """
    hits = []
    for doc, score in scored:
        md = getattr(doc, "metadata", {}) or {}
        rid = (
            md.get("reg_id")
            or (md.get("metadata") or {}).get("reg_id")
            or md.get("id")
            or md.get("_id")
            or ""
        )
        snippet = (
            getattr(doc, "page_content", None)
            or md.get("text")
            or md.get("snippet")
            or ""
        )[:snippet_len]
        hits.append(RegulationHit(
            reg_id=rid,
            snippet=snippet,
            score=to_similarity(score, score_is_distance),
        ))
    return hits

# rules.py
def policy_abstain(text: str, cfg: Dict) -> bool:
    """
    Use policy.yaml patterns. Expected shape:
      abstain_patterns:
        ambiguous_geo: ["\\bglobal except\\b", ...]
      minors_hints: [...]
      pf_hints: [...]
      csam_hints: [...]
    """
    t = (text or "").lower()
    pats = (cfg or {}).get("abstain_patterns", {}) or {}

    def _has_any(key: str) -> bool:
        return any(re.search(p, t, re.I) for p in (pats.get(key) or []))

    amb = _has_any("ambiguous_geo")
    minors = any(re.search(p, t, re.I) for p in (cfg.get("minors_hints") or []))
    pf = any(re.search(p, t, re.I) for p in (cfg.get("pf_hints") or []))
    csam = any(re.search(p, t, re.I) for p in (cfg.get("csam_hints") or []))
    has_legal_words = bool(re.search(r"\b(comply|law|regulation|statute|act)\b", t))
    return bool(amb and not (minors or pf or csam or has_legal_words))

def _canon_one(x: str, canon_map: Dict[str, str]) -> str:
    y = re.sub(r"[^a-z0-9\s_]", "", (x or "").strip().lower())
    y = re.sub(r"\s+", " ", y).strip()
    return (canon_map or {}).get(y, y)

def canonicalize_regs(regs: List[str], allowed: List[str], canon_map: Dict[str, str]) -> List[str]:
    """
    Map synonyms via policy.yaml then filter to allowed set. Case/whitespace/char-normalized.
    """
    allowed_set = set(allowed or [])
    out = []
    for r in regs or []:
        mapped = _canon_one(r, canon_map)
        if mapped in allowed_set:
            out.append(mapped)
    return sorted(set(out))

def evidence_gate(
    decision: GeoDecision,
    hits: List[RegulationHit],
    min_sim: float,
    allowed: List[str],
    canon_map: Dict[str, str] | None = None,
    max_regs: Optional[int] = 1,          # cap kept regs
    tie_eps: float = 0.02,                # allow near-ties
) -> GeoDecision:
    allowed_set = set(allowed or [])
    canon_map = canon_map or {}

    # Aggregate best similarity per canonicalized hit id
    best: Dict[str, float] = {}
    for h in hits:
        if not h.reg_id:
            continue
        hid = _canon_one(h.reg_id, canon_map)
        if hid in allowed_set:
            best[hid] = max(best.get(hid, 0.0), float(h.score))

    if not best:
        logger.warning("GATE found no usable reg_id in hits; skipping drop to avoid nuking regs.")
        return decision

    proposed = decision.regulations or []
    scored = [(r, best.get(r, 0.0)) for r in proposed if r in allowed_set]
    scored = [(r, s) for r, s in scored if s >= float(min_sim)]
    scored.sort(key=lambda x: x[1], reverse=True)

    if max_regs is not None and len(scored) > max_regs:
        cutoff = scored[max_regs - 1][1] - tie_eps
        scored = [rs for rs in scored if rs[1] >= cutoff][:max_regs]

    decision.regulations = [r for r, _ in scored]
    return decision

def apply_rule_overrides(feature_text: str, decision: GeoDecision) -> GeoDecision:
    """
    Lightweight rule nudges. Conservative & auditable.
    """
    t = (feature_text or "").lower()

    # CSAM reporting to us_ncmec_reporting
    if re.search(r"\b(cs[ab]m|ncmec|cybertip(line)?)\b", t):
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.90)
        if "us_ncmec_reporting" not in decision.regulations:
            decision.regulations.append("us_ncmec_reporting")

    # California minors + personalized feed = california_kids_act
    if (
        re.search(r"\b(california|\bca\b)\b", t)
        and re.search(r"\b(teen|minor)s?\b|\b1[3-7]\b", t)
        and re.search(r"\b(recommend|feed|personaliz)", t)
    ):
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.85)
        if "california_kids_act" not in decision.regulations:
            decision.regulations.append("california_kids_act")

    # Business-only holdback/performance gives false
    if (
        re.search(r"\b(holdback|market testing|performance reason|a/b)\b", t)
        and not re.search(r"\b(comply|law|regulat)", t)
    ):
        decision.requires_geo_logic = False
        decision.confidence = max(decision.confidence, 0.90)
        decision.regulations = []

    decision.regulations = sorted(set(decision.regulations))
    return decision