import re
from typing import Dict, List, Optional
from .models import GeoDecision, RegulationHit

def to_similarity(distance_or_score: float, score_is_distance: bool = True) -> float:
    """
    Convert Qdrant score to similarity in [0,1].
    - If score_is_distance=True (typical cosine distance), map as 1 - clamp01(d).
    - If it's already a similarity, just clamp.
    """
    s = float(distance_or_score or 0.0)
    if score_is_distance:
        d = max(0.0, min(1.0, s))  # clamp distance
        return 1.0 - d
    return max(0.0, min(1.0, s))   # already similarity

def build_reg_hits(scored, score_is_distance: bool = True) -> List[RegulationHit]:
    """
    Build hits from Qdrant results, using ONLY reg_id in payload.
    Be robust to reg_id being nested: payload['reg_id'] OR payload['metadata']['reg_id'].
    """
    hits: List[RegulationHit] = []
    for doc, score in scored:
        md = getattr(doc, "metadata", {}) or {}
        # robust extraction
        rid: Optional[str] = (
            md.get("reg_id")
            or (md.get("metadata") or {}).get("reg_id")
            or md.get("id")
            or ""
        )
        snippet = (getattr(doc, "page_content", None) or md.get("text") or md.get("snippet") or "")[:500]
        hits.append(RegulationHit(reg_id=rid, snippet=snippet, score=to_similarity(score, score_is_distance)))
    return hits

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
    pats = (cfg or {}).get("abstain_patterns", {})
    def _has_any(key: str) -> bool:
        return any(re.search(p, t, re.I) for p in (pats.get(key) or []))
    amb = _has_any("ambiguous_geo")
    minors = any(re.search(p, t, re.I) for p in (cfg.get("minors_hints") or []))
    pf = any(re.search(p, t, re.I) for p in (cfg.get("pf_hints") or []))
    csam = any(re.search(p, t, re.I) for p in (cfg.get("csam_hints") or []))
    has_legal_words = any(w in t for w in ["comply", "law", "regulation", "statute", "act"])
    return bool(amb and not (minors or pf or csam or has_legal_words))

def canonicalize_regs(regs: List[str], allowed: List[str], canon_map: Dict[str, str]) -> List[str]:
    """
    Map synonyms via policy.yaml then filter to allowed set. Case/whitespace/char-normalized.
    """
    out = []
    allowed_set = set(allowed or [])
    canon = canon_map or {}
    for r in regs or []:
        x = re.sub(r"[^a-z0-9\s_]", "", (r or "").strip().lower())
        x = re.sub(r"\s+", " ", x).strip()
        mapped = canon.get(x, x)
        if mapped in allowed_set:
            out.append(mapped)
    return sorted(set(out))

def evidence_gate(decision: GeoDecision, hits: List[RegulationHit], min_sim: float, allowed: List[str]) -> GeoDecision:
    """
    Keep only regs that have retrieval support >= min_sim.
    """
    best: Dict[str, float] = {}
    for h in hits:
        if not h.reg_id:
            continue
        best[h.reg_id] = max(best.get(h.reg_id, 0.0), float(h.score))
    keep = [r for r in (decision.regulations or [])
            if r in set(allowed or []) and best.get(r, 0.0) >= float(min_sim)]
    decision.regulations = sorted(set(keep))
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
    if (re.search(r"\b(california|\bca\b)\b", t)
        and re.search(r"\b(teen|minor)s?\b|\b1[3-7]\b", t)
        and re.search(r"\b(recommend|feed|personaliz)", t)):
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.85)
        if "california_kids_act" not in decision.regulations:
            decision.regulations.append("california_kids_act")

    # Business-only holdback/performance gives false
    if (re.search(r"\b(holdback|market testing|performance reason|a/b)\b", t)
        and not re.search(r"\b(comply|law|regulat)", t)):
        decision.requires_geo_logic = False
        decision.confidence = max(decision.confidence, 0.90)
        decision.regulations = []

    decision.regulations = sorted(set(decision.regulations))
    return decision