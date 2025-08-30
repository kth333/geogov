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
    bo_hints = (cfg or {}).get("business_only_hints") or []

    def _any(patterns):
        return any(re.search(p, t, re.I) for p in (patterns or []))

    ambiguous_geo = _any(pats.get("ambiguous_geo"))
    has_legal_words = bool(re.search(r"\b(comply|compliance|law|regulat(ion|ory)|statute|act)\b", t, re.I))
    minors = _any((cfg.get("minors_hints") or []))
    pf     = _any((cfg.get("pf_hints") or []))
    csam   = _any((cfg.get("csam_hints") or []))

    return bool(ambiguous_geo and not (minors or pf or csam or has_legal_words))

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
    max_regs: Optional[int] = 1,
    tie_eps: float = 0.02,
    allow_fallback_when_empty: bool = False,   # <-- NEW
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

    if not best and allowed_set:
        logger.warning("GATE: no usable reg_id in hits within allowed scope.")
    
    proposed = decision.regulations or []

    # If model proposed nothing:
    if not proposed:
        # Only allow retrieval to pick regs when EXPLICIT hook exists AND scope is non-empty
        if allow_fallback_when_empty and allowed_set and best:
            top = sorted(best.items(), key=lambda x: x[1], reverse=True)
            kept = [r for r, s in top if s >= float(min_sim)]
            if max_regs is not None:
                kept = kept[:max_regs]
            decision.regulations = kept
        else:
            decision.regulations = []
        return decision

    # Otherwise, score model-proposed regs against retrieval evidence
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
    Hard, auditable triggers so regs only attach when the legal hook is explicit.
    """
    t = (feature_text or "").lower()

    # Signals
    has_minors   = bool(re.search(r"\b(minor|teen|under\s?1[3-8]|13-1[6-9]|age[-\s]?gate|parental)\b", t, re.I))
    has_pf       = bool(re.search(r"\b(pf|personaliz(e|ation|ed)|recommend(er|ation)s?)\b", t, re.I))
    has_controls = bool(re.search(r"\b(parental|consent|age[-\s]?verify|verification|guardian|oversight|restrict|control)\b", t, re.I))
    has_curfew   = bool(re.search(r"\b(curfew|time[-\s]?limit|quiet\s*hours|night|after\s*\d{1,2}\s*(pm|am)|login\s*block|shutdown)\b", t, re.I))

    in_utah      = bool(re.search(r"\butah\b", t, re.I))
    in_florida   = bool(re.search(r"\bflorida\b", t, re.I))
    in_calif = bool(re.search(r'\bcalifornia\b', t, re.I))

    has_dsa      = bool(re.search(r"\b(dsa|digital\s+services\s+act|vlop|very\s+large\s+online\s+platform|article\s?(16|17|28|34|39))\b", t, re.I))
    has_csam     = bool(re.search(r"\b(cs[ab]m|child\s+sexual\s+abuse|ncmec|cybertip(?:line)?|2258a)\b", t, re.I))

    business_only = bool(re.search(
        r"(a/?b|a-?b)\s*(test|testing)|\bexperiment\b|\bholdback\b|\bmarket\s+testing\b|\bstaged\s+rollout\b|\bfeature\s+flag\b|\bgeo-?fence(?:s|d)?\b",
        t, re.I
    ))

    amb_geo = bool(re.search(
        r"(global(?:ly)?|worldwide|all\s*regions)\s+except\b"
        r"|(?:except|excluding)\s+(kr|korea|jp|japan|uk|gb|eu|eea|us|ca|au|in)\b"
        r"|(?:eu|eea|uk|us|ca|kr|jp)\s*[- ]?only\b"
        r"|only\s+in\s+(eu|eea|uk|us|ca|kr|jp)\b",
        t, re.I
    ))

    # Start from current regs, but we only add when hooks are explicit
    current = set(decision.regulations or [])

    # --- Pure product/A-B/UX → requires_geo_logic = False, no regs ---
    # Only when no explicit legal hook is present
    if business_only and not (has_dsa or has_csam or (in_calif and has_minors and has_pf) or
                              (in_utah and has_minors and (has_curfew or has_controls)) or
                              (in_florida and has_minors and has_controls)):
        decision.requires_geo_logic = False
        decision.regulations = []
        return decision

    # --- NCMEC only when CSAM/NCMEC/2258A explicit ---
    if has_csam:
        decision.requires_geo_logic = True
        current.add("us_ncmec_reporting")

    # --- California Kids Act only when CA + minors + PF/recommenders ---
    if in_calif and has_minors and has_pf:
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.85)
        current.add("california_kids_act")

    # --- Utah only when Utah + minors + curfews/controls ---
    if in_utah and has_minors and (has_curfew or has_controls):
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.80)
        current.add("utah_social_media_regulation")

    # --- Florida only when Florida + minors controls ---
    if in_florida and has_minors and has_controls:
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.80)
        current.add("florida_online_protections")

    # --- DSA only when actually about DSA compliance ---
    if has_dsa:
        decision.requires_geo_logic = True
        decision.confidence = max(decision.confidence, 0.80)
        current.add("dsa")

    # --- No explicit legal hook ---
    explicit_hook = (
        has_csam
        or has_dsa
        or (in_calif and has_minors and has_pf)
        or (in_utah and has_minors and (has_curfew or has_controls))
        or (in_florida and has_minors and has_controls)
    )

    if not explicit_hook and not current:
        if has_minors:
            # minors mentioned but no geo/legal hook -> null
            decision.requires_geo_logic = None
        else:
            # pure product/ops/UX with no legal hook -> false
            decision.requires_geo_logic = False
        decision.regulations = []
        return decision
    
    if amb_geo and not current:
        # No explicit legal hook; intention unclear → human review
        decision.requires_geo_logic = None
        decision.regulations = []
        return decision
    
    decision.regulations = sorted(current)
    return decision