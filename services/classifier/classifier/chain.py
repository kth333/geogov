import os
import re
import json
import yaml
import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_qdrant import Qdrant as LCQdrant
from langchain_community.chat_models import ChatOllama

from .models import GeoDecision, RegulationHit
from .preprocess import preprocess_payload
from .prompts import build_prompt, format_evidence
from .rules import (
    build_reg_hits,
    evidence_gate,
    policy_abstain,
    canonicalize_regs,
    apply_rule_overrides,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("classifier")

ALLOWED_FALLBACK = [
    "dsa",
    "california_kids_act",
    "florida_online_protections",
    "utah_social_media_regulation",
    "us_ncmec_reporting",
]

# -------------------- helpers --------------------

def _make_embeddings():
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)

def _make_vectorstore(cfg: Dict[str, Any]):
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    collection = (cfg.get("retrieval") or {}).get("collection", "regs")
    top_k = int((cfg.get("retrieval") or {}).get("top_k", 6))
    vs = LCQdrant(
        client=client,
        collection_name=collection,
        embeddings=_make_embeddings(),
        content_payload_key="text",     # maps payload['text'] -> Document.page_content
        metadata_payload_key="metadata"
    )
    return client, vs, top_k

def _make_llm():
    model = os.getenv("MODEL_NAME", "qwen2.5:3b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    # keep format="json" to bias pure JSON, but still robustly parse below
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.1,
        keep_alive="30m",
        num_ctx=1024,
        num_predict=96,
        num_thread=0,
        format="json",
    )

def _load_policy() -> Dict[str, Any]:
    path = os.getenv("POLICY_PATH", "/app/data/policy.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("Policy file not found at %s; proceeding with defaults.", path)
        return {}
    except Exception as e:
        log.exception("Failed to load policy file %s: %s", path, e)
        return {}

def _support_by_reg(hits: List[RegulationHit], canon_map: Dict[str, str] | None = None) -> Dict[str, float]:
    """
    Aggregate best similarity per (optionally) canonicalized reg_id.
    """
    canon_map = canon_map or {}
    def _canon(x: str) -> str:
        y = re.sub(r"[^a-z0-9\s_]", "", (x or "").strip().lower())
        y = re.sub(r"\s+", " ", y).strip()
        return canon_map.get(y, y)

    best: Dict[str, float] = {}
    for h in hits:
        rid = getattr(h, "reg_id", "") or ""
        if not rid:
            continue
        hid = _canon(rid)
        best[hid] = max(best.get(hid, 0.0), float(h.score))
    return best

def _recalibrate_confidence(decision: GeoDecision, hits: List[RegulationHit], canon_map: Dict[str, str] | None = None) -> GeoDecision:
    """
    Blend LLM confidence with retrieval support and cap. Prevents 1.0 everywhere.
    Uses canonicalization so synonyms line up with decision.regulations.
    """
    best = _support_by_reg(hits, canon_map=canon_map)

    if not decision.regulations:
        decision.confidence = round(min(float(decision.confidence or 0.0), 0.60), 3)
        return decision

    sims = [best.get(r, 0.0) for r in decision.regulations]
    support = max(sims) if sims else 0.0  # strongest retrieved evidence for any kept reg
    mapped = 0.60 + 0.35 * support        # 0.60..0.95 band
    decision.confidence = round(min(float(decision.confidence or 0.0), mapped, 0.95), 3)
    return decision

# -------------------- main class --------------------

class RAGClassifier:
    def __init__(self):
        self.cfg = _load_policy()
        th = self.cfg.get("thresholds") or {}
        self.allowed_ids: List[str] = self.cfg.get("regulations_allowed") or ALLOWED_FALLBACK
        self.canon_map: Dict[str, str] = self.cfg.get("regulation_synonyms") or {}
        self.min_conf = float(th.get("confidence_min", 0.55))
        self.min_sim = float(th.get("reg_evidence_min_sim", 0.50))
        self.max_regs = int(th.get("regulations_max", 1))
        self.tie_eps  = float(th.get("gating_tie_eps", 0.02))
        self.qdrant_client, self.vectorstore, self.top_k = _make_vectorstore(self.cfg)
        self.llm = _make_llm()
        self.parser = PydanticOutputParser(pydantic_object=GeoDecision)
        self.prompt = build_prompt(self.allowed_ids, self.parser)

    # ---------- helpers ----------

    def _feature_id_for(self, title: str, description: str, docs: List[str]) -> str:
        raw = f"{title or ''}\n{description or ''}\n" + "\n".join(docs or [])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _lookup_feedback(self, feature_id: str) -> Tuple[Optional[bool], List[str]]:
        import sqlite3
        try:
            db = os.getenv("AUDIT_DB_PATH", "/app/outputs/audit.db")
            os.makedirs(os.path.dirname(db), exist_ok=True)  # ensure parent dir exists
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("""
              CREATE TABLE IF NOT EXISTS feedback(
                ts INTEGER, feature_id TEXT, title TEXT, description TEXT,
                docs_json TEXT, requires_geo_logic TEXT, regulations_json TEXT,
                comment TEXT, user TEXT
              )
            """)
            cur.execute(
                "SELECT requires_geo_logic, regulations_json FROM feedback WHERE feature_id=? ORDER BY ts DESC LIMIT 1",
                (feature_id,),
            )
            row = cur.fetchone()
            con.close()
            if not row:
                return None, []
            rgl = json.loads(row[0]) if row[0] else None
            regs = json.loads(row[1]) if row[1] else []
            return rgl, regs
        except Exception as e:
            log.exception("feedback lookup failed: %s", e)
            return None, []

    def _retrieve(self, text: str):
        try:
            scored = self.vectorstore.similarity_search_with_score(text, k=self.top_k)
            try:
                log.info("QDRANT HITS:")
                for d, s in scored:
                    # This shows what Document.metadata actually contains
                    log.info("%s", {"score": float(s), "metadata": getattr(d, "metadata", {}), "page_len": len(getattr(d, "page_content", "") or "")})
                for d, s in scored:
                    log.info("DEBUG payload->metadata keys=%s sample_reg=%s",
                            list((getattr(d, "metadata", {}) or {}).keys()),
                            ((getattr(d, "metadata", {}) or {}).get("reg_id")
                            or ((getattr(d, "metadata", {}) or {}).get("metadata") or {}).get("reg_id")))
            except Exception:
                pass
            return scored
        except Exception as e:
            log.exception("vector search failed: %s", e)
            return []

    def _parse_or_repair(self, raw) -> GeoDecision:
        text = getattr(raw, "content", None) or str(raw)

        # Primary: parser (expects a dict)
        try:
            return self.parser.parse(text)
        except Exception:
            pass

        # If model returned JSON string/list, handle that
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return GeoDecision.model_validate(obj)
        except Exception:
            pass

        # Last-ditch: extract first JSON object
        m = re.search(r"\{.*?\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return GeoDecision.model_validate(obj)
            except Exception:
                pass

        return GeoDecision(
            requires_geo_logic=None,
            reasoning="Could not parse model JSON",
            confidence=0.0,
            regulations=[],
        )

    def _rules_only(self, feature_text: str, hits: List[RegulationHit]) -> GeoDecision:
        d = GeoDecision(requires_geo_logic=None, reasoning="Rules-only fallback", confidence=0.5, regulations=[])
        d = apply_rule_overrides(feature_text, d)
        d.regulations = canonicalize_regs(d.regulations, self.allowed_ids, self.canon_map)

        # Gate only if hits have usable reg_ids
        if any(h.reg_id for h in hits):
            d = evidence_gate(d, hits, self.min_sim, self.allowed_ids)
        else:
            log.warning("GATE found no usable reg_id in hits; skipping drop in rules-only.")

        if policy_abstain(feature_text, self.cfg):
            d.requires_geo_logic = None
            d.regulations = []
        return d
    
    def _scope_allowed(self, text: str) -> List[str]:
        t = (text or "").lower()
        scope = set()

        if re.search(r"\butah\b", t):
            scope.add("utah_social_media_regulation")
        if re.search(r"\bflorida\b", t):
            scope.add("florida_online_protections")
        if re.search(r"\bcalifornia|\bca\b", t):
            scope.add("california_kids_act")
        if re.search(r"\b(eu|eea|europe|dsa|vlop)\b", t):
            scope.add("dsa")
        if re.search(r"\b(cs[ab]m|ncmec|cybertip(line)?|2258a|united states|us\b)", t):
            scope.add("us_ncmec_reporting")

        # If nothing matched, allow all (status quo)
        return sorted(scope or self.allowed_ids)

    def _compact_evidence(self, hits, per_reg=1, max_total=3, snippet_len=120) -> str:
        # best hit per reg_id
        best = {}
        for h in hits:
            rid = (h.reg_id or "").strip()
            if not rid:
                continue
            prev = best.get(rid)
            if prev is None or h.score > prev.score:
                best[rid] = RegulationHit(reg_id=rid, score=h.score, snippet=(h.snippet or "")[:snippet_len])

        # top regs overall
        top = sorted(best.values(), key=lambda x: x.score, reverse=True)[:max_total]
        # super-compact tab-separated lines; VERY few tokens
        return "\n".join(f"{h.reg_id}\t{h.score:.3f}\t{h.snippet}" for h in top)

    # ---------- main ----------

    def run(self, title: str, description: Optional[str], docs: List[str]) -> Dict[str, Any]:
        # 1) Preprocess
        title, description, docs = preprocess_payload(title or "", description or "", docs or [])
        feature_text = " ".join([title or "", description or "", " ".join(docs or [])]).strip()

        # 2) Retrieval (so we can gate feedback too)
        scored = self._retrieve(feature_text)
        score_is_distance = bool((self.cfg.get("retrieval") or {}).get("score_is_distance", True))
        reg_hits = build_reg_hits(scored, score_is_distance=score_is_distance, snippet_len=160)
        log.info("EVIDENCE: %s", [{"reg_id": getattr(h, 'reg_id', ''), "score": round(h.score, 3)} for h in reg_hits])
        evidence_text = self._compact_evidence(reg_hits, max_total=3, snippet_len=120)

        # 3) Human feedback (if present) → early return
        feature_id = self._feature_id_for(title, description, docs)
        rgl_fix, regs_fix = self._lookup_feedback(feature_id)
        if rgl_fix is not None or regs_fix:
            decision = GeoDecision(
                requires_geo_logic=rgl_fix,
                reasoning="Applied confirmed human feedback.",
                confidence=0.99,
                regulations=regs_fix or [],
            )
            decision.regulations = canonicalize_regs(decision.regulations, self.allowed_ids, self.canon_map)

            if any(h.reg_id for h in reg_hits):
                allowed_scoped = self._scope_allowed(feature_text)
                decision = evidence_gate(
                    decision, reg_hits, self.min_sim, allowed_scoped,
                    self.canon_map, max_regs=self.max_regs, tie_eps=self.tie_eps
                )
            else:
                log.warning("GATE found no usable reg_id in hits; skipping drop (feedback path).")

            decision = _recalibrate_confidence(decision, reg_hits, self.canon_map)
            log.info("FINAL DECISION (feedback): %s", decision.model_dump())
            return decision.model_dump()

        # 4) Prompt + LLM
        try:
            raw = (self.prompt | self.llm).invoke({
                "title": title,
                "description": description,
                "docs": docs,
                "evidence": evidence_text,
            })
            decision = self._parse_or_repair(raw)
        except Exception as e:
            log.exception("LLM path failed, using rules-only: %s", e)
            decision = self._rules_only(feature_text, reg_hits)

        # 5) Policy-driven post-processing
        decision = apply_rule_overrides(feature_text, decision)
        decision.regulations = canonicalize_regs(decision.regulations, self.allowed_ids, self.canon_map)
        log.info("LLM regs (pre): %s", decision.regulations)

        if any(h.reg_id for h in reg_hits):
            allowed_scoped = self._scope_allowed(feature_text)            
            decision = evidence_gate(
                decision, reg_hits, self.min_sim, allowed_scoped,
                self.canon_map, max_regs=self.max_regs, tie_eps=self.tie_eps
            )
        else:
            log.warning("GATE found no usable reg_id in hits; skipping drop to avoid nuking regs.")
        log.info("LLM regs (post gate): %s", decision.regulations)

        # Confidence calibration (prevents 1.0 everywhere)
        decision = _recalibrate_confidence(decision, reg_hits, self.canon_map)

        # Abstain & min-confidence enforcement
        if policy_abstain(feature_text, self.cfg):
            decision.requires_geo_logic = None
            decision.regulations = []

        if decision.requires_geo_logic is not None and float(decision.confidence) < self.min_conf:
            decision.requires_geo_logic = None

        log.info("FINAL DECISION: %s", decision.model_dump())
        return decision.model_dump()