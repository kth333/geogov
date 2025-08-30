import os
import re
import json
import yaml
import logging
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from .models import GeoDecision
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

def _load_policy() -> Dict[str, Any]:
    path = os.getenv("POLICY_PATH", "/app/data/policy.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _make_embeddings():
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)

def _make_vectorstore(cfg: Dict[str, Any]):
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    collection = (cfg.get("retrieval") or {}).get("collection", "regs")
    top_k = int((cfg.get("retrieval") or {}).get("top_k", 6))
    vs = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=_make_embeddings(),
        content_payload_key="text",
        metadata_payload_key=None,
    )
    return client, vs, top_k

def _make_llm():
    model = os.getenv("MODEL_NAME", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.2,
        keep_alive="30m",
        num_ctx=1024,
        num_predict=220,
        num_thread=0,
    )

class RAGClassifier:
    def __init__(self):
        self.cfg = _load_policy()
        th = self.cfg.get("thresholds") or {}
        self.allowed_ids: List[str] = self.cfg.get("regulations_allowed") or ALLOWED_FALLBACK
        self.canon_map: Dict[str, str] = self.cfg.get("regulation_synonyms") or {}
        self.min_conf = float(th.get("confidence_min", 0.45))
        self.min_sim = float(th.get("reg_evidence_min_sim", 0.00))
        self.qdrant_client, self.vectorstore, self.top_k = _make_vectorstore(self.cfg)
        self.llm = _make_llm()
        self.parser = PydanticOutputParser(pydantic_object=GeoDecision)
        self.prompt = build_prompt(self.allowed_ids, self.parser)

    def _retrieve(self, text: str):
        try:
            scored = self.vectorstore.similarity_search_with_score(text, k=self.top_k)
            try:
                log.info("QDRANT HITS:")
                for d, s in scored:
                    log.info("%s", {"score": float(s), "payload": getattr(d, "metadata", {})})
            except Exception:
                pass
            return scored
        except Exception as e:
            log.exception("vector search failed: %s", e)
            return []

    def _parse_or_repair(self, raw) -> GeoDecision:
        text = getattr(raw, "content", None)
        if text is None:
            text = str(raw)
        try:
            return self.parser.parse(text)
        except ValidationError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return self.parser.parse(m.group(0))
            return GeoDecision(
                requires_geo_logic=None,
                reasoning="Could not parse model JSON",
                confidence=0.0,
                regulations=[],
            )

    def _rules_only(self, feature_text: str, hits: List) -> GeoDecision:
        d = GeoDecision(requires_geo_logic=None, reasoning="Rules-only fallback", confidence=0.5, regulations=[])
        d = apply_rule_overrides(feature_text, d)
        d.regulations = canonicalize_regs(d.regulations, self.allowed_ids, self.canon_map)
        d = evidence_gate(d, hits, self.min_sim, self.allowed_ids)
        if policy_abstain(feature_text, self.cfg):
            d.requires_geo_logic = None
            d.regulations = []
        return d

    def run(self, title: str, description: Optional[str], docs: List[str]) -> Dict[str, Any]:
        # 1) Preprocess
        title, description, docs = preprocess_payload(title or "", description or "", docs or [])
        feature_text = " ".join([title or "", description or "", " ".join(docs or [])]).strip()

        # 2) Retrieval
        scored = self._retrieve(feature_text)
        reg_hits = build_reg_hits(scored)
        log.info("EVIDENCE: %s", [{"reg_id": h.reg_id, "score": round(h.score, 3)} for h in reg_hits])
        evidence_text = format_evidence(reg_hits)      # human-readable evidence block

        # 3) Prompt + LLM
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

        # 4) Policy-driven post-processing
        decision = apply_rule_overrides(feature_text, decision)
        decision.regulations = canonicalize_regs(decision.regulations, self.allowed_ids, self.canon_map)
        log.info("LLM regs (pre): %s", decision.regulations)
        decision = evidence_gate(decision, reg_hits, self.min_sim, self.allowed_ids)
        log.info("LLM regs (post gate): %s", decision.regulations)

        if policy_abstain(feature_text, self.cfg):
            decision.requires_geo_logic = None
            decision.regulations = []

        if decision.requires_geo_logic is not None and float(decision.confidence) < self.min_conf:
            decision.requires_geo_logic = None

        log.info("FINAL DECISION: %s", decision.model_dump())
        return decision.model_dump()