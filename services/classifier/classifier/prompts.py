import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

PROMPT_TMPL = """You are a compliance triage assistant. Output ONLY one JSON object per {format_instructions}. No prose/markdown/fences.

Inputs:
- Title: {title}
- Description: {description}
- Docs: {docs}
- Retrieval (supportive only; never creates regs): {evidence}

Decision (apply in order):
1) EXPLICIT HOOK → requires_geo_logic=true
   - Hook = (law/regulator/article) OR (jurisdiction + minors + curfew/parental-controls) OR
     (jurisdiction + minors + PF/recs) OR explicit CSAM/NCMEC/2258A terms.
   - Retrieval may help choose among allowed regs but MUST NOT create a reg.

2) BUSINESS-ONLY → requires_geo_logic=false
   - If clearly business-only (A/B, experiment, holdback, feature flag, staged rollout, performance/autoplay,
     UI/UX/layout/theme, leaderboard, payouts, upload limits, friend suggestions, avatars/GIFs, video replies)
     AND no explicit hook.

3) AMBIGUOUS GEO INTENT ONLY → requires_geo_logic=null
   - Geo phrases without a hook and not clearly product-only (e.g., “global except KR”, “EU-only”, “exclude JP”).

Regulations:
- Must be subset of {allowed_regs} (IDs only). Set ONLY when requires_geo_logic=true and the hook appears in
  Title/Description/Docs (not retrieval). Else [].

Fields:
- reasoning (≤ 10 words): For false: DO NOT mention laws/regulators/jurisdictions; explain product-only. For null: explain geo-ambiguity
- confidence: 0–1; be conservative when unsure.
"""

def build_prompt(allowed_ids, parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(PROMPT_TMPL).partial(
        format_instructions=parser.get_format_instructions(),
        allowed_regs=json.dumps(sorted(allowed_ids or [])),
    )

def format_evidence(reg_hits) -> str:
    lines = []
    for h in reg_hits:
        rid = h.reg_id or "<missing>"
        snip = (h.snippet or "").replace("\n", " ").strip()[:500]
        lines.append(f"[{rid} | score={h.score:.3f}] {snip}")
    return "\n\n".join(lines) or "(no retrieval evidence)"