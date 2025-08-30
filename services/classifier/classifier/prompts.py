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
   - Retrieval can help choose among allowed regs but MUST NOT create a reg if inputs lack a hook.

2) BUSINESS-ONLY → requires_geo_logic=false
   - If clearly business related, e.g., A/B, experiment, holdback, feature flag, staged rollout, performance/autoplay,
     UI/UX/layout/theme, leaderboard, payouts, upload limits, friend suggestions, avatars/GIFs,
     video replies — and no explicit hook.

3) AMBIGUOUS GEO INTENT ONLY → requires_geo_logic=null
   - Geo constraints (e.g., “global except KR”, “EU-only”, “exclude JP”) with no explicit hook and
     not clearly product-only.

Regulations:
- Subset of {allowed_regs}, exact IDs. Only set if requires_geo_logic=true **and** the hook appears
  in Title/Description/Docs (not retrieval). Else [].

Fields:
- reasoning (≤ 25 words): If false/null, do NOT mention laws, regulators, or jurisdictions; give a
  product-only rationale. 
- confidence: 0–1, conservative when unsure.
"""

def build_prompt(allowed_ids, parser: PydanticOutputParser) -> ChatPromptTemplate:
    """Recreate the original ChatPromptTemplate with format_instructions + allowed regs."""
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