import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

PROMPT_TMPL = """You are a compliance triage assistant.

Return ONLY a single JSON object matching this schema:
{format_instructions}
No prose, no markdown, no code fences, no backticks, no lists/arrays at the top level.

Inputs:
- Feature Title: {title}
- Description: {description}
- Docs: {docs}
- Retrieved Evidence (regulations): {evidence}

Decision rules:
- TRUE only if text indicates a legal obligation (e.g., minors + personalized feed, CSAM reporting, or explicit 'to comply with ...').
- FALSE if clearly business-only (A/B test, holdback, performance, geofence for market testing) with no safety/legal cues.
- NULL if geography is mentioned but the intent/legal trigger is unclear (e.g., 'global except KR' without rationale).
- Regulations must be a subset of: {allowed_regs}. Omit if no evidence.
- When listing "regulations", return the REG_ID exactly as shown (e.g., dsa, california_kids_act).

Rules for `reasoning`:
- Use your own words; DO NOT quote or closely paraphrase the Title/Description.
- Base any regulation claims ONLY on the provided evidence block; if none, set requires_geo_logic=null and regulations=[].

Be concise in reasoning. Set confidence 0–1.
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