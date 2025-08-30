# Project Name (to be decided)

**TikTok TechJam 2025 Track 3 Submission**

Given a feature title, description, and docs, this project flags whether **geo-specific compliance logic** is required, explains **why**, and names the **related regulation families**

---

## Demo links

---

## Problem fit

Shipping global features means navigating region-specific obligations (minors, CSAM reporting, recommender transparency, etc.). Today, detection is manual and inconsistent. GeoGov turns that into a **repeatable, auditable step**:

* **Reduce governance cost:** automatic first-pass triage, less manual review.
* **Mitigate exposure:** catch compliance triggers before launch.
* **Enable audits:** structured, exportable decisions for every feature.

---

## What the system returns

For each feature, GeoGov emits a compact, structured decision:

```json
{
  "requires_geo_logic": true | false | null,
  "reasoning": "short explanation",
  "confidence": 0.0-1.0,
  "regulations": ["dsa", "california_kids_act", ...]
}
```

* **true** → clearly compliance-driven (e.g., “California teens + personalized feed default off”).
* **false** → clearly business-only (A/B, perf test) with no legal cues.
* **null** → ambiguous (e.g., “global except KR” with no stated rationale) → human review.

Supported regulation families (focused for the hackathon brief):

* `dsa` — EU Digital Services Act
* `california_kids_act` — CA Protecting Our Kids from Social Media Addiction Act
* `florida_online_protections` — Florida Online Protections for Minors
* `utah_social_media_regulation` — Utah Social Media Regulation Act
* `us_ncmec_reporting` — US CSAM reporting to NCMEC

---

## How it works (architecture)

**Retrieval-Augmented + Rules-Assisted**

1. **Retrieve**: Vector search in Qdrant over concise regulation notes (one file per `reg_id`).
2. **Decide**: LLM (Ollama local model) returns valid JSON with reasoning + confidence.
3. **Rule assists**: Deterministic nudges for explicit triggers (e.g., “NCMEC”, “California + teens + PF”).
4. **Evidence gate**: Keep a regulation only if it’s **allowed** *and* **retrieved** with enough similarity.
5. **Policy abstain**: Mark **null** for ambiguous geo mentions with no legal signals.
6. **Audit**: Log Qdrant hits + final decision; batch runner emits CSV for records.

*Everything runs locally via Docker; no user data leaves the machine.*

---

## Quickstart

1) Environment Setup
```bash
# Copy the sample env file

# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

2) Open .env and set these values (or paste this block):

```bash
# Embeddings + Vector DB
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://qdrant:6333

# Policy file
POLICY_PATH=/app/data/policy.yaml

# Service URLs (used by the gateway)
CLASSIFIER_URL=http://classifier:8000

# LLM (Ollama local)
OLLAMA_BASE_URL=http://ollama:11434
MODEL_NAME=llama3

# Logging
DEBUG=1
```
3) Build & start
```bash
docker compose up -d --build
```

3) Single screening (HTTP)
```bash
curl -s http://localhost:8001/infer \
  -H "content-type: application/json" \
  -d '{"title":"Curfew login blocker with ASL and GH for Utah minors",
       "description":"To comply with the Utah Social Media Regulation Act...",
       "docs":[]}'
```

4) Batch screening (CSV -> CSV)
```bash
docker compose exec gateway \
  python -m gateway.batch \
  --input /app/data/sample_dataset.csv \
  --output /app/outputs/run_outputs.csv

```
**Input CSV format:** `id,title,description,doc`
**Output CSV format:** `id,title,requires_geo_logic,reasoning,regulations`

---

## Features & functionality

* **RAG core:** Qdrant + HF embeddings to ground the LLM with targeted regulation notes.
* **Deterministic rule assists:** Explicit cues (NCMEC, “California + teens + PF”) set conservative defaults.
* **Evidence gating:** Regulations are kept only when retrieved with similarity + allowed by policy.
* **Ambiguity handling:** Returns `null` for inputs without legal intent and need human evaluation.
* **Canonicalization:** Normalizes free-form mentions (e.g., “EU DSA”, “Digital Services Act”) to `dsa`.
* **Auditability:** Logs Qdrant hits and final JSON; batch runner produces a signed-off CSV artifact.

---

## Development tools used

* **Docker & Docker Compose**
* **Python 3.10**
* **FastAPI** (service endpoints)
* **Uvicorn** (ASGI server)

## APIs used

* **Local REST**: `/health`, `/infer` (FastAPI)
* **Qdrant**: vector search (via `qdrant-client` and `langchain-qdrant`)
* **Ollama**: local LLM inference (via `langchain_community.chat_models.ChatOllama`)

## Libraries used

* **LangChain** (prompting, vector store integration, Pydantic parsing)
* **qdrant-client**, **langchain-qdrant**
* **sentence-transformers** (HF embeddings)
* **pydantic**, **PyYAML**, **httpx**

## Assets used

* **Synthetic regulation notes** in `data/regulations/` (one per `reg_id`)
* **Hackathon sample dataset**: `data/sample_dataset.csv`

---

## Mapping to the challenge

* **Boosting LLM precision:** RAG + canonicalization + evidence gating reduce hallucinations.
* **Full automation:** Batch runner + policy file + local stack = hands-off first pass; logs enable human-in-the-loop escalation.
* **Alternative detection:** (Future work) static/runtime analysis hooks could enrich signals, but out of scope per brief.

---

## License
This project is licensed under the **GNU General Public License v3.0**. You are free to modify, distribute, and use this project as long as the same license applies to any derivative work.
For more details, see the [LICENSE](LICENSE) file.
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

---