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
6. **Audit**: Store feedback in SQLite; batch runner emits CSV for records.

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

4) Seed retrieval (first run or when regs change)
```bash
docker compose exec retriever python -m retriever.seed
```

5) Try the UI

Open http://localhost:8001/static/index.html

Paste a feature → Run → (optionally) Send feedback

6) Single screening (HTTP)
```bash
curl -s http://localhost:8001/infer \
  -H "content-type: application/json" \
  -d '{"title":"Curfew login blocker with ASL and GH for Utah minors",
       "description":"To comply with the Utah Social Media Regulation Act...",
       "docs":[]}'
```

7) Batch screening (CSV -> CSV)
```bash
docker compose exec gateway \
  python -m gateway.batch \
  --input /app/data/sample_dataset.csv \
  --output /app/outputs/run_outputs.csv

```
**Input CSV format:** `id,title,description,doc`
**Output CSV format:** `id,title,requires_geo_logic,reasoning,regulations`

---

## Feedback & audit

* **POST /feedback** (Classifier)

```bash
curl -s http://localhost:8000/feedback \
  -H "content-type: application/json" \
  -d '{
    "feature_id":"feat-123",
    "title":"PF default toggle with NR enforcement for California teens",
    "description":"…",
    "docs":[],
    "requires_geo_logic": true,
    "regulations": ["california_kids_act"],
    "comment":"This is indeed CA minors + PF default off.",
    "user":"reviewerA"
  }'
```

* Feedback is stored in **`/app/outputs/audit.db`** (SQLite) inside the classifier container.
* Inspect it:

```bash
docker compose exec classifier sqlite3 /app/outputs/audit.db \
  ".schema feedback" \
  "SELECT ts, feature_id, requires_geo_logic, regulations_json, user FROM feedback ORDER BY ts DESC LIMIT 5;"
```

---

## Features & functionality

* **RAG core:** Qdrant + HF embeddings to ground the LLM with targeted regulation notes.
* **Deterministic rule assists:** Explicit cues (NCMEC, “California + teens + PF”).
* **Evidence gating:** Regulations are kept only when retrieved with similarity + allowed by policy.
* **Ambiguity handling:** Returns `null` for inputs without legal intent and needing human evaluation.
* **Canonicalization:** Normalizes free-form mentions to the canonical `reg_id`s.
* **Auditability & feedback:** `/feedback` endpoint writes to SQLite; UI includes a simple feedback flow.
* **Batch mode:** One-shot CSV screenings for large lists of features.

---

## Development tools used

* **Docker & Docker Compose**
* **Python 3.10**
* **FastAPI** (service endpoints)
* **Uvicorn** (ASGI server)

## APIs used

* **Classifier (FastAPI):** `/health`, `/infer`, `/feedback`
* **Qdrant:** vector search (`qdrant-client`, `langchain-qdrant`)
* **Ollama**: local LLM inference (`langchain_community.chat_models.ChatOllama`)

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
* **Full automation:** Batch runner + policy file + local stack = hands-off first pass; feedback + SQLite enable human-in-the-loop and future self-evolution.
* **User journey:** UI for screening and feedback; APIs for bulk and automation.

---

## License
This project is licensed under the **GNU General Public License v3.0**. You are free to modify, distribute, and use this project as long as the same license applies to any derivative work.
For more details, see the [LICENSE](LICENSE) file.
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

---