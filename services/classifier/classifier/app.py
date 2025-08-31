import os
import logging
from typing import List, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from fastapi.responses import FileResponse
import sqlite3, time
import hashlib

from classifier.chain import RAGClassifier

app = FastAPI()
rag = None

# Input schema
class InferIn(BaseModel):
    feature_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    docs: List[str] = []

# Output schema
class InferOut(BaseModel):
    feature_id: str
    requires_geo_logic: Optional[bool]
    reasoning: str
    confidence: float
    regulations: List[str]

def _ensure_outputs_dir():
    os.makedirs("/app/outputs", exist_ok=True)

def _compute_feature_id(title, description, docs):
    raw = f"{title or ''}\n{description or ''}\n" + "\n".join(docs or [])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _fetch_latest_feedback(feature_id: str) -> dict | None:
    try:
        con = sqlite3.connect("/app/outputs/audit.db")
        cur = con.cursor()
        cur.execute("""
            SELECT requires_geo_logic, regulations_json, comment, ts
            FROM feedback
            WHERE feature_id = ?
            ORDER BY ts DESC
            LIMIT 1
        """, (feature_id,))
        row = cur.fetchone()
        con.close()
        if not row: 
            return None
        requires_geo_logic, regulations_json, comment, ts = row
        regs = json.loads(regulations_json or "[]")
        return {"requires_geo_logic": json.loads(requires_geo_logic), "regulations": regs, "comment": comment, "ts": ts}
    except Exception:
        return None
    
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/infer", response_model=InferOut)
def infer(inp: InferIn) -> InferOut:
    global rag
    if rag is None:
        rag = RAGClassifier()

    try:       
        feature_hash = rag._feature_id_for(inp.title, inp.description or "", inp.docs or [])
        out = rag.run(inp.title, inp.description, inp.docs)
        out["feature_id"] = feature_hash
        if inp.feature_id:
            fb = _fetch_latest_feedback(inp.feature_id)
            if fb:
                # Overwrite with human feedback, but keep model's reasoning for traceability
                out["requires_geo_logic"] = fb["requires_geo_logic"]
                out["regulations"] = fb["regulations"]
                out["reasoning"] = (out.get("reasoning") or "") + f" [nudged by feedback @ {fb['ts']}]"
    except Exception as e:
        logging.exception("infer failed")
        out = {
            "feature_id": "",  # or feature_hash if available
            "requires_geo_logic": None,
            "reasoning": f"Unhandled error in classifier: {e}",
            "confidence": 0.0,
            "regulations": [],
        }
    return InferOut(**out)

class Feedback(BaseModel):
    feature_id: str
    title: str = ""
    description: str = ""
    docs: list[str] = []
    requires_geo_logic: bool | None
    regulations: list[str] = []
    comment: str = ""
    user: str = "anon"

@app.post("/feedback")
def save_feedback(fb: Feedback):
    fid = (fb.feature_id or "").strip() or _compute_feature_id(fb.title, fb.description, fb.docs)
    con = sqlite3.connect("/app/outputs/audit.db")
    cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS feedback(
        ts INTEGER, feature_id TEXT, title TEXT, description TEXT,
        docs_json TEXT, requires_geo_logic TEXT, regulations_json TEXT,
        comment TEXT, user TEXT
      )
    """)
    cur.execute("""INSERT INTO feedback VALUES(?,?,?,?,?,?,?,?,?)""", (
        int(time.time()), fid, fb.title, fb.description,
        json.dumps(fb.docs), json.dumps(fb.requires_geo_logic),
        json.dumps(fb.regulations), fb.comment, fb.user
    ))
    con.commit(); con.close()
    return {"ok": True}

@app.get("/audit/recent")
def audit_recent(limit: int = 10):
    _ensure_outputs_dir()
    con = sqlite3.connect("/app/outputs/audit.db")
    cur = con.cursor()
    cur.execute("""
      SELECT ts, feature_id, requires_geo_logic, regulations_json, comment, user
      FROM feedback ORDER BY ts DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    con.close()
    out = []
    for ts, fid, rgl, regs_json, comment, user in rows:
        out.append({
            "ts": ts,
            "feature_id": fid,
            "requires_geo_logic": json.loads(rgl),
            "regulations": json.loads(regs_json or "[]"),
            "comment": comment,
            "user": user
        })
    return {"items": out}

# static UI
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/")
def ui_root():
    return FileResponse("/app/static/index.html")