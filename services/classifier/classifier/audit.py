import os, json, time, sqlite3, hashlib
from typing import Tuple, List, Optional

DB_PATH = os.getenv("AUDIT_DB", "/app/outputs/audit.db")

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def feature_hash(title: str, description: str, docs: List[str]) -> str:
    blob = (title or "") + "\n" + (description or "") + "\n".join(docs or [])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

def ensure_tables():
    con = _conn(); cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS decisions(
        ts INTEGER,
        feature_hash TEXT,
        title TEXT,
        description TEXT,
        docs_json TEXT,
        evidence_json TEXT,
        raw_text TEXT,
        requires_geo_logic TEXT,
        confidence REAL,
        regulations_json TEXT,
        model TEXT,
        version TEXT
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS feedback(
        ts INTEGER,
        feature_hash TEXT,
        title TEXT,
        description TEXT,
        docs_json TEXT,
        requires_geo_logic TEXT,
        regulations_json TEXT,
        comment TEXT,
        user TEXT
      )
    """)
    con.commit(); con.close()

def log_decision(feature_hash: str, title: str, description: str, docs: List[str],
                 evidence: list, raw_text: str, requires_geo_logic, confidence: float,
                 regulations: List[str], model: str, version: str = "v1") -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("""INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (
        int(time.time()),
        feature_hash,
        title or "",
        description or "",
        json.dumps(docs or []),
        json.dumps(evidence or []),
        raw_text or "",
        json.dumps(requires_geo_logic),
        float(confidence or 0.0),
        json.dumps(regulations or []),
        model or "",
        version,
    ))
    con.commit(); con.close()

def log_feedback(feature_hash: str, title: str, description: str, docs: List[str],
                 requires_geo_logic, regulations: List[str], comment: str, user: str="anon") -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("""INSERT INTO feedback VALUES(?,?,?,?,?,?,?,?,?)""", (
        int(time.time()),
        feature_hash,
        title or "",
        description or "",
        json.dumps(docs or []),
        json.dumps(requires_geo_logic),
        json.dumps(regulations or []),
        comment or "",
        user or "anon",
    ))
    con.commit(); con.close()

def latest_feedback(feature_hash: str) -> Tuple[Optional[bool], List[str]]:
    con = _conn(); cur = con.cursor()
    cur.execute("""SELECT requires_geo_logic, regulations_json
                   FROM feedback WHERE feature_hash=?
                   ORDER BY ts DESC LIMIT 1""", (feature_hash,))
    row = cur.fetchone()
    con.close()
    if not row: return None, []
    try:
        rgl = json.loads(row[0]) if row[0] else None
        regs = json.loads(row[1]) if row[1] else []
        return rgl, regs
    except Exception:
        return None, []