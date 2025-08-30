from sqlalchemy import create_engine, text
import os, json

DB_URL = os.getenv("DB_URL", "sqlite:////app/data/audit.db")

def get_engine():
    return create_engine(DB_URL, future=True)

def ensure_schema():
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            docs TEXT,
            requires_geo_logic TEXT,
            reasoning TEXT,
            confidence REAL,
            regulations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))

def upsert_record(rec: dict):
    eng = get_engine()
    with eng.begin() as con:
        con.execute(text("""
            INSERT INTO audit_log (id,title,description,docs,requires_geo_logic,reasoning,confidence,regulations)
            VALUES (:id,:title,:description,:docs,:requires_geo_logic,:reasoning,:confidence,:regulations)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              description=excluded.description,
              docs=excluded.docs,
              requires_geo_logic=excluded.requires_geo_logic,
              reasoning=excluded.reasoning,
              confidence=excluded.confidence,
              regulations=excluded.regulations,
        """), rec)
