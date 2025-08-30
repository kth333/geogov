import os, csv, asyncio
from fastapi import FastAPI, HTTPException
from .models import Feature, ScreenResult
from .client import infer, seed_retriever
from .db import ensure_schema, upsert_record

app = FastAPI(title="GeoGov Gateway")

@app.on_event("startup")
async def startup():
    ensure_schema()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/seed")
async def seed():
    return await seed_retriever()

@app.post("/screen_feature", response_model=ScreenResult)
async def screen_feature(feat: Feature):
    res = await infer(feat.dict(exclude={"id"}))
    record = {
        "id": feat.id,
        "title": feat.title,
        "description": feat.description,
        "docs": "\n".join(feat.docs or []),
        "requires_geo_logic": "null" if res.get("requires_geo_logic") is None else str(bool(res["requires_geo_logic"])).lower(),
        "reasoning": res.get("reasoning",""),
        "confidence": float(res.get("confidence",0.0)),
        "regulations": str(res.get("regulations", [])),
    }
    upsert_record(record)
    return {
        "id": feat.id,
        **res
    }
