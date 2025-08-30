import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from classifier.chain import RAGClassifier

app = FastAPI()
rag = None

# Input schema
class InferIn(BaseModel):
    title: str
    description: Optional[str] = None
    docs: List[str] = []

# Output schema
class InferOut(BaseModel):
    requires_geo_logic: Optional[bool]
    reasoning: str
    confidence: float
    regulations: List[str]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/infer", response_model=InferOut)
def infer(inp: InferIn) -> InferOut:
    global rag
    if rag is None:
        rag = RAGClassifier()

    try:
        out = rag.run(inp.title, inp.description, inp.docs)
    except Exception as e:
        logging.exception("infer failed")
        out = {
            "requires_geo_logic": None,
            "reasoning": f"Unhandled error in classifier: {e}",
            "confidence": 0.0,
            "regulations": [],
        }
    return InferOut(**out)