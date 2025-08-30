from pydantic import BaseModel, Field
from typing import List, Optional

class RegulationHit(BaseModel):
    reg_id: str
    snippet: str
    score: float  # similarity in [0..1]

class GeoDecision(BaseModel):
    requires_geo_logic: Optional[bool] = Field(None, description="true/false/null if ambiguous")
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    regulations: List[str] = Field(default_factory=list)  # canonical IDs

class InferenceRequest(BaseModel):
    title: str
    description: Optional[str] = None
    docs: List[str] = []

class InferenceResponse(BaseModel):
    requires_geo_logic: Optional[bool]
    reasoning: str
    confidence: float
    regulations: List[RegulationHit]