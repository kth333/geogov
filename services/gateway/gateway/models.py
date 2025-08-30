from pydantic import BaseModel
from typing import List, Optional

class Feature(BaseModel):
    id: str
    title: str
    description: str
    docs: List[str] = []

class ScreenResult(BaseModel):
    id: str
    requires_geo_logic: Optional[bool]
    reasoning: str
    confidence: float
    regulations: list