from pydantic import BaseModel
from typing import Optional, List


class PlanRequest(BaseModel):
    """Request model for trip planning."""
    city: str
    days: int = 3
    preferences: Optional[str] = None
    query: Optional[str] = None


class PlanResponse(BaseModel):
    """Response model for trip planning with RAG metadata."""
    plan: str
    pois: List[str]
    weather: str
    sources: List[str]
    confidence: float
