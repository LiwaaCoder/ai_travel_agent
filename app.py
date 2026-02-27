"""
Compatibility wrapper to expose FastAPI app and models under module name `app`.
This keeps tests and external docs working when the main server lives in `server.py`.
"""

from server import app, PlanRequest, PlanResponse  # re-export for backward compatibility

from typing import List, Optional

__all__ = ["app", "PlanRequest", "PlanResponse"]
from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.graph import build_travel_agent

app = FastAPI(
    title="AI Travel Agent",
    description="RAG-powered travel planning using LangChain + LangGraph",
    version="1.0.0",
)
agent = build_travel_agent()


class PlanRequest(BaseModel):
    city: str
    days: int = 3
    preferences: Optional[str] = None
    query: Optional[str] = None


class PlanResponse(BaseModel):
    plan: str
    pois: List[str]
    weather: str
    sources: List[str]
    confidence: float


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
async def plan_trip(req: PlanRequest) -> PlanResponse:
    """
    Generate a RAG-grounded travel plan.
    
    The agent:
    1. Classifies intent (plan/info/events/book)
    2. Retrieves relevant knowledge from vector store
    3. Fetches real-time weather and POI data
    4. Synthesizes a grounded response using LLM
    """
    result = await agent.plan_trip(
        city=req.city,
        days=req.days,
        preferences=req.preferences,
        query=req.query,
    )
    return PlanResponse(
        plan=result.summary,
        pois=result.pois,
        weather=result.weather,
        sources=result.sources,
        confidence=result.confidence,
    )


if __name__ == "__main__":
    import uvicorn

    # Run using the `app` object directly to avoid name collision
    # with a package/directory named `app` in the workspace.
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
