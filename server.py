from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()  # Load .env file before importing any OpenAI-dependent modules

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

    # Use import string with renamed module (server.py) to allow reload
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
