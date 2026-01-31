from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.graph import build_travel_agent

app = FastAPI(title="AI Travel Agent")
agent = build_travel_agent()


class PlanRequest(BaseModel):
    city: str
    days: int = 3
    preferences: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan")
async def plan_trip(req: PlanRequest) -> dict:
    """Generate a simple plan via the agent scaffold."""
    result = await agent.plan_trip(req.city, req.days, req.preferences)
    return {"plan": result.summary, "pois": result.pois, "weather": result.weather}
