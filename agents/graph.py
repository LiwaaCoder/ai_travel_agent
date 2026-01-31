"""
Legacy agents module - imports from src.agents.graph for backward compatibility.
"""

from src.agents.graph import (
    TravelAgentState,
    TravelPlan,
    TravelAgent,
    build_travel_agent,
    build_travel_graph,
)

__all__ = [
    "TravelAgentState",
    "TravelPlan", 
    "TravelAgent",
    "build_travel_agent",
    "build_travel_graph",
]
