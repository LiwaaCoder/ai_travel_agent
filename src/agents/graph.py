"""
LangGraph Travel Agent: Optimized for speed with parallel execution.

Optimizations:
1. Skip intent classification (default to "plan") - saves 1 LLM call
2. Run RAG retrieval + real-time data in parallel
3. Cache retriever instance
4. Reduced token limits and timeouts
5. Streamlined prompts
"""

from typing import TypedDict, Optional
from dataclasses import dataclass
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from src.rag.pipeline import get_retriever
from src.tools.weather import fetch_weather_summary
from src.tools.poi import fetch_pois
from src.prompts import PromptLibrary, PromptType


# ============================================================================
# Cached Resources (avoid recreating on each request)
# ============================================================================

_cached_retriever = None
_cached_llm = None


def get_cached_retriever():
    """Get or create cached retriever."""
    global _cached_retriever
    if _cached_retriever is None:
        _cached_retriever = get_retriever(k=3, score_threshold=0.3)
    return _cached_retriever


def get_cached_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Get or create cached LLM instance."""
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=2000,  # Increased for detailed responses
            timeout=30,
            request_timeout=30,
        )
    return _cached_llm


# ============================================================================
# State Definition
# ============================================================================

class TravelAgentState(TypedDict):
    """State passed between nodes in the travel agent graph."""
    # Input
    city: str
    days: int
    preferences: Optional[str]
    user_query: str
    
    # Processing
    intent: str  # "plan", "info", "events", "book"
    retrieved_context: list[str]
    weather_data: str
    poi_data: list[str]
    
    # Output
    response: str
    sources: list[str]
    confidence: float


# ============================================================================
# Graph Nodes (Optimized)
# ============================================================================

def create_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Create a ChatOpenAI instance optimized for latency."""
    return get_cached_llm(temperature)


async def classify_intent(state: TravelAgentState) -> dict:
    """Node: Fast intent classification without LLM call."""
    # Skip LLM call - infer intent from query keywords for speed
    query = (state.get("user_query") or "").lower()
    
    if any(w in query for w in ["book", "reserve", "ticket"]):
        intent = "book"
    elif any(w in query for w in ["event", "concert", "festival", "show"]):
        intent = "events"
    elif "?" in query or any(w in query for w in ["what", "how", "when", "where", "is", "do", "can"]):
        intent = "info"
    else:
        intent = "plan"
    
    return {"intent": intent}


async def fetch_all_data(state: TravelAgentState) -> dict:
    """Node: Fetch RAG + real-time data IN PARALLEL for speed."""
    
    async def get_rag_context():
        try:
            retriever = get_cached_retriever()
            query = f"{state['city']} {state.get('preferences', '')}"
            docs = await asyncio.to_thread(retriever.invoke, query)
            context = [doc.page_content for doc in docs]
            sources = list(set(doc.metadata.get("filename", "kb") for doc in docs))
            return context, sources
        except Exception:
            return [], []
    
    async def get_weather():
        try:
            return await asyncio.wait_for(fetch_weather_summary(state["city"]), timeout=5)
        except Exception:
            return "Weather unavailable"
    
    async def get_pois():
        try:
            return await asyncio.wait_for(fetch_pois(state["city"], state.get("preferences")), timeout=5)
        except Exception:
            return []
    
    # Run ALL three in parallel
    (context, sources), weather, pois = await asyncio.gather(
        get_rag_context(),
        get_weather(),
        get_pois(),
    )
    
    return {
        "retrieved_context": context,
        "sources": sources,
        "weather_data": weather,
        "poi_data": pois,
    }


async def synthesize_response(state: TravelAgentState) -> dict:
    """Node: Generate professional travel agent response using LLM + RAG."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    city = state["city"]
    days_count = state["days"]
    pois = state.get("poi_data", [])[:12]
    weather = state.get("weather_data", "Pleasant Mediterranean weather expected")
    prefs = state.get("preferences") or "culture, food, and authentic local experiences"
    context = state.get("retrieved_context", [])
    user_query = state.get("user_query", "")
    
    logger.info(f"Synthesizing response for {city}, {days_count} days")
    logger.info(f"POIs found: {len(pois)}")
    logger.info(f"RAG context chunks: {len(context)}")
    
    # Build rich context from RAG knowledge base
    if context:
        rag_knowledge = "\n\n".join([f"📚 {c}" for c in context])
    else:
        rag_knowledge = f"""No specific knowledge base entries found for {city}. 
Use your general knowledge about this destination, but be clear about what you know vs. recommend researching."""
    
    # Build POI list with numbering
    if pois:
        poi_formatted = "\n".join([f"  {i+1}. {poi}" for i, poi in enumerate(pois)])
    else:
        poi_formatted = "  - Major landmarks and attractions in the city center"
    
    # ==========================================================================
    # PROFESSIONAL TRAVEL AGENT SYSTEM PROMPT
    # ==========================================================================
    system_prompt = """You are Liwaa, an award-winning travel consultant with 18 years of experience crafting bespoke travel experiences. You've personally visited over 80 countries and have deep local knowledge.

YOUR PERSONALITY:
- Warm, enthusiastic, and genuinely passionate about travel
- You speak like a trusted friend who happens to be a travel expert
- You share insider tips, local secrets, and personal anecdotes
- You're specific, never generic - you name actual places, dishes, streets
- You anticipate questions and address logistics proactively

YOUR EXPERTISE:
- You know the best times to visit attractions to avoid crowds
- You understand local customs, tipping culture, and etiquette
- You recommend restaurants like a local, not a tourist guide
- You balance must-see spots with hidden gems
- You consider weather, walking distances, and realistic timing

RESPONSE STYLE:
- Enthusiastic but professional
- Specific and actionable (names, addresses, times)
- Include "pro tips" and "insider secrets" naturally
- Vary your language - don't repeat phrases
- End each day with something memorable"""

    # ==========================================================================
    # DETAILED USER PROMPT WITH RAG CONTEXT
    # ==========================================================================
    user_prompt = f"""I need you to create an INCREDIBLE {days_count}-day itinerary for {city}!

═══════════════════════════════════════════════════════════════
📋 TRIP DETAILS
═══════════════════════════════════════════════════════════════
🌍 Destination: {city}
📅 Duration: {days_count} days
❤️ Traveler Interests: {prefs}
❓ Special Request: {user_query if user_query else "None - create the best possible trip!"}

═══════════════════════════════════════════════════════════════
🌤️ WEATHER FORECAST
═══════════════════════════════════════════════════════════════
{weather}

═══════════════════════════════════════════════════════════════
📍 VERIFIED ATTRACTIONS & POINTS OF INTEREST
═══════════════════════════════════════════════════════════════
{poi_formatted}

═══════════════════════════════════════════════════════════════
📚 LOCAL KNOWLEDGE BASE (Use this for authentic recommendations!)
═══════════════════════════════════════════════════════════════
{rag_knowledge}

═══════════════════════════════════════════════════════════════
✍️ YOUR TASK
═══════════════════════════════════════════════════════════════
Create a detailed day-by-day itinerary using this EXACT format:

## Day 1: [Creative Evocative Title - e.g., "Gaudí's Dreamworld & Gothic Treasures"]

- **Morning (9:00 AM):** [SPECIFIC attraction name] — [2-3 sentences: what makes it special, what to look for, and one insider tip like "arrive 15 min before opening" or "the view from the east side is best for photos"]

- **Lunch (12:30 PM):** [SPECIFIC neighborhood or street] — [Recommend actual local dishes by name, e.g., "Try the patatas bravas and pan con tomate at a tapas bar in El Born." Mention what makes the food culture here unique]

- **Afternoon (2:30 PM):** [SPECIFIC attraction or activity] — [Why this fits well after lunch, what to see/do, time needed, and practical tip]

- **Evening (7:00 PM):** [SPECIFIC dinner recommendation or activity] — [Type of cuisine, neighborhood vibe, and one signature dish or experience to try. Maybe suggest a sunset spot or evening activity]

💡 **Pro Tip for Day 1:** [One golden piece of advice for this day]

---

## Day 2: [Next Theme Title]
[Continue same detailed format...]

---

[Continue for all {days_count} days]

---

## 🎒 Practical Tips for Your Trip

**Getting Around:** [Specific transport advice]
**Money-Saving Secrets:** [1-2 genuine tips]  
**Cultural Etiquette:** [Important local customs]
**Best Photo Spots:** [2-3 specific locations]

---

## 💬 Liwaa's Final Thoughts

[A warm, personal 2-3 sentence sign-off with your best piece of advice for making this trip unforgettable]

═══════════════════════════════════════════════════════════════
IMPORTANT REMINDERS:
- Use REAL place names from the POI list and knowledge base
- Be SPECIFIC about dishes, streets, neighborhoods
- Include TIMING tips (when to go, how long to spend)
- Make it feel PERSONAL, like advice from a well-traveled friend
- Each day should have a logical FLOW (geography, energy levels)
═══════════════════════════════════════════════════════════════

Now, create this amazing itinerary!"""

    llm = get_cached_llm()
    
    try:
        logger.info("Calling LLM for synthesis...")
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        itinerary = response.content
        logger.info(f"LLM response received: {len(itinerary)} characters")
        confidence = 0.9 if context else 0.75
        
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        # Fallback to basic template if LLM fails
        itinerary = _generate_fallback_itinerary(city, days_count, pois, weather)
        confidence = 0.5
    
    return {
        "response": itinerary,
        "confidence": confidence,
    }


def _generate_fallback_itinerary(city: str, days: int, pois: list, weather: str) -> str:
    """Fallback template if LLM fails."""
    lines = [f"# Your {days}-Day Adventure in {city}\n"]
    lines.append(f"*Note: Generating detailed itinerary... Please try again if this appears basic.*\n")
    
    pois_per_day = max(2, len(pois) // days) if pois else 0
    
    for d in range(1, days + 1):
        start_idx = (d-1) * pois_per_day
        day_pois = pois[start_idx : start_idx + pois_per_day] if pois else []
        
        lines.append(f"## Day {d}: Exploring {city}")
        lines.append(f"- **Morning (9:00 AM):** Visit {day_pois[0] if day_pois else 'the historic center'}")
        lines.append(f"- **Lunch (12:30 PM):** Enjoy local cuisine in the neighborhood")
        if len(day_pois) > 1:
            lines.append(f"- **Afternoon (2:00 PM):** Discover {day_pois[1]}")
        else:
            lines.append(f"- **Afternoon (2:00 PM):** Explore local shops and cafés")
        lines.append(f"- **Evening (7:00 PM):** Dinner at a local restaurant")
        lines.append("")
    
    lines.append(f"\n**Current Weather:** {weather}")
    return "\n".join(lines)



# ============================================================================
# Graph Construction (Optimized: 2 nodes instead of 4)
# ============================================================================

def build_travel_graph() -> StateGraph:
    """
    Build optimized LangGraph workflow.
    
    Optimized structure (2 steps instead of 4):
    START -> classify_intent (no LLM) -> fetch_all_data (parallel) -> synthesize_response -> END
    """
    workflow = StateGraph(TravelAgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("fetch_all_data", fetch_all_data)
    workflow.add_node("synthesize_response", synthesize_response)
    
    # Define edges (streamlined flow)
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "fetch_all_data")
    workflow.add_edge("fetch_all_data", "synthesize_response")
    workflow.add_edge("synthesize_response", END)
    
    return workflow.compile()


# ============================================================================
# Agent Interface (backward compatible)
# ============================================================================

@dataclass
class TravelPlan:
    """Result from the travel agent."""
    city: str
    days: int
    preferences: Optional[str]
    summary: str
    pois: list[str]
    weather: str
    sources: list[str]
    confidence: float


class TravelAgent:
    """
    RAG-powered Travel Agent using LangGraph.
    
    This agent:
    1. Classifies user intent
    2. Retrieves relevant knowledge from vector store
    3. Fetches real-time weather and POI data
    4. Synthesizes a grounded response using LLM
    """
    
    def __init__(self):
        self.graph = build_travel_graph()
    
    async def plan_trip(
        self,
        city: str,
        days: int,
        preferences: Optional[str] = None,
        query: Optional[str] = None,
    ) -> TravelPlan:
        """
        Generate a travel plan using the RAG pipeline.
        
        Args:
            city: Destination city
            days: Trip duration in days
            preferences: User preferences (food, art, nightlife, etc.)
            query: Optional specific query
        
        Returns:
            TravelPlan with grounded recommendations
        """
        initial_state: TravelAgentState = {
            "city": city,
            "days": days,
            "preferences": preferences,
            "user_query": query or f"Plan a {days}-day trip to {city}",
            "intent": "",
            "retrieved_context": [],
            "weather_data": "",
            "poi_data": [],
            "response": "",
            "sources": [],
            "confidence": 0.0,
        }
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return TravelPlan(
            city=city,
            days=days,
            preferences=preferences,
            summary=result["response"],
            pois=result["poi_data"],
            weather=result["weather_data"],
            sources=result["sources"],
            confidence=result["confidence"],
        )


def build_travel_agent() -> TravelAgent:
    """Factory function to create a TravelAgent instance."""
    return TravelAgent()
