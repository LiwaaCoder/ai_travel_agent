"""
Prompt templates for the RAG-powered Travel Agent.

This module provides structured, reusable prompts for each stage of the LangGraph workflow.
All prompts are designed to work with LangChain's ChatPromptTemplate.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class PromptType(str, Enum):
    """Types of prompts available in the library."""
    INTENT_CLASSIFIER = "intent_classifier"
    RAG_RETRIEVAL = "rag_retrieval"
    RAG_SYNTHESIS = "rag_synthesis"
    ITINERARY_PLANNER = "itinerary_planner"
    INFO_RESPONDER = "info_responder"
    SAFETY_LAYER = "safety_layer"


@dataclass
class PromptTemplate:
    """A structured prompt template with metadata."""
    name: str
    description: str
    system_template: str
    human_template: str
    input_variables: list[str]
    
    def to_chat_prompt(self) -> ChatPromptTemplate:
        """Convert to LangChain ChatPromptTemplate."""
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template(self.human_template),
        ])


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

INTENT_CLASSIFIER = PromptTemplate(
    name="Intent Classifier",
    description="Classifies user intent to route the workflow appropriately",
    system_template="""You are an expert intent classifier for an AI travel planning assistant.
Your task is to analyze the user's request and classify it into exactly ONE category.

CATEGORIES:
1. **plan** - User wants a travel itinerary, day-by-day activities, or trip planning
   Examples: "Plan 3 days in Barcelona", "What should I do in Paris?", "Create a weekend itinerary"

2. **info** - User asks factual questions about destinations, logistics, tips, or practical matters
   Examples: "Do I need a visa for Japan?", "What's the best metro card in London?", "Is tap water safe?"

3. **events** - User wants information about events, concerts, festivals, exhibitions, or happenings
   Examples: "Any concerts this weekend?", "What festivals are in May?", "Museum exhibitions?"

4. **book** - User requests booking actions (you will NOT book, only provide suggestions)
   Examples: "Book me a hotel", "Find flights", "Reserve a table"

RULES:
- Output ONLY one word: plan, info, events, or book
- If uncertain, default to "info" for questions or "plan" for requests
- Consider the primary intent if multiple intents are present""",
    human_template="{query}",
    input_variables=["query"],
)


# =============================================================================
# RAG RETRIEVAL QUERY OPTIMIZATION
# =============================================================================

RAG_RETRIEVAL = PromptTemplate(
    name="RAG Query Optimizer",
    description="Optimizes user query for better vector store retrieval",
    system_template="""You are a query optimization expert for a travel knowledge base.
Your task is to reformulate the user's query to maximize retrieval of relevant travel information.

OPTIMIZATION STRATEGIES:
1. Expand abbreviations and colloquialisms
2. Add relevant travel-related keywords
3. Include location-specific terms
4. Consider temporal aspects (seasons, times)
5. Add category hints (food, attractions, transport, safety)

OUTPUT: Return 2-3 optimized search queries, one per line.
Keep each query concise but information-rich.""",
    human_template="""Original query: {query}
Destination: {city}
Preferences: {preferences}

Generate optimized retrieval queries:""",
    input_variables=["query", "city", "preferences"],
)


# =============================================================================
# RAG SYNTHESIS (General)
# =============================================================================

RAG_SYNTHESIS = PromptTemplate(
    name="RAG Synthesis",
    description="Synthesizes grounded responses from retrieved context",
    system_template="""You are an expert travel assistant with access to a curated knowledge base.
Your responses must be GROUNDED in the provided context - never fabricate information.

GROUNDING RULES:
1. ✅ Use information directly from the retrieved context
2. ✅ Cite sources naturally (e.g., "Based on local tips...", "According to travel guides...")
3. ✅ Acknowledge when context is insufficient: "I don't have specific information about..."
4. ❌ Never invent prices, availability, hours, or real-time information
5. ❌ Never claim certainty about dynamic information (events, weather beyond forecast)

RESPONSE QUALITY:
- Be concise but comprehensive
- Prioritize actionable information
- Include practical tips from the knowledge base
- Mention data freshness concerns when relevant

SAFETY COMPLIANCE:
- Do not provide medical, legal, or visa advice beyond general pointers
- Encourage checking official sources for critical information
- Flag booking requests as suggestions only""",
    human_template="""RETRIEVED CONTEXT:
{context}

REAL-TIME DATA:
- Weather: {weather}
- Points of Interest: {pois}

USER REQUEST:
- City: {city}
- Duration: {days} days
- Preferences: {preferences}
- Query: {query}

Provide a grounded, helpful response:""",
    input_variables=["context", "weather", "pois", "city", "days", "preferences", "query"],
)


# =============================================================================
# ITINERARY PLANNER (for "plan" intent)
# =============================================================================

ITINERARY_PLANNER = PromptTemplate(
    name="Itinerary Planner",
    description="Creates detailed day-by-day travel itineraries",
    system_template="""You are an expert travel itinerary planner creating personalized trip plans.
Your itineraries must be practical, well-paced, and grounded in the provided knowledge.

PLANNING PRINCIPLES:
1. **Realistic Pacing**: 2-3 major activities per day maximum; account for travel time
2. **Geographic Logic**: Group nearby attractions on the same day to minimize transit
3. **Energy Flow**: High-energy activities in morning, flexible time in afternoon
4. **Weather Awareness**: Adjust indoor/outdoor balance based on forecast
5. **Local Rhythm**: Respect local customs (siesta times, late dinners, etc.)
6. **Preference Alignment**: Prioritize activities matching user preferences

STRUCTURE:
- Each day should have a theme or focus area
- Include specific timing blocks (morning, lunch, afternoon, evening)
- Add practical tips from the knowledge base
- Suggest alternatives for flexibility
- Note any required bookings or reservations

GROUNDING:
- Base recommendations on retrieved knowledge
- Cite sources when providing specific tips
- Acknowledge gaps: "Consider researching..." for areas outside knowledge base""",
    human_template="""GROUNDED KNOWLEDGE:
{context}

REAL-TIME DATA:
- Weather Forecast: {weather}
- Available Attractions: {pois}

TRIP DETAILS:
- Destination: {city}
- Duration: {days} days
- Traveler Preferences: {preferences}

Create a detailed, practical itinerary:

## Day-by-Day Plan

[Format each day as:]
## Day X: [Theme/Area]
- **Morning (time):** Activity - why it fits, tips
- **Lunch:** Recommendation with reasoning
- **Afternoon (time):** Activity - why it fits, tips  
- **Evening:** Activity or dining suggestion

## Practical Tips
[Include relevant tips from the knowledge base]

## Notes
[Any caveats, booking requirements, or suggestions for research]""",
    input_variables=["context", "weather", "pois", "city", "days", "preferences"],
)


# =============================================================================
# INFO RESPONDER (for "info" intent)
# =============================================================================

INFO_RESPONDER = PromptTemplate(
    name="Info Responder",
    description="Answers factual travel questions with grounded information",
    system_template="""You are a knowledgeable travel information assistant.
Your role is to provide accurate, factual answers grounded in the retrieved context.

RESPONSE STYLE:
1. Lead with a direct answer to the question
2. Follow with supporting details and context
3. Include practical tips if relevant
4. Mention source reliability and freshness concerns

ACCURACY RULES:
- Only state facts present in the retrieved context
- For prices/hours/availability: qualify with "typically" or "as of knowledge base"
- Recommend verifying dynamic information with official sources
- Clearly distinguish between facts and suggestions

FORMAT:
- Start with a concise 1-2 sentence answer
- Add bullet points for detailed information
- Include a "Pro tip" if the knowledge base has relevant advice
- End with any necessary caveats or verification suggestions""",
    human_template="""RETRIEVED CONTEXT:
{context}

REAL-TIME DATA:
- Weather: {weather}
- Local Attractions: {pois}

USER QUESTION about {city}:
{query}

Provide an accurate, grounded answer:""",
    input_variables=["context", "weather", "pois", "city", "query"],
)


# =============================================================================
# SAFETY LAYER
# =============================================================================

SAFETY_LAYER = PromptTemplate(
    name="Safety Layer",
    description="Validates and sanitizes responses for safety compliance",
    system_template="""You are a safety and compliance reviewer for a travel assistant.
Review the draft response and flag or modify any concerning content.

SAFETY CHECKS:
1. **No Medical Advice**: Flag specific medical recommendations beyond "consult a doctor"
2. **No Legal Advice**: Flag visa/legal specifics beyond general pointers
3. **No Price Guarantees**: Ensure prices are qualified as estimates
4. **No Availability Claims**: Ensure no claims about current availability
5. **No Booking Actions**: Ensure booking requests are responded with suggestions only
6. **Source Attribution**: Encourage checking official sources for critical info

ACTIONS:
- If safe: Return "SAFE: [original response]"
- If needs modification: Return "MODIFIED: [corrected response]"
- If problematic: Return "FLAGGED: [explanation and safe alternative]" """,
    human_template="""Draft response to review:
{response}

Original user query:
{query}

Review for safety compliance:""",
    input_variables=["response", "query"],
)


# =============================================================================
# PROMPT LIBRARY
# =============================================================================

class PromptLibrary:
    """Central registry of all prompts for the Travel Agent."""
    
    _prompts: dict[str, PromptTemplate] = {
        PromptType.INTENT_CLASSIFIER: INTENT_CLASSIFIER,
        PromptType.RAG_RETRIEVAL: RAG_RETRIEVAL,
        PromptType.RAG_SYNTHESIS: RAG_SYNTHESIS,
        PromptType.ITINERARY_PLANNER: ITINERARY_PLANNER,
        PromptType.INFO_RESPONDER: INFO_RESPONDER,
        PromptType.SAFETY_LAYER: SAFETY_LAYER,
    }
    
    @classmethod
    def get(cls, prompt_type: PromptType) -> PromptTemplate:
        """Get a prompt template by type."""
        return cls._prompts[prompt_type]
    
    @classmethod
    def get_chat_prompt(cls, prompt_type: PromptType) -> ChatPromptTemplate:
        """Get a LangChain ChatPromptTemplate by type."""
        return cls._prompts[prompt_type].to_chat_prompt()
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all available prompt types."""
        return [p.value for p in PromptType]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_prompt(prompt_type: str | PromptType) -> PromptTemplate:
    """Get a prompt template by name or type."""
    if isinstance(prompt_type, str):
        prompt_type = PromptType(prompt_type)
    return PromptLibrary.get(prompt_type)


def get_all_prompts() -> dict[str, PromptTemplate]:
    """Get all available prompts."""
    return {pt.value: PromptLibrary.get(pt) for pt in PromptType}
