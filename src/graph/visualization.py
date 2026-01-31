"""
Graph visualization and export utilities.

Provides tools for visualizing the LangGraph workflow structure.
"""

from typing import Optional
import json


def generate_mermaid_diagram() -> str:
    """
    Generate a Mermaid diagram of the Travel Agent graph.
    
    Returns:
        Mermaid diagram string for rendering.
    """
    return """
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4F46E5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#3730A3', 'lineColor': '#6366F1', 'secondaryColor': '#818CF8', 'tertiaryColor': '#C7D2FE'}}}%%
flowchart TD
    START([🚀 START]) --> A[🎯 classify_intent]
    A --> B[📚 retrieve_knowledge]
    B --> C[🌐 fetch_realtime_data]
    C --> D[🤖 synthesize_response]
    D --> END([✅ END])
    
    subgraph "Intent Classification"
        A
    end
    
    subgraph "RAG Pipeline"
        B
    end
    
    subgraph "External Tools"
        C
        C --> C1[🌤️ Weather API]
        C --> C2[📍 POI API]
    end
    
    subgraph "LLM Synthesis"
        D
    end
    
    style A fill:#4F46E5,stroke:#3730A3,color:#fff
    style B fill:#059669,stroke:#047857,color:#fff
    style C fill:#D97706,stroke:#B45309,color:#fff
    style D fill:#7C3AED,stroke:#6D28D9,color:#fff
```
"""


def generate_ascii_diagram() -> str:
    """
    Generate an ASCII representation of the graph.
    
    Returns:
        ASCII art diagram string.
    """
    return """
╔══════════════════════════════════════════════════════════════════╗
║                    AI TRAVEL AGENT GRAPH                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║    ┌─────────┐                                                   ║
║    │  START  │                                                   ║
║    └────┬────┘                                                   ║
║         │                                                        ║
║         ▼                                                        ║
║    ┌─────────────────────┐                                       ║
║    │  classify_intent    │  ← LLM: Determine user intent         ║
║    │  (plan/info/events) │                                       ║
║    └──────────┬──────────┘                                       ║
║               │                                                  ║
║               ▼                                                  ║
║    ┌─────────────────────┐                                       ║
║    │  retrieve_knowledge │  ← RAG: Query ChromaDB                ║
║    │  (vector search)    │    Retrieve relevant docs             ║
║    └──────────┬──────────┘                                       ║
║               │                                                  ║
║               ▼                                                  ║
║    ┌─────────────────────┐      ┌─────────────────┐              ║
║    │ fetch_realtime_data │ ───► │  Weather API    │              ║
║    │   (parallel calls)  │ ───► │  POI API (OSM)  │              ║
║    └──────────┬──────────┘      └─────────────────┘              ║
║               │                                                  ║
║               ▼                                                  ║
║    ┌─────────────────────┐                                       ║
║    │ synthesize_response │  ← LLM: Generate grounded answer      ║
║    │  (prompt routing)   │    Based on intent + context          ║
║    └──────────┬──────────┘                                       ║
║               │                                                  ║
║               ▼                                                  ║
║    ┌─────────┐                                                   ║
║    │   END   │  → TravelPlan(summary, pois, weather,             ║
║    └─────────┘              sources, confidence)                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def generate_graph_json() -> dict:
    """
    Generate a JSON representation of the graph structure.
    
    Returns:
        Dictionary describing nodes and edges.
    """
    return {
        "name": "TravelAgentGraph",
        "version": "1.0.0",
        "description": "RAG-powered travel planning with LangGraph",
        "nodes": [
            {
                "id": "classify_intent",
                "type": "llm",
                "description": "Classify user intent (plan/info/events/book)",
                "inputs": ["user_query"],
                "outputs": ["intent"],
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
            {
                "id": "retrieve_knowledge",
                "type": "retriever",
                "description": "Retrieve relevant travel knowledge from vector store",
                "inputs": ["city", "preferences"],
                "outputs": ["retrieved_context", "sources"],
                "vector_store": "ChromaDB",
                "embedding_model": "text-embedding-3-small",
                "top_k": 5,
            },
            {
                "id": "fetch_realtime_data",
                "type": "tool",
                "description": "Fetch real-time weather and POI data",
                "inputs": ["city", "preferences"],
                "outputs": ["weather_data", "poi_data"],
                "tools": ["weather_api", "poi_api"],
                "parallel": True,
            },
            {
                "id": "synthesize_response",
                "type": "llm",
                "description": "Generate grounded response from all context",
                "inputs": ["intent", "retrieved_context", "weather_data", "poi_data", "city", "days", "preferences"],
                "outputs": ["response", "confidence"],
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "prompt_routing": {
                    "plan": "ITINERARY_PLANNER",
                    "info": "INFO_RESPONDER",
                    "default": "RAG_SYNTHESIS",
                },
            },
        ],
        "edges": [
            {"from": "START", "to": "classify_intent"},
            {"from": "classify_intent", "to": "retrieve_knowledge"},
            {"from": "retrieve_knowledge", "to": "fetch_realtime_data"},
            {"from": "fetch_realtime_data", "to": "synthesize_response"},
            {"from": "synthesize_response", "to": "END"},
        ],
        "state_schema": {
            "city": "str",
            "days": "int",
            "preferences": "Optional[str]",
            "user_query": "str",
            "intent": "str",
            "retrieved_context": "list[str]",
            "weather_data": "str",
            "poi_data": "list[str]",
            "response": "str",
            "sources": "list[str]",
            "confidence": "float",
        },
    }


def export_graph_visualization(output_path: str, format: str = "mermaid") -> str:
    """
    Export graph visualization to a file.
    
    Args:
        output_path: Path to save the visualization
        format: Output format (mermaid, ascii, json)
    
    Returns:
        Path to the saved file
    """
    if format == "mermaid":
        content = generate_mermaid_diagram()
        ext = ".md"
    elif format == "ascii":
        content = generate_ascii_diagram()
        ext = ".txt"
    elif format == "json":
        content = json.dumps(generate_graph_json(), indent=2)
        ext = ".json"
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    filepath = f"{output_path}{ext}"
    with open(filepath, "w") as f:
        f.write(content)
    
    return filepath


def print_graph() -> None:
    """Print the graph structure to console."""
    print(generate_ascii_diagram())


if __name__ == "__main__":
    print_graph()
