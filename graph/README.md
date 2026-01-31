# Graph Module

LangGraph workflow definitions and visualization utilities.

## Structure

```
graph/
├── README.md              # This file
src/graph/
├── __init__.py
└── visualization.py       # Graph visualization utilities
src/agents/
└── graph.py               # Main LangGraph workflow
```

## Workflow Architecture

```
START → classify_intent → retrieve_knowledge → fetch_realtime_data → synthesize_response → END
```

### Nodes

| Node | Type | Description |
|------|------|-------------|
| `classify_intent` | LLM | Classifies user intent (plan/info/events/book) |
| `retrieve_knowledge` | RAG | Retrieves relevant docs from ChromaDB |
| `fetch_realtime_data` | Tools | Parallel calls to Weather & POI APIs |
| `synthesize_response` | LLM | Generates grounded response |

## Visualization

```python
from src.graph import print_graph, generate_mermaid_diagram

# Print ASCII diagram
print_graph()

# Get Mermaid diagram for docs
mermaid = generate_mermaid_diagram()

# Export to file
from src.graph import export_graph_visualization
export_graph_visualization("graph_structure", format="json")
```

## State Schema

```python
class TravelAgentState(TypedDict):
    # Input
    city: str
    days: int
    preferences: Optional[str]
    user_query: str
    
    # Processing
    intent: str
    retrieved_context: list[str]
    weather_data: str
    poi_data: list[str]
    
    # Output
    response: str
    sources: list[str]
    confidence: float
```
