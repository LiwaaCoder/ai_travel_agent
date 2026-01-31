# Prompts Module

This folder contains documentation for the Travel Agent's prompt engineering strategy.

## Structure

```
prompts/
├── README.md              # This file
└── travel_prompts.md      # Detailed prompt documentation
```

## Implementation

The actual prompts are implemented in Python at `src/prompts/templates.py`:

```python
from src.prompts import PromptLibrary, PromptType

# Available prompt types
PromptType.INTENT_CLASSIFIER  # Route user requests
PromptType.RAG_RETRIEVAL      # Optimize queries for vector search
PromptType.RAG_SYNTHESIS      # General grounded responses
PromptType.ITINERARY_PLANNER  # Day-by-day trip plans
PromptType.INFO_RESPONDER     # Factual Q&A
PromptType.SAFETY_LAYER       # Compliance validation
```

## Usage in LangChain

```python
# Get LangChain-compatible ChatPromptTemplate
prompt = PromptLibrary.get_chat_prompt(PromptType.ITINERARY_PLANNER)

# Use in a chain
chain = prompt | llm | StrOutputParser()
result = await chain.ainvoke({"city": "Paris", "days": 3, ...})
```

## RAG Integration

The prompts are designed for RAG workflows:
1. **Query Optimization**: Transform user input for better retrieval
2. **Context Grounding**: Inject retrieved knowledge into prompts
3. **Source Citation**: Include provenance in responses
4. **Confidence Signaling**: Acknowledge knowledge gaps

See [travel_prompts.md](travel_prompts.md) for detailed documentation.
