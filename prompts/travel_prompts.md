# Travel Agent Prompt Library

> **Note**: This file documents the prompt strategies. The actual prompts are implemented in `src/prompts/templates.py` for use with LangChain.

## Overview

The Travel Agent uses a multi-stage prompting strategy optimized for RAG (Retrieval-Augmented Generation):

```
User Query → Intent Classification → Query Optimization → Retrieval → Synthesis → Safety Check → Response
```

## Prompt Categories

### 1. Intent Classifier
**Purpose**: Route user requests to the appropriate response strategy.

**Categories**:
| Intent | Description | Example |
|--------|-------------|---------|
| `plan` | Itinerary/activity requests | "Plan 3 days in Barcelona" |
| `info` | Factual questions | "Do I need a visa?" |
| `events` | Event/festival queries | "Any concerts this weekend?" |
| `book` | Booking requests (suggestions only) | "Book me a hotel" |

**Key Design Decisions**:
- Zero-shot classification with few-shot examples in prompt
- Temperature = 0 for deterministic routing
- Single-word output for reliable parsing

---

### 2. RAG Query Optimizer
**Purpose**: Transform user queries for better vector retrieval.

**Optimization Strategies**:
1. Expand abbreviations and colloquialisms
2. Add travel-related keywords
3. Include location-specific terms
4. Consider temporal aspects (seasons, times)
5. Add category hints (food, attractions, transport)

**Example**:
```
Input: "best food in Barcelona"
Output: 
- "Barcelona Spain local cuisine restaurants tapas paella food guide"
- "Barcelona culinary experiences traditional Catalan dishes where to eat"
- "Barcelona food markets La Boqueria local dining recommendations"
```

---

### 3. RAG Synthesis Prompt
**Purpose**: Generate grounded responses from retrieved context + real-time data.

**Grounding Rules**:
- ✅ Use information directly from retrieved context
- ✅ Cite sources naturally
- ✅ Acknowledge insufficient context
- ❌ Never invent prices, availability, hours
- ❌ Never claim certainty about dynamic info

**Input Variables**:
- `{context}` - Retrieved knowledge chunks
- `{weather}` - Real-time weather data
- `{pois}` - Points of interest from API
- `{city}`, `{days}`, `{preferences}`, `{query}` - User inputs

---

### 4. Itinerary Planner
**Purpose**: Create detailed day-by-day travel plans.

**Planning Principles**:
1. **Realistic Pacing**: 2-3 major activities per day max
2. **Geographic Logic**: Group nearby attractions
3. **Energy Flow**: High-energy AM, flexible PM
4. **Weather Awareness**: Indoor/outdoor balance
5. **Local Rhythm**: Respect siesta, late dinners, etc.
6. **Preference Alignment**: Match user preferences

**Output Structure**:
```markdown
## Day X: [Theme/Area]
- **Morning (time):** Activity - tips
- **Lunch:** Recommendation
- **Afternoon (time):** Activity - tips
- **Evening:** Activity or dining

## Practical Tips
- [From knowledge base]
```

---

### 5. Info Responder
**Purpose**: Answer factual travel questions.

**Response Style**:
1. Lead with direct answer
2. Support with details
3. Include practical tips
4. Mention source reliability

**Accuracy Rules**:
- Qualify dynamic info with "typically" or "as of knowledge base"
- Recommend verifying with official sources
- Distinguish facts from suggestions

---

### 6. Safety Layer
**Purpose**: Validate responses for compliance.

**Safety Checks**:
| Category | Action |
|----------|--------|
| Medical advice | Flag, suggest "consult a doctor" |
| Legal/visa specifics | Flag, provide general pointers only |
| Price guarantees | Ensure qualified as estimates |
| Availability claims | Remove certainty |
| Booking actions | Convert to suggestions |

---

## Implementation

All prompts are implemented as `PromptTemplate` dataclasses in `src/prompts/templates.py`:

```python
from src.prompts import PromptLibrary, PromptType

# Get a LangChain-compatible prompt
prompt = PromptLibrary.get_chat_prompt(PromptType.ITINERARY_PLANNER)

# Use in a chain
chain = prompt | llm | StrOutputParser()
response = await chain.ainvoke({"city": "Barcelona", ...})
```

## Adding New Prompts

1. Add new `PromptType` enum value
2. Create `PromptTemplate` with system + human templates
3. Register in `PromptLibrary._prompts`
4. Document in this file
