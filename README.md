# AI Travel Agent (Liwaa)

[![CI](https://github.com/LiwaaCoder/ai_travel_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LiwaaCoder/ai_travel_agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)](pyproject.toml)

Professional, RAG-powered travel planning assistant that generates grounded, personalized itineraries using LangChain and LangGraph. Tailored for demos, portfolios, and as a foundation for production deployment.

## 🌟 Highlights

- RAG-backed recommendations (ChromaDB) to ground LLM outputs
- LangGraph workflow for robust orchestration (intent → retrieval → tools → synthesis)
- LangChain for embeddings, prompts, and LLM orchestration
- Real-time weather (Open-Meteo) and POI (OpenStreetMap) integration
- Streamlit UI + FastAPI + CLI for multi-surface demos
- Persona-driven assistant (`Liwaa`) with insider tips and practical guidance

## 🏗️ Architecture (high level)

The app accepts a trip request (UI / API / CLI), retrieves relevant knowledge from the vector store, fetches real-time tools (weather, POIs), and synthesizes a grounded itinerary via the LLM.

```
╔══════════════════════════════════════════════════════════════════╗
║                    AI TRAVEL AGENT ARCHITECTURE                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            ║
║  │  FastAPI    │   │   Typer     │   │  Streamlit  │            ║
║  │   /plan     │   │    CLI      │   │     UI      │            ║
║  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            ║
║         │                 │                 │                    ║
║         └─────────────────┼─────────────────┘                    ║
║                           │                                      ║
║                           ▼                                      ║
║         ┌─────────────────────────────────────┐                 ║
║         │         LangGraph Workflow           │                 ║
║         │  ┌─────────────────────────────────┐ │                 ║
║         │  │     classify_intent (LLM)       │ │                 ║
║         │  └──────────────┬──────────────────┘ │                 ║
║         │                 │                    │                 ║
║         │  ┌──────────────▼──────────────────┐ │                 ║
║         │  │   retrieve_knowledge (RAG)      │ │                 ║
║         │  │   └─► ChromaDB Vector Store     │ │                 ║
║         │  └──────────────┬──────────────────┘ │                 ║
║         │                 │                    │                 ║
║         │  ┌──────────────▼──────────────────┐ │                 ║
║         │  │   fetch_realtime_data (Tools)   │ │                 ║
║         │  │   ├─► Weather API               │ │                 ║
║         │  │   └─► POI API (OpenStreetMap)   │ │                 ║
║         │  └──────────────┬──────────────────┘ │                 ║
║         │                 │                    │                 ║
║         │  ┌──────────────▼──────────────────┐ │                 ║
║         │  │   synthesize_response (LLM)     │ │                 ║
║         │  │   └─► Prompt Library (6 types)  │ │                 ║
║         │  └─────────────────────────────────┘ │                 ║
║         └─────────────────────────────────────┘                 ║
║                           │                                      ║
║                           ▼                                      ║
║         ┌─────────────────────────────────────┐                 ║
║         │  TravelPlan(summary, pois, weather, │                 ║
║         │            sources, confidence)     │                 ║
║         └─────────────────────────────────────┘                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

## 🚀 Quickstart

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp env.sample .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Build Knowledge Base
```bash
# Ingest travel knowledge into vector store
python cli.py build-kb

# Or force rebuild
python cli.py build-kb --force
```

### 3. Run the Agent

**CLI:**
```bash
python cli.py plan --city "Barcelona" --days 3 --preferences "food, art"
```

**API Server:**
```bash
uvicorn server:app --reload
# Then POST to http://localhost:8000/plan
```

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

### 4. Run Tests
```bash
pytest -v
```

## 📁 Project Structure

```
├── server.py                  # FastAPI application (main entry)
├── app.py                     # Thin wrapper re-exporting `server.app`
├── cli.py                     # Typer CLI with rich formatting
├── streamlit_app.py           # Streamlit web interface
├── models.py                  # Pydantic request/response models
│
├── src/
│   ├── agents/
│   │   └── graph.py           # 🔷 LangGraph workflow & TravelAgent class
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py       # 📝 Prompt library (6 prompt types)
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py        # 📚 Document loading, embedding, retrieval
│   ├── graph/
│   │   ├── __init__.py
│   │   └── visualization.py   # 📊 Graph visualization utilities
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logger.py          # 📋 Structured logging
│   │   ├── tracing.py         # 🔍 LangSmith integration
│   │   └── metrics.py         # 📈 Performance metrics
│   └── tools/
│       ├── weather.py         # 🌤️ Open-Meteo weather API
│       └── poi.py             # 📍 Overpass/OSM POI API
│
├── data/
│   └── knowledge/             # 📖 Markdown knowledge base documents
│       ├── travel_tips.md
│       └── destinations.md
│
├── prompts/
│   └── travel_prompts.md      # Prompt documentation
│
├── vector_db/                 # ChromaDB persistent storage
│
└── tests/
    ├── conftest.py            # Pytest fixtures
    ├── test_smoke.py          # Basic smoke tests
    └── test_agent.py          # Comprehensive test suite
```

## 🔧 Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `OPENAI_API_KEY` | Required. Your OpenAI API key |
| `TRAVEL_AGENT_DEFAULT_CITY` | Optional. Default city for UI |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | "json" for production logging |
| `LANGCHAIN_TRACING_V2` | "true" to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |

## 📚 How It Works

1. **Intent Classification**: LLM classifies user request (plan/info/events/book)
2. **Knowledge Retrieval**: ChromaDB retrieves relevant travel knowledge chunks
3. **Real-time Data**: Fetches weather forecast and POIs from external APIs
4. **Response Synthesis**: LLM combines retrieved context + real-time data into grounded recommendations
5. **Confidence Scoring**: Response confidence based on retrieval quality and data availability

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | **LangGraph** - Stateful workflow graphs |
| LLM Framework | **LangChain** - Chains, prompts, embeddings |
| Vector Store | **ChromaDB** - Persistent embeddings storage |
| LLM Provider | **OpenAI** - GPT-4o-mini, text-embedding-3-small |
| API Framework | **FastAPI** - REST API with async support |
| CLI Framework | **Typer + Rich** - Beautiful CLI formatting |
| Web UI | **Streamlit** - Interactive web interface |
| Observability | **LangSmith** - Tracing & debugging |

## 📝 Adding Knowledge

Add markdown files to `data/knowledge/`, then rebuild:
```bash
python cli.py build-kb --force
```

The RAG pipeline will automatically chunk, embed, and index the new content.

## 🔍 Observability

### View Graph Structure
```python
from src.graph import print_graph
print_graph()
```

### Enable LangSmith Tracing
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
export LANGCHAIN_PROJECT=travel-agent
```

### Structured Logging
```python
from src.observability import AgentLogger

logger = AgentLogger(session_id="request-123")
with logger.node_context("retrieve_knowledge", city="Paris"):
    logger.info("Retrieved 5 documents")
```

## 📄 License

MIT
