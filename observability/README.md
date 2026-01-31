# Observability Module

Comprehensive logging, tracing, and metrics for the AI Travel Agent.

## Components

```
src/observability/
├── __init__.py
├── logger.py      # Structured logging with JSON/colored output
├── tracing.py     # LangSmith integration & distributed tracing
└── metrics.py     # Performance metrics collection
```

## Quick Start

### Logging

```python
from src.observability import get_logger, AgentLogger

# Simple logger
logger = get_logger("my_module")
logger.info("Processing request")

# Agent logger with context
agent_logger = AgentLogger(session_id="abc123")
with agent_logger.node_context("retrieve_knowledge", city="Barcelona"):
    agent_logger.info("Retrieved 5 documents")
```

### Tracing (LangSmith)

```python
from src.observability import LangSmithTracer

tracer = LangSmithTracer(project_name="travel-agent")
tracer.start_trace("plan_trip")

with tracer.span("retrieve_knowledge") as span:
    span.metadata["num_docs"] = 5
    # ... operation ...

print(tracer.get_trace_url())  # LangSmith URL
```

### Metrics

```python
from src.observability import MetricsCollector

metrics = MetricsCollector()

with metrics.timer("llm_call"):
    # ... LLM operation ...

metrics.increment("documents_retrieved", 5)
print(metrics.summary())
```

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | Output format ("json" for production) |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing ("true") |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |

## Output Formats

### Development (Colored)
```
[   INFO] 10:30:45 [retrieve_knowledge] Retrieved 5 documents (23ms)
```

### Production (JSON)
```json
{"timestamp": "2024-01-15T10:30:45Z", "level": "INFO", "node": "retrieve_knowledge", "message": "Retrieved 5 documents", "latency_ms": 23}
```
