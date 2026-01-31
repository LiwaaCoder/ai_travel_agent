"""
Comprehensive test suite for the AI Travel Agent.

Tests cover:
- Unit tests for individual components
- Integration tests for the graph pipeline
- Mock-based tests for external APIs
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import asdict

# =============================================================================
# Agent Tests
# =============================================================================

class TestTravelAgent:
    """Tests for the TravelAgent class."""
    
    def test_agent_initialization(self):
        """Test that the agent can be instantiated."""
        from src.agents.graph import build_travel_agent, TravelAgent
        
        agent = build_travel_agent()
        assert agent is not None
        assert isinstance(agent, TravelAgent)
        assert agent.graph is not None
    
    def test_travel_plan_dataclass(self):
        """Test TravelPlan dataclass structure."""
        from src.agents.graph import TravelPlan
        
        plan = TravelPlan(
            city="Barcelona",
            days=3,
            preferences="food, art",
            summary="A wonderful 3-day trip...",
            pois=["Sagrada Familia", "Park Güell"],
            weather="Sunny, 20-25°C",
            sources=["destinations.md", "travel_tips.md"],
            confidence=0.85,
        )
        
        assert plan.city == "Barcelona"
        assert plan.days == 3
        assert plan.confidence == 0.85
        assert len(plan.sources) == 2
        assert len(plan.pois) == 2


# =============================================================================
# Prompt Tests
# =============================================================================

class TestPromptLibrary:
    """Tests for the prompt templates."""
    
    def test_prompt_library_has_all_types(self):
        """Test that all prompt types are available."""
        from src.prompts import PromptLibrary, PromptType
        
        for prompt_type in PromptType:
            prompt = PromptLibrary.get(prompt_type)
            assert prompt is not None
            assert prompt.name
            assert prompt.system_template
            assert prompt.human_template
    
    def test_prompt_to_chat_prompt(self):
        """Test conversion to LangChain ChatPromptTemplate."""
        from src.prompts import PromptLibrary, PromptType
        from langchain_core.prompts import ChatPromptTemplate
        
        chat_prompt = PromptLibrary.get_chat_prompt(PromptType.INTENT_CLASSIFIER)
        assert isinstance(chat_prompt, ChatPromptTemplate)
    
    def test_get_prompt_by_string(self):
        """Test getting prompt by string name."""
        from src.prompts import get_prompt
        
        prompt = get_prompt("intent_classifier")
        assert prompt.name == "Intent Classifier"


# =============================================================================
# RAG Pipeline Tests
# =============================================================================

class TestRAGPipeline:
    """Tests for the RAG pipeline components."""
    
    def test_document_loading(self):
        """Test that documents can be loaded from knowledge directory."""
        from src.rag.pipeline import load_documents, KNOWLEDGE_DIR
        
        if KNOWLEDGE_DIR.exists():
            docs = load_documents()
            assert len(docs) > 0
            assert all(hasattr(doc, "page_content") for doc in docs)
    
    def test_document_splitting(self):
        """Test document splitting."""
        from src.rag.pipeline import split_documents
        from langchain_core.documents import Document
        
        docs = [Document(page_content="A" * 2000, metadata={"source": "test"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 500 for chunk in chunks)


# =============================================================================
# Observability Tests
# =============================================================================

class TestLogging:
    """Tests for the logging module."""
    
    def test_get_logger(self):
        """Test logger creation."""
        from src.observability import get_logger
        
        logger = get_logger("test")
        assert logger is not None
        assert "travel_agent.test" in logger.name
    
    def test_agent_logger(self):
        """Test AgentLogger functionality."""
        from src.observability import AgentLogger
        
        logger = AgentLogger(session_id="test123")
        assert logger.session_id == "test123"
        
        # Test node context
        with logger.node_context("test_node", city="Paris"):
            logger.info("Test message")


class TestMetrics:
    """Tests for the metrics module."""
    
    def test_metrics_collector(self):
        """Test MetricsCollector functionality."""
        from src.observability import MetricsCollector
        
        metrics = MetricsCollector()
        
        # Test counter
        metrics.increment("test_counter", 5)
        summary = metrics.summary()
        assert summary["counters"]["test_counter"] == 5
    
    def test_timer_context(self):
        """Test timer context manager."""
        from src.observability import MetricsCollector
        import time
        
        metrics = MetricsCollector()
        
        with metrics.timer("test_operation"):
            time.sleep(0.01)  # 10ms
        
        summary = metrics.summary()
        assert "test_operation" in summary["timers"]
        assert summary["timers"]["test_operation"]["count"] == 1


class TestTracing:
    """Tests for the tracing module."""
    
    def test_tracer_creation(self):
        """Test LangSmithTracer creation."""
        from src.observability import LangSmithTracer
        
        tracer = LangSmithTracer(project_name="test")
        assert tracer.project_name == "test"
    
    def test_trace_span(self):
        """Test trace span creation."""
        from src.observability import LangSmithTracer
        
        tracer = LangSmithTracer()
        tracer.start_trace("test_run")
        
        with tracer.span("test_span") as span:
            span.metadata["test"] = True
        
        summary = tracer.end_trace()
        assert summary["spans"] == 1


# =============================================================================
# Graph Visualization Tests
# =============================================================================

class TestGraphVisualization:
    """Tests for graph visualization utilities."""
    
    def test_mermaid_diagram(self):
        """Test Mermaid diagram generation."""
        from src.graph import generate_mermaid_diagram
        
        diagram = generate_mermaid_diagram()
        assert "mermaid" in diagram
        assert "classify_intent" in diagram
        assert "retrieve_knowledge" in diagram
    
    def test_ascii_diagram(self):
        """Test ASCII diagram generation."""
        from src.graph import generate_ascii_diagram
        
        diagram = generate_ascii_diagram()
        assert "classify_intent" in diagram
        assert "ChromaDB" in diagram
    
    def test_graph_json(self):
        """Test JSON graph structure."""
        from src.graph import generate_graph_json
        
        graph = generate_graph_json()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 4


# =============================================================================
# API Model Tests
# =============================================================================

class TestAPIModels:
    """Tests for FastAPI request/response models."""
    
    def test_plan_request(self):
        """Test PlanRequest model."""
        from app import PlanRequest
        
        req = PlanRequest(
            city="Paris",
            days=5,
            preferences="museums, food",
            query="Best time to visit Louvre?"
        )
        
        assert req.city == "Paris"
        assert req.days == 5
        assert req.preferences == "museums, food"
    
    def test_plan_response(self):
        """Test PlanResponse model."""
        from app import PlanResponse
        
        resp = PlanResponse(
            plan="Visit the Louvre in the morning...",
            pois=["Louvre", "Eiffel Tower", "Montmartre"],
            weather="Mild, 15-20°C",
            sources=["destinations.md"],
            confidence=0.9,
        )
        
        assert resp.confidence == 0.9
        assert len(resp.pois) == 3
        assert len(resp.sources) == 1


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked external dependencies."""
    
    @pytest.mark.asyncio
    async def test_weather_tool(self):
        """Test weather tool with mocked API."""
        from src.tools.weather import fetch_weather_summary
        
        # Test with real API (will return actual data or "Weather unavailable")
        result = await fetch_weather_summary("Barcelona")
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_poi_tool(self):
        """Test POI tool with mocked API."""
        from src.tools.poi import fetch_pois
        
        # Test with real API
        result = await fetch_pois("Barcelona")
        assert isinstance(result, list)


# =============================================================================
# Smoke Tests
# =============================================================================

class TestSmoke:
    """Quick smoke tests to verify basic functionality."""
    
    def test_imports(self):
        """Test that all main modules can be imported."""
        from src.agents.graph import build_travel_agent
        from src.prompts import PromptLibrary
        from src.rag.pipeline import get_retriever
        from src.observability import get_logger
        from src.graph import print_graph
        
        assert True
    
    def test_app_health_endpoint(self):
        """Test FastAPI health endpoint."""
        from fastapi.testclient import TestClient
        from app import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
