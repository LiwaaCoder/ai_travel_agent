"""Test configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
    )
    return client


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Barcelona is a beautiful city known for Gaudí architecture."
    mock_doc.metadata = {"filename": "destinations.md", "source": "data/knowledge/destinations.md"}
    retriever.invoke.return_value = [mock_doc]
    return retriever


@pytest.fixture
def sample_travel_state():
    """Sample state for testing graph nodes."""
    return {
        "city": "Barcelona",
        "days": 3,
        "preferences": "food, architecture",
        "user_query": "Plan a 3-day trip to Barcelona",
        "intent": "",
        "retrieved_context": [],
        "weather_data": "",
        "poi_data": [],
        "response": "",
        "sources": [],
        "confidence": 0.0,
    }


@pytest.fixture
def mock_weather_response():
    """Mock weather API response."""
    return "Sunny, temperature range: 18-25°C"


@pytest.fixture
def mock_pois_response():
    """Mock POI API response."""
    return ["Sagrada Familia", "Park Güell", "La Rambla", "Gothic Quarter", "Casa Batlló"]
