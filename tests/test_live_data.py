"""
Tests for live data tools (flights, events) and database caching.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


# =============================================================================
# Database Tests
# =============================================================================

class TestDatabase:
    """Tests for the SQLite caching layer."""
    
    def test_database_initialization(self):
        """Test that database initializes correctly."""
        from src.database import get_connection, init_db
        
        init_db()
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "flights" in tables
        assert "events" in tables
        conn.close()
    
    def test_cache_validity(self):
        """Test TTL-based cache validation."""
        from src.database import is_cache_valid
        
        # Recent timestamp should be valid
        recent = datetime.now().isoformat()
        assert is_cache_valid(recent, ttl_seconds=3600) is True
        
        # Old timestamp should be invalid
        old = (datetime.now() - timedelta(hours=2)).isoformat()
        assert is_cache_valid(old, ttl_seconds=3600) is False
    
    def test_flights_cache_operations(self):
        """Test saving and retrieving flights from cache."""
        from src.database import save_flights, get_cached_flights, clear_cache
        
        clear_cache()
        
        test_flights = [
            {
                "airline": "Test Air",
                "flight_number": "TA123",
                "origin": "London",
                "departure_time": "2024-01-15T10:00:00",
                "arrival_time": "2024-01-15T14:00:00",
                "price": 199,
                "currency": "USD",
            }
        ]
        
        save_flights("barcelona", test_flights)
        cached = get_cached_flights("barcelona")
        
        assert len(cached) == 1
        assert cached[0]["airline"] == "Test Air"
        assert cached[0]["flight_number"] == "TA123"
        
        clear_cache()
    
    def test_events_cache_operations(self):
        """Test saving and retrieving events from cache."""
        from src.database import save_events, get_cached_events, clear_cache
        
        clear_cache()
        
        test_events = [
            {
                "name": "Barcelona vs Real Madrid",
                "venue": "Camp Nou",
                "event_date": "2024-01-20T20:00:00",
                "teams": ["Barcelona", "Real Madrid"],
                "league": "La Liga",
            }
        ]
        
        save_events("barcelona", test_events, "football")
        cached = get_cached_events("barcelona", "football")
        
        assert len(cached) == 1
        assert cached[0]["name"] == "Barcelona vs Real Madrid"
        assert cached[0]["venue"] == "Camp Nou"
        assert cached[0]["teams"] == ["Barcelona", "Real Madrid"]
        
        clear_cache()


# =============================================================================
# Flights Tool Tests
# =============================================================================

class TestFlightsTool:
    """Tests for the flights data tool."""
    
    def test_city_to_iata(self):
        """Test city name to IATA code conversion."""
        from src.tools.flights import city_to_iata
        
        assert city_to_iata("Barcelona") == "BCN"
        assert city_to_iata("tokyo") == "TYO"
        assert city_to_iata("UnknownCity") is None
    
    @pytest.mark.asyncio
    async def test_fetch_flights_returns_list(self):
        """Test that fetch_flights returns a list."""
        from src.tools.flights import fetch_flights
        
        result = await fetch_flights("Barcelona")
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_fetch_flights_mock_fallback(self):
        """Test that flights tool returns mock data when API unavailable."""
        from src.tools.flights import fetch_flights
        from src.database import clear_cache
        
        clear_cache()
        
        # Without API key, should return mock data
        result = await fetch_flights("Barcelona")
        
        assert len(result) > 0
        assert "airline" in result[0]
        assert "flight_number" in result[0]
    
    def test_format_flights_for_prompt(self):
        """Test flight data formatting for LLM prompt."""
        from src.tools.flights import format_flights_for_prompt
        
        flights = [
            {
                "airline": "Test Air",
                "flight_number": "TA123",
                "origin": "London",
                "departure_time": "2024-01-15T10:00:00",
                "price": 199,
            }
        ]
        
        formatted = format_flights_for_prompt(flights)
        assert "Test Air" in formatted
        assert "TA123" in formatted
        assert "London" in formatted
        assert "$199" in formatted
    
    def test_format_flights_empty(self):
        """Test formatting with no flights."""
        from src.tools.flights import format_flights_for_prompt
        
        formatted = format_flights_for_prompt([])
        assert "No flight information" in formatted


# =============================================================================
# Events Tool Tests
# =============================================================================

class TestEventsTool:
    """Tests for the events data tool."""
    
    @pytest.mark.asyncio
    async def test_fetch_events_returns_list(self):
        """Test that fetch_events returns a list."""
        from src.tools.events import fetch_events
        
        result = await fetch_events("Barcelona")
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_fetch_events_mock_fallback(self):
        """Test that events tool returns mock data when API unavailable."""
        from src.tools.events import fetch_events
        from src.database import clear_cache
        
        clear_cache()
        
        result = await fetch_events("Barcelona")
        
        assert len(result) > 0
        assert "name" in result[0]
        assert "venue" in result[0]
    
    def test_format_events_for_prompt(self):
        """Test event data formatting for LLM prompt."""
        from src.tools.events import format_events_for_prompt
        
        events = [
            {
                "name": "Barcelona vs Real Madrid",
                "venue": "Camp Nou",
                "event_date": "2024-01-20T20:00:00",
                "league": "La Liga",
            }
        ]
        
        formatted = format_events_for_prompt(events)
        assert "Barcelona vs Real Madrid" in formatted
        assert "Camp Nou" in formatted
        assert "La Liga" in formatted
    
    def test_format_events_empty(self):
        """Test formatting with no events."""
        from src.tools.events import format_events_for_prompt
        
        formatted = format_events_for_prompt([])
        assert "No upcoming events" in formatted


# =============================================================================
# Integration Tests
# =============================================================================

class TestLiveDataIntegration:
    """Integration tests for live data with the agent."""
    
    @pytest.mark.asyncio
    async def test_agent_state_includes_live_data(self):
        """Test that agent state includes flight and event data fields."""
        from src.agents.graph import TravelAgentState
        
        # Verify the TypedDict has the new fields
        annotations = TravelAgentState.__annotations__
        assert "flight_data" in annotations
        assert "event_data" in annotations
    
    @pytest.mark.asyncio
    async def test_fetch_all_data_includes_live_sources(self):
        """Test that fetch_all_data fetches flights and events."""
        from src.agents.graph import fetch_all_data
        
        state = {
            "city": "Barcelona",
            "days": 3,
            "preferences": "football, food",
            "user_query": "",
        }
        
        result = await fetch_all_data(state)
        
        assert "flight_data" in result
        assert "event_data" in result
        assert isinstance(result["flight_data"], list)
        assert isinstance(result["event_data"], list)
